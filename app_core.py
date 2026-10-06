# app_core.py —— 应用级基础设施：配置/缓存持久化 + 运行事件出口

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from collections import deque

# ===============================================================
#  一、目录与文件路径
# ===============================================================

_ENV_HOME = "ARKNIGHT_AUTO_EDITING_HOME"
_APP_DIR_NAME = "arknight-auto-editing"

CONFIG_FILE = "config.json"
GPU_PROFILE_FILE = "gpu_profile.json"
BENCH_FILE = "bench.json"

_lock = threading.RLock()
_home_cache: str | None = None


def _env_home() -> str | None:
    raw = (os.environ.get(_ENV_HOME) or "").strip()
    if not raw:
        return None
    try:
        return os.path.abspath(os.path.expanduser(raw))
    except Exception:
        return None


def base_dir() -> str:
    """程序所在目录（冻结包为 exe 目录）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def bundle_dir() -> str:
    """PyInstaller onefile 解包目录（未冻结时等同程序目录）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.abspath(meipass)
    return base_dir()


def _candidates() -> list[str]:
    out: list[str] = []
    env = _env_home()
    if env:
        out.append(env)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if local:
        out.append(os.path.join(local, _APP_DIR_NAME))
    out.append(os.path.join(base_dir(), "config"))
    out.append(os.path.join(tempfile.gettempdir(), _APP_DIR_NAME))
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        k = os.path.normcase(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def _writable(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".write-test")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def app_home() -> str:
    """可写的配置根目录（首次调用做一次写测试并缓存）。"""
    global _home_cache
    with _lock:
        if _home_cache:
            return _home_cache
        for cand in _candidates():
            if _writable(cand):
                _home_cache = cand
                return cand
        _home_cache = tempfile.gettempdir()
        return _home_cache


def set_home(path: str | None) -> None:
    """覆盖配置根目录（测试用；None 恢复自动探测）。"""
    global _home_cache
    with _lock:
        _home_cache = os.path.abspath(path) if path else None


# ===============================================================
#  二、JSON 原子读写
# ===============================================================

def _path(name: str) -> str:
    return os.path.join(app_home(), name)


def _read_json(name: str) -> dict:
    try:
        with open(_path(name), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, UnicodeDecodeError):
        return {}


def _write_json(name: str, data: dict) -> bool:
    p = _path(name)
    tmp = None
    try:
        with _lock:
            fd, tmp = tempfile.mkstemp(
                prefix=name + ".", suffix=".tmp", dir=os.path.dirname(p) or ".")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
            tmp = None
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ===============================================================
#  三、配置 / 探测缓存 / 测速缓存
# ===============================================================

_config_cache: dict | None = None


def load_config(force: bool = False) -> dict:
    global _config_cache
    with _lock:
        if _config_cache is None or force:
            _config_cache = _read_json(CONFIG_FILE)
        return dict(_config_cache)


def save_config(cfg: dict | None = None) -> bool:
    global _config_cache
    with _lock:
        if cfg is not None:
            _config_cache = dict(cfg)
        data = dict(_config_cache or {})
    return _write_json(CONFIG_FILE, data)


def get(key: str, default=None):
    with _lock:
        return load_config().get(key, default)


def set_value(key: str, value) -> bool:
    global _config_cache
    with _lock:
        cfg = load_config()
        if cfg.get(key) == value:
            return True
        cfg[key] = value
        _config_cache = cfg
    return save_config()


def update_config(**kw) -> bool:
    global _config_cache
    with _lock:
        cfg = load_config()
        cfg.update(kw)
        _config_cache = cfg
    return save_config()


def load_gpu_profile() -> dict:
    return _read_json(GPU_PROFILE_FILE)


def save_gpu_profile(profile: dict) -> bool:
    return _write_json(GPU_PROFILE_FILE, profile)


def load_bench() -> dict:
    return _read_json(BENCH_FILE)


def save_bench(bench: dict) -> bool:
    return _write_json(BENCH_FILE, bench)


def clear_caches() -> None:
    for name in (GPU_PROFILE_FILE, BENCH_FILE):
        p = _path(name)
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


# ===============================================================
#  四、运行事件
#
#  冻结为 windowed exe 后没有 stdout，降级/回退原因若只 print 就完全不可见，
#  因此统一走这里：环形缓冲留历史，订阅者即时收到（UI 自行切回主线程）。
# ===============================================================

LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"

_LEVEL_LABEL = {LEVEL_INFO: "信息", LEVEL_WARN: "降级", LEVEL_ERROR: "错误"}
_MAX_EVENTS = 300

_events: deque = deque(maxlen=_MAX_EVENTS)
_callbacks: list = []
_echo_to_stdout = True


def set_echo(enabled: bool) -> None:
    global _echo_to_stdout
    with _lock:
        _echo_to_stdout = bool(enabled)


def report(level: str, message: str, source: str = "") -> dict:
    level = level if level in _LEVEL_LABEL else LEVEL_INFO
    event = {"level": level, "message": str(message),
             "source": str(source or ""), "time": time.time()}
    with _lock:
        _events.append(event)
        callbacks = list(_callbacks)
        echo = _echo_to_stdout
    if echo:
        try:
            prefix = f"[{_LEVEL_LABEL.get(level, level)}]"
            if source:
                prefix += f"[{source}]"
            print(f"{prefix} {message}", flush=True)
        except Exception:
            pass
    for cb in callbacks:
        try:
            cb(event)
        except Exception:
            pass
    return event


def info(message: str, source: str = "") -> dict:
    return report(LEVEL_INFO, message, source)


def warn(message: str, source: str = "") -> dict:
    return report(LEVEL_WARN, message, source)


def error(message: str, source: str = "") -> dict:
    return report(LEVEL_ERROR, message, source)


def register(callback) -> None:
    with _lock:
        if callback not in _callbacks:
            _callbacks.append(callback)


def unregister(callback) -> None:
    with _lock:
        if callback in _callbacks:
            _callbacks.remove(callback)


def last_events(limit: int = 60) -> list[dict]:
    with _lock:
        items = list(_events)
    return items[-limit:] if limit and limit > 0 else items


def has_error() -> bool:
    with _lock:
        return any(e["level"] == LEVEL_ERROR for e in _events)


def clear() -> None:
    with _lock:
        _events.clear()


def format_event(event: dict, with_time: bool = True) -> str:
    level = _LEVEL_LABEL.get(event.get("level"), event.get("level", ""))
    ts = ""
    if with_time:
        try:
            ts = time.strftime("%H:%M:%S", time.localtime(float(event.get("time", 0))))
        except Exception:
            ts = "--:--:--"
    src = event.get("source") or ""
    head = f"{ts} [{level}]" if ts else f"[{level}]"
    return f"{head} {src + ': ' if src else ''}{event.get('message', '')}"


def history_text(limit: int = 60) -> str:
    events = last_events(limit)
    return "\n".join(format_event(e) for e in events) if events else "（暂无运行事件）"
