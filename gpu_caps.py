# gpu_caps.py —— GPU / FFmpeg 能力探测、编码参数、解码变体、可选 CUDA 加速包
#
# 所有能力都靠实测得出，而不是查表：不同 ffmpeg 构建与驱动下，
# 列表里存在的编码器/加速方式实际上可能初始化失败。
#
# 厂商路径：
#   NVIDIA  解码 cuda(NVDEC) / 缩放 scale_cuda→scale_npp→swscale / 编码 *_nvenc
#   AMD     解码 d3d11va→dxva2 / 缩放 swscale / 编码 *_amf
#   Intel   解码 qsv / 缩放 scale_qsv→vpp_qsv→swscale / 编码 *_qsv
#   兜底    软件解码 + swscale + libx264

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field, asdict

import cv2

import app_core

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# ===============================================================
#  编码器清单
# ===============================================================

# 顺序 = 能力优先级（新编码器在前，便于用户手动挑更高压缩率）
ENCODER_ORDER = [
    "av1_nvenc", "hevc_nvenc", "h264_nvenc",
    "av1_qsv", "hevc_qsv", "h264_qsv",
    "av1_amf", "hevc_amf", "h264_amf",
    "h264_videotoolbox", "hevc_videotoolbox",
]

VENDOR_OF_ENCODER = {
    "av1_nvenc": "nvidia", "hevc_nvenc": "nvidia", "h264_nvenc": "nvidia",
    "av1_qsv": "intel", "hevc_qsv": "intel", "h264_qsv": "intel",
    "av1_amf": "amd", "hevc_amf": "amd", "h264_amf": "amd",
    "h264_videotoolbox": "apple", "hevc_videotoolbox": "apple",
}

VENDOR_LABEL = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel",
                "apple": "Apple", "none": "无硬件加速"}

# 「自动」选默认编码器时的偏好：h264 兼容性最好，其次 hevc、av1。
# 列表顺序仍按压缩率排，供用户手动选择。
DEFAULT_ENCODER_PREFERENCE = [
    "h264_nvenc", "hevc_nvenc", "av1_nvenc",
    "h264_qsv", "hevc_qsv", "av1_qsv",
    "h264_amf", "hevc_amf", "av1_amf",
    "h264_videotoolbox", "hevc_videotoolbox",
]

_PRESET_SETS = {
    "nvenc": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    "qsv": ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
    "amf": ["speed", "balanced", "quality"],
    "x26x": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
             "slow", "slower", "veryslow"],
}

# ===============================================================
#  FFmpeg 定位与信息
# ===============================================================

_PROBE_LOCK = threading.Lock()          # 探测串行化，避免并发抢 GPU 造成假阴性
_INFO_LOCK = threading.Lock()
_FFMPEG_INFO_CACHE: dict[str, dict] = {}
_ENCODER_CACHE: dict[str, bool] = {}
_PROFILE_LOCK = threading.Lock()
_PROFILE_MEM: dict | None = None

_BUNDLED_FFMPEG_NAMES = ("ffmpeg.exe", "ffmpeg") if sys.platform == "win32" else ("ffmpeg",)


def resolve_ffmpeg_path(ffmpeg_path: str | None = None) -> str:
    """显式路径 → 内置(exe 目录/解包目录) → PATH → imageio_ffmpeg。"""
    if ffmpeg_path:
        p = os.path.expanduser(str(ffmpeg_path).strip())
        if p and os.path.isfile(p):
            return os.path.abspath(p)
        found = shutil.which(p) if p else None
        if found:
            return found
        raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")

    for base in (app_core.base_dir(), app_core.bundle_dir()):
        for name in _BUNDLED_FFMPEG_NAMES:
            cand = os.path.join(base, name)
            if os.path.isfile(cand):
                return os.path.abspath(cand)

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return os.path.abspath(exe)
    except Exception:
        pass

    raise FileNotFoundError(
        "未找到 FFmpeg：程序目录、PATH、imageio_ffmpeg 都没有。"
        "请在「基本」页指定 ffmpeg.exe 路径（硬件编解码需要完整构建，"
        "例如 Gyan/BtbN 的 full/gpl 版本）。"
    )


def _run(args: list[str], timeout: float = 20.0, text: bool = True):
    return subprocess.run(
        args, check=False, capture_output=True, timeout=timeout,
        text=text, encoding="utf-8" if text else None,
        errors="replace" if text else None,
        creationflags=_NO_WINDOW,
    )


def ffmpeg_info(ffmpeg_path: str | None = None, force: bool = False) -> dict:
    """{path, version, hwaccels, encoders_text}（按路径缓存）。"""
    exe = resolve_ffmpeg_path(ffmpeg_path)
    key = os.path.normcase(exe)
    with _INFO_LOCK:
        cached = _FFMPEG_INFO_CACHE.get(key)
    if cached is not None and not force:
        return cached

    info = {"path": exe, "version": "", "hwaccels": [], "encoders_text": "", "error": ""}
    try:
        res = _run([exe, "-hide_banner", "-version"], timeout=15)
        first = (res.stdout or "").splitlines()
        info["version"] = (first[0].strip() if first else "")[:200]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    try:
        res = _run([exe, "-hide_banner", "-hwaccels"], timeout=15)
        names = []
        for line in (res.stdout or "").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("hardware acceleration"):
                continue
            names.append(line)
        info["hwaccels"] = names
    except Exception as exc:
        info["error"] = info["error"] or f"{type(exc).__name__}: {exc}"

    try:
        res = _run([exe, "-hide_banner", "-encoders"], timeout=20)
        info["encoders_text"] = res.stdout or ""
    except Exception as exc:
        info["error"] = info["error"] or f"{type(exc).__name__}: {exc}"

    with _INFO_LOCK:
        _FFMPEG_INFO_CACHE[key] = info
    return info


def _encoder_listed(info: dict, enc: str) -> bool:
    text = info.get("encoders_text") or ""
    return bool(text) and re.search(rf"^\s*\S+\s+{re.escape(enc)}\s", text, re.MULTILINE) is not None


def has_encoder(enc: str, ffmpeg_path: str | None = None) -> bool:
    try:
        return _encoder_listed(ffmpeg_info(ffmpeg_path), enc)
    except FileNotFoundError:
        return False


# ===============================================================
#  编码器实测
# ===============================================================

def encoder_probe_args(enc: str) -> list[str]:
    if enc.endswith("_nvenc"):
        return ["-c:v", enc, "-preset", "p4", "-cq", "24"]
    if enc.endswith("_qsv"):
        return ["-c:v", enc, "-global_quality", "24"]
    if enc.endswith("_amf"):
        return ["-c:v", enc, "-usage", "transcoding", "-quality", "speed",
                "-rc", "cqp", "-qp_i", "24", "-qp_p", "24"]
    if enc.endswith("_videotoolbox"):
        return ["-c:v", enc, "-q:v", "50"]
    return ["-c:v", enc]


def probe_encoder(enc: str, ffmpeg_path: str | None = None,
                  timeout: float = 12.0, use_cache: bool = True) -> bool:
    """实测编码器能否初始化。

    只有「ffmpeg 正常退出并报错」才缓存 False；超时/驱动初始化等瞬时失败
    不缓存，下次重试。
    """
    try:
        exe = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        return False
    cache_key = f"{enc}|{os.path.normcase(exe)}"
    with _PROBE_LOCK:
        if use_cache and cache_key in _ENCODER_CACHE:
            return _ENCODER_CACHE[cache_key]
        try:
            res = subprocess.run(
                [exe, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=0.1",
                 *encoder_probe_args(enc), "-f", "null", "-"],
                check=False, capture_output=True, timeout=timeout,
                creationflags=_NO_WINDOW)
        except FileNotFoundError:
            _ENCODER_CACHE[cache_key] = False
            return False
        except Exception:
            return False
        ok = res.returncode == 0
        _ENCODER_CACHE[cache_key] = ok
        return ok


def list_gpu_encoders(ffmpeg_path: str | None = None) -> list[str]:
    """ffmpeg 里「存在」的硬件编码器（未实测）。"""
    try:
        info = ffmpeg_info(ffmpeg_path)
    except FileNotFoundError:
        return []
    return [enc for enc in ENCODER_ORDER if _encoder_listed(info, enc)]


def list_working_gpu_encoders(ffmpeg_path: str | None = None) -> list[str]:
    return [e for e in list_gpu_encoders(ffmpeg_path)
            if probe_encoder(e, ffmpeg_path=ffmpeg_path)]


def default_encoder(working: list[str]) -> str:
    for enc in DEFAULT_ENCODER_PREFERENCE:
        if enc in working:
            return enc
    return "libx264"


def resolve_gpu_encoder(gpu_encoder: str | None,
                        ffmpeg_path: str | None = None) -> str | None:
    """把用户选择解析成实测可用的编码器；不可用返回 None（调用方回退 CPU）。"""
    selected = (gpu_encoder or "").strip()
    if selected and selected.startswith("libx"):
        return None
    if selected:
        return selected if probe_encoder(selected, ffmpeg_path=ffmpeg_path) else None
    working = list_working_gpu_encoders(ffmpeg_path)
    return default_encoder(working) if working else None


# ===============================================================
#  编码参数
# ===============================================================

def preset_options(enc: str) -> list[str]:
    if enc.endswith("_nvenc"):
        return list(_PRESET_SETS["nvenc"])
    if enc.endswith("_qsv"):
        return list(_PRESET_SETS["qsv"])
    if enc.endswith("_amf"):
        return list(_PRESET_SETS["amf"])
    if enc.startswith("libx"):
        return list(_PRESET_SETS["x26x"])
    return []


def default_preset(enc: str) -> str:
    if enc.endswith("_nvenc"):
        return "p4"
    if enc.endswith("_qsv"):
        return "veryfast"
    if enc.endswith("_amf"):
        return "speed"
    if enc.startswith("libx"):
        return "veryfast"
    return ""


def _qp_from_quality(quality: int) -> int:
    """UI 的「质量 0-10」→ 量化参数（18 最好 … 28 最省）。"""
    q = max(0, min(10, int(quality)))
    return 18 + (10 - q)


def encoder_cmd_args(enc: str, quality: int, preset: str | None = None) -> list[str]:
    qp = str(_qp_from_quality(quality))
    if enc.endswith("_nvenc"):
        args = ["-c:v", enc, "-preset", preset or default_preset(enc),
                "-tune", "hq", "-rc", "vbr", "-cq", qp, "-b:v", "0"]
        if enc != "av1_nvenc":
            args += ["-spatial-aq", "1", "-rc-lookahead", "20"]
        return args
    if enc.endswith("_qsv"):
        return ["-c:v", enc, "-preset", preset or default_preset(enc),
                "-global_quality", qp]
    if enc.endswith("_amf"):
        args = ["-c:v", enc, "-usage", "transcoding",
                "-quality", preset or default_preset(enc),
                "-rc", "cqp", "-qp_i", qp, "-qp_p", qp]
        if enc != "av1_amf":
            args += ["-qp_b", qp]
        return args
    if enc.endswith("_videotoolbox"):
        return ["-c:v", enc, "-q:v", qp]
    if enc == "libx265":
        return ["-c:v", "libx265", "-preset", preset or default_preset(enc),
                "-crf", qp, "-tag:v", "hvc1"]
    if enc == "libx264":
        return ["-c:v", "libx264", "-preset", preset or default_preset(enc), "-crf", qp]
    return ["-c:v", enc, "-q:v", qp]


def video_encoder_args(quality: int, use_gpu: bool, gpu_encoder: str,
                       ffmpeg_path: str | None = None,
                       preset: str | None = None) -> list[str]:
    if use_gpu:
        enc = resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg_path)
        if enc:
            return encoder_cmd_args(enc, quality, preset) + \
                ["-pix_fmt", "yuv420p", "-threads", "0"]
    return encoder_cmd_args("libx264", quality, preset) + \
        ["-pix_fmt", "yuv420p", "-threads", "0"]


# ===============================================================
#  解码变体：硬件解码 + 缩放 + 转灰
# ===============================================================

@dataclass
class DecodeVariant:
    key: str
    vendor: str
    label: str
    hwaccel: list[str]
    vf: str                      # 含 {w} {h} 占位符

    def format(self, w: int, h: int) -> "DecodeVariant":
        return DecodeVariant(self.key, self.vendor, self.label,
                             list(self.hwaccel), self.vf.format(w=int(w), h=int(h)))

    def to_dict(self) -> dict:
        return asdict(self)


def build_decode_variants(hwaccels: list[str]) -> list[DecodeVariant]:
    have = {h.lower() for h in hwaccels}
    out: list[DecodeVariant] = []

    if "cuda" in have:
        hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        out.append(DecodeVariant("nv_scale_cuda", "nvidia", "NVDEC + scale_cuda",
                                 hw, "scale_cuda={w}:{h},hwdownload,format=nv12,format=gray"))
        out.append(DecodeVariant("nv_scale_npp", "nvidia", "NVDEC + scale_npp",
                                 hw, "scale_npp={w}:{h},hwdownload,format=nv12,format=gray"))
        out.append(DecodeVariant("nv_hwdec_swscale", "nvidia", "NVDEC + 软件缩放",
                                 ["-hwaccel", "cuda"],
                                 "scale={w}:{h}:flags=area,format=gray"))

    if "qsv" in have:
        hw = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        out.append(DecodeVariant("intel_scale_qsv", "intel", "QSV + scale_qsv",
                                 hw, "scale_qsv={w}:{h},hwdownload,format=nv12,format=gray"))
        out.append(DecodeVariant("intel_vpp_qsv", "intel", "QSV + vpp_qsv",
                                 hw, "vpp_qsv=w={w}:h={h},hwdownload,format=nv12,format=gray"))
        out.append(DecodeVariant("intel_hwdec_swscale", "intel", "QSV + 软件缩放",
                                 ["-hwaccel", "qsv"],
                                 "scale={w}:{h}:flags=area,format=gray"))

    # D3D11VA / DXVA2 是 Windows 通用硬解接口，NVIDIA/AMD/Intel 都能走
    if "d3d11va" in have:
        out.append(DecodeVariant("d3d11va", "d3d11", "D3D11VA 硬解 + 软件缩放",
                                 ["-hwaccel", "d3d11va"],
                                 "scale={w}:{h}:flags=area,format=gray"))
    if "dxva2" in have:
        out.append(DecodeVariant("dxva2", "d3d11", "DXVA2 硬解 + 软件缩放",
                                 ["-hwaccel", "dxva2"],
                                 "scale={w}:{h}:flags=area,format=gray"))

    out.append(DecodeVariant("sw", "none", "软件解码 + 软件缩放", [],
                             "scale={w}:{h}:flags=area,format=gray"))
    return out


def build_decode_cmd(ffmpeg: str, video_path: str, frames: int, w: int, h: int,
                     variant: DecodeVariant) -> list[str]:
    """硬解 → 缩放到 proc_res → 转灰 → rawvideo 管道。

    -fps_mode passthrough 等同步参数不要增删：会改变输出帧序/时间戳语义。
    """
    v = variant.format(w, h)
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        *v.hwaccel,
        "-i", video_path,
        "-an",
        "-frames:v", str(int(frames)),
        "-vf", v.vf,
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-fps_mode", "passthrough",
        "pipe:1",
    ]


def _make_probe_clips(ffmpeg: str, tmpdir: str, info: dict) -> list[str]:
    """生成 h264 / hevc 小片段，用于实测解码变体（HEVC 是常见素材编码）。"""
    clips: list[str] = []
    h264 = os.path.join(tmpdir, "probe_h264.mp4")
    try:
        res = _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=0.5",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                    h264], timeout=60)
        if res.returncode == 0 and os.path.isfile(h264):
            clips.append(h264)
    except Exception:
        pass
    if _encoder_listed(info, "libx265"):
        hevc = os.path.join(tmpdir, "probe_hevc.mp4")
        try:
            res = _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=0.5",
                        "-pix_fmt", "yuv420p", "-c:v", "libx265", "-preset", "ultrafast",
                        "-x265-params", "log-level=none", hevc], timeout=120)
            if res.returncode == 0 and os.path.isfile(hevc):
                clips.append(hevc)
        except Exception:
            pass
    return clips


def probe_decode_variant(ffmpeg: str, variant: DecodeVariant, clips: list[str],
                         w: int = 400, h: int = 225, frames: int = 12,
                         timeout: float = 30.0) -> tuple[bool, str]:
    """实测解码变体：每个测试片段都必须解出准确的 frames×w×h 字节。"""
    if not clips:
        return False, "没有可用的测试片段"
    bpf = int(w) * int(h)
    want = int(frames) * bpf
    for clip in clips:
        cmd = build_decode_cmd(ffmpeg, clip, frames, w, h, variant)
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                 creationflags=_NO_WINDOW)
        except subprocess.TimeoutExpired:
            return False, f"解码超时（{variant.key}）"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if res.returncode != 0:
            err_lines = (res.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
            return False, err_lines[-1][:220] if err_lines else f"exit {res.returncode}"
        data = res.stdout or b""
        if len(data) != want:
            return False, f"输出字节数不符（{len(data)} != {want}）"
        if not any(data[:2048]):
            return False, "输出内容全为 0"
    return True, ""


# ===============================================================
#  计算设备
# ===============================================================

def opencl_available() -> bool:
    try:
        return bool(cv2.ocl.haveOpenCL())
    except Exception:
        return False


def cv2_cuda_devices() -> int:
    """opencv-python 官方轮子不含 CUDA 模块，这里通常恒为 0。"""
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except Exception:
        return 0


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def open_video_capture(path: str):
    """打开视频采集器，优先硬件解码（Windows 上 D3D11VA 对三家通用）。

    失败自动回退普通打开，像素语义不变。
    """
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return cv2.VideoCapture(path)


# ===============================================================
#  CUDA 匹配加速包（可选）
#
#  主包不内置 torch（打包会 +2~3GB 且 CUDA 版本受构建机驱动限制）。
#  加载顺序：直接 import torch → 从加速包目录加载。
# ===============================================================

_ENV_PACK_URL = "ARKNIGHT_GPU_PACK_URL"
_TORCH_SUBDIR = "gpu-torch"

_pack_lock = threading.RLock()
_torch_module = None
_torch_status: dict | None = None
_dll_handles: list = []          # Windows 需要保持 DLL 目录句柄存活


def torch_dir() -> str:
    return os.path.join(app_core.app_home(), _TORCH_SUBDIR)


def _search_dirs() -> list[str]:
    dirs = [torch_dir()]
    bundle = app_core.bundle_dir()
    base = app_core.base_dir()
    for d in (os.path.join(bundle, _TORCH_SUBDIR), bundle,
              os.path.join(base, _TORCH_SUBDIR), base):
        if d not in dirs:
            dirs.append(d)
    return dirs


def _has_torch_package(directory: str) -> bool:
    return os.path.isdir(os.path.join(directory, "torch"))


def _dll_directories(root: str) -> list[str]:
    candidates = [os.path.join(root, "torch", "lib"), os.path.join(root, "torch")]
    nvidia = os.path.join(root, "nvidia")
    if os.path.isdir(nvidia):
        for name in sorted(os.listdir(nvidia)):
            candidates.append(os.path.join(nvidia, name, "bin"))
            candidates.append(os.path.join(nvidia, name, "lib"))
    return [c for c in candidates if os.path.isdir(c)]


def _register_paths(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for d in _dll_directories(root):
            try:
                _dll_handles.append(os.add_dll_directory(d))
            except OSError:
                pass


def _probe_torch_module(torch) -> dict:
    status = {
        "installed": True,
        "version": getattr(torch, "__version__", "?"),
        "cuda_available": False,
        "device_name": "",
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None) or "",
        "torch_dml": False,
        "reason": "",
    }
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
        if status["cuda_available"]:
            status["device_name"] = torch.cuda.get_device_name(0)
            try:
                status["device_count"] = int(torch.cuda.device_count())
            except Exception:
                status["device_count"] = 1
        else:
            status["reason"] = "torch 已安装但 torch.cuda 不可用（驱动过旧或非 CUDA 版本）"
    except Exception as exc:
        status["reason"] = f"torch.cuda 检测失败: {type(exc).__name__}: {exc}"
    try:
        import torch_directml  # type: ignore
        status["torch_dml"] = bool(torch_directml.is_available())
    except Exception:
        status["torch_dml"] = False
    return status


def load_torch(force: bool = False):
    """返回 torch 模块或 None；结果（含失败原因）缓存。"""
    global _torch_module, _torch_status
    with _pack_lock:
        if not force and _torch_status is not None:
            return _torch_module

        module = None
        source = "import"
        try:
            import torch  # type: ignore
            module = torch
        except Exception as exc:
            first_error = f"{type(exc).__name__}: {exc}"
            for d in _search_dirs():
                if not _has_torch_package(d):
                    continue
                try:
                    _register_paths(d)
                    import importlib
                    importlib.invalidate_caches()
                    import torch  # type: ignore
                    module = torch
                    source = d
                    break
                except Exception as exc2:
                    app_core.warn(f"从 {d} 加载 torch 失败: {type(exc2).__name__}: {exc2}",
                                  "gpu")
            if module is None:
                _torch_module = None
                _torch_status = {"installed": False, "version": "", "cuda_available": False,
                                 "device_name": "", "cuda_version": "", "torch_dml": False,
                                 "reason": f"未找到可用的 torch（{first_error}）"}
                return None

        _torch_module = module
        status = _probe_torch_module(module)
        status["source"] = source
        _torch_status = status
        return module


def torch_status(refresh: bool = False) -> dict:
    """torch / CUDA 可用性摘要。"""
    with _pack_lock:
        if _torch_status is None or refresh:
            load_torch(force=refresh)
        return dict(_torch_status or {"installed": False, "cuda_available": False})


def is_declined() -> bool:
    return bool(app_core.get("gpu_pack_declined", False))


def set_declined(flag: bool = True) -> bool:
    return app_core.set_value("gpu_pack_declined", bool(flag))


def should_ask() -> bool:
    """NVIDIA 机器 + CUDA 不可用 + 用户没拒绝过 → 才提示下载。"""
    st = torch_status()
    if st.get("cuda_available") or is_declined():
        return False
    return _nvidia_hint()


def _nvidia_hint() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        subprocess.run([exe, "-L"], check=False, capture_output=True,
                       timeout=6, creationflags=_NO_WINDOW)
        return True
    except Exception:
        return False


def pack_url() -> str:
    return (os.environ.get(_ENV_PACK_URL)
            or str(app_core.get("gpu_pack_url", "") or "")).strip()


def uv_path() -> str | None:
    names = ["uv.exe"] if sys.platform == "win32" else ["uv"]
    for base in (app_core.bundle_dir(), app_core.base_dir()):
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p):
                return p
    return shutil.which("uv")


def _download(url: str, dest: str, progress_cb=None, cancel_event=None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "arknight-auto-editing"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("已取消")
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(min(0.95, (done / total * 0.95) if total else 0.0), "下载中")
    if progress_cb:
        progress_cb(0.96, "下载完成，校验中")


def _sha256_file(path: str, progress_cb=None, cancel_event=None) -> str:
    h = hashlib.sha256()
    total = os.path.getsize(path) or 1
    done = 0
    with open(path, "rb") as fh:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("已取消")
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(min(1.0, done / total), "校验中")
    return h.hexdigest()


def _safe_extract(zip_path: str, target: str, progress_cb=None, cancel_event=None) -> None:
    os.makedirs(target, exist_ok=True)
    root = os.path.abspath(target)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        total = max(1, len(names))
        for i, info in enumerate(names):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("已取消")
            member = os.path.abspath(os.path.join(target, info))
            if member != root and not member.startswith(root + os.sep):
                raise RuntimeError(f"加速包内含非法路径: {info}")
            zf.extract(info, target)
            if progress_cb and (i % 50 == 0 or i == len(names) - 1):
                progress_cb(0.96 + (i + 1) / total * 0.04, "解压中")


def _normalize_layout(directory: str) -> None:
    """zip 里若多套了一层目录，把内容上提，保证 directory/torch 存在。"""
    if _has_torch_package(directory):
        return
    try:
        entries = list(os.listdir(directory))
    except OSError:
        return
    for name in entries:
        sub = os.path.join(directory, name)
        if not os.path.isdir(sub) or not _has_torch_package(sub):
            continue
        for item in os.listdir(sub):
            src, dst = os.path.join(sub, item), os.path.join(directory, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            os.rmdir(sub)
        except OSError:
            pass
        return


def install_from_url(url: str, sha256: str | None = None,
                     progress_cb=None, cancel_event=None) -> tuple[bool, str]:
    """下载并解压 GPU 加速包 zip。"""
    target = torch_dir()
    tmp_zip = target + ".download.zip"
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        _download(url, tmp_zip, progress_cb, cancel_event)
        if sha256:
            got = _sha256_file(tmp_zip, progress_cb, cancel_event)
            if got.lower() != sha256.lower():
                return False, f"校验失败：期望 {sha256[:12]}…，实际 {got[:12]}…"
        if progress_cb:
            progress_cb(0.96, "解压中")
        _safe_extract(tmp_zip, target, progress_cb, cancel_event)
        _normalize_layout(target)
        if not _has_torch_package(target):
            return False, "加速包解压后没有找到 torch 目录，包结构可能不对"
        if progress_cb:
            progress_cb(1.0, "完成")
        st = torch_status(refresh=True)
        if not st.get("installed"):
            return False, f"加速包已解压但加载失败：{st.get('reason', '未知原因')}"
        if not st.get("cuda_available"):
            return False, f"加速包已加载但 CUDA 不可用：{st.get('reason', '')}"
        return True, f"已启用 CUDA 匹配加速（torch {st.get('version')} / {st.get('device_name')}）"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if os.path.isfile(tmp_zip):
                os.remove(tmp_zip)
        except OSError:
            pass


def install_with_uv(progress_cb=None, cancel_event=None,
                    cuda_index: str = "https://download.pytorch.org/whl/cu128") -> tuple[bool, str]:
    """用内置/系统 uv 把 torch 装进加速包目录（不依赖宿主 Python 环境）。"""
    uv = uv_path()
    if not uv:
        return False, "未找到 uv（内置 uv.exe 或 PATH 中都没有），请改用离线加速包"
    target = torch_dir()
    os.makedirs(target, exist_ok=True)
    cmd = [uv, "pip", "install", "--target", target,
           "--python-platform", "windows" if sys.platform == "win32" else sys.platform,
           "--python-version", f"{sys.version_info.major}.{sys.version_info.minor}",
           "--index-url", cuda_index, "--upgrade", "torch"]
    if progress_cb:
        progress_cb(0.02, "准备下载 torch（约 2.5GB，首次较慢）")
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW, text=True, encoding="utf-8", errors="replace")
    except Exception as exc:
        return False, f"启动 uv 失败: {type(exc).__name__}: {exc}"
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                return False, "已取消"
            line = line.rstrip()
            if line:
                lines.append(line)
                del lines[:-40]
                if progress_cb:
                    progress_cb(0.5, line[:120])
        rc = proc.wait(timeout=1800)
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        return False, f"安装中断: {type(exc).__name__}: {exc}"
    if rc != 0:
        return False, f"uv 安装失败（退出码 {rc}）：" + "\n".join(lines[-6:])[:500]
    if progress_cb:
        progress_cb(1.0, "完成")
    st = torch_status(refresh=True)
    if not st.get("cuda_available"):
        return False, f"torch 已安装但 CUDA 不可用：{st.get('reason', '')}"
    return True, f"已启用 CUDA 匹配加速（torch {st.get('version')} / {st.get('device_name')}）"


def install(progress_cb=None, cancel_event=None) -> tuple[bool, str]:
    """优先官方/自托管加速包 zip，否则用 uv 在线安装。"""
    url = pack_url()
    if url:
        return install_from_url(url, progress_cb=progress_cb, cancel_event=cancel_event)
    return install_with_uv(progress_cb=progress_cb, cancel_event=cancel_event)


def uninstall() -> tuple[bool, str]:
    d = torch_dir()
    if not os.path.isdir(d):
        return True, "加速包目录不存在"
    try:
        shutil.rmtree(d, ignore_errors=False)
    except OSError as exc:
        return False, f"删除失败: {type(exc).__name__}: {exc}"
    torch_status(refresh=True)
    return True, "已删除 GPU 加速包"


def pack_status_text() -> str:
    """加速包状态一行摘要（供 UI 使用）。"""
    st = torch_status()
    if not st.get("installed"):
        return "未安装（模板匹配走 OpenCL / CPU）"
    if st.get("cuda_available"):
        src = st.get("source", "")
        where = "本地" if src == "import" else src
        return (f"已启用 torch {st.get('version')} · CUDA {st.get('cuda_version')} · "
                f"{st.get('device_name')}（{where}）")
    return f"已安装但不可用：{st.get('reason', '')}"


# ===============================================================
#  GpuProfile
# ===============================================================

@dataclass
class GpuProfile:
    vendor: str = "none"
    device_name: str = ""
    ffmpeg_path: str = ""
    ffmpeg_version: str = ""
    hwaccels: list[str] = field(default_factory=list)
    encoders_listed: list[str] = field(default_factory=list)
    encoders_working: list[str] = field(default_factory=list)
    encoder: str = ""
    decode_variants: list[dict] = field(default_factory=list)
    decode_variant: str = ""
    export_workers: int = 1
    opencl: bool = False
    cv2_cuda_devices: int = 0
    torch: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    probed_at: float = 0.0
    from_cache: bool = False

    def decode_variant_obj(self, w: int = 400, h: int = 225) -> DecodeVariant | None:
        if not self.decode_variant:
            return None
        for v in build_decode_variants(self.hwaccels):
            if v.key == self.decode_variant:
                return v.format(w, h)
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GpuProfile":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def _default_workers(vendor: str) -> int:
    if vendor == "nvidia":
        return 3
    if vendor == "intel":
        return 2
    if vendor == "amd":
        return 1
    return max(1, cpu_count() // 2)


def _cache_key(ffmpeg_path: str, version: str, device: str, torch_sig: str = "") -> str:
    return "|".join([os.path.normcase(ffmpeg_path or ""), version or "", device or "",
                     torch_sig or ""])


def _torch_sig(torch_st: dict) -> str:
    """缓存键里必须带上 torch 状态：同一台机器上「有 torch 的源码环境」与
    「没内置 torch 的 exe」会写出不同的可用后端，不能互相复用缓存。"""
    st = torch_st or {}
    return f"{bool(st.get('installed'))}:{st.get('version', '')}:{bool(st.get('cuda_available'))}"


def _device_name(vendor: str, torch_st: dict) -> str:
    if torch_st.get("cuda_available") and torch_st.get("device_name"):
        return str(torch_st["device_name"])
    if vendor == "nvidia":
        exe = shutil.which("nvidia-smi")
        if exe:
            try:
                res = _run([exe, "--query-gpu=name,driver_version",
                            "--format=csv,noheader"], timeout=8)
                line = (res.stdout or "").strip().splitlines()
                if line:
                    return line[0].strip()[:120]
            except Exception:
                pass
    if vendor == "none":
        return ""
    return f"{VENDOR_LABEL.get(vendor, vendor)}（型号未知）"


def _probe_impl(ffmpeg_path: str | None = None,
                probe_decoders: bool = True) -> GpuProfile:
    profile = GpuProfile(probed_at=time.time())
    torch_st = torch_status()
    profile.torch = torch_st
    profile.opencl = opencl_available()
    profile.cv2_cuda_devices = cv2_cuda_devices()

    try:
        exe = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError as exc:
        profile.notes.append(str(exc))
        app_core.error(f"未找到 FFmpeg：{exc}", "gpu")
        return profile

    info = ffmpeg_info(exe)
    profile.ffmpeg_path = exe
    profile.ffmpeg_version = info.get("version", "")
    profile.hwaccels = list(info.get("hwaccels") or [])
    if info.get("error"):
        profile.notes.append(f"ffmpeg 信息读取异常: {info['error']}")

    listed = list_gpu_encoders(exe)
    profile.encoders_listed = listed
    working = [e for e in listed if probe_encoder(e, ffmpeg_path=exe)]
    profile.encoders_working = working
    if working:
        profile.encoder = default_encoder(working)
        profile.vendor = VENDOR_OF_ENCODER.get(working[0], "none")
    profile.device_name = _device_name(profile.vendor, torch_st)
    profile.export_workers = _default_workers(profile.vendor)

    if not profile.hwaccels:
        profile.notes.append("ffmpeg 未报告任何硬件加速方式（可能是精简构建）")
    if profile.vendor == "none" and listed:
        profile.notes.append("列出了硬件编码器但实测均不可用（驱动/构建不匹配），导出将走 CPU")
    if not profile.opencl:
        profile.notes.append("OpenCL 不可用，匹配只能用 CPU")

    if probe_decoders and profile.hwaccels:
        variants = build_decode_variants(profile.hwaccels)
        with tempfile.TemporaryDirectory(prefix="aae_gpu_probe_") as tmpdir:
            clips = _make_probe_clips(exe, tmpdir, info)
            if not clips:
                profile.notes.append("无法生成解码测试片段，跳过硬件解码实测")
            results: list[dict] = []
            vendor_done: set[str] = set()
            for v in variants:
                if not clips:
                    results.append({**v.to_dict(), "ok": None, "error": "未测试"})
                    continue
                if v.vendor != "none" and v.vendor in vendor_done:
                    # 同厂商已有可用变体，其余只登记不实测（省时间）
                    results.append({**v.to_dict(), "ok": None,
                                    "error": "未测试（同厂商已有可用变体）"})
                    continue
                with _PROBE_LOCK:
                    ok, err = probe_decode_variant(exe, v, clips)
                results.append({**v.to_dict(), "ok": ok, "error": err})
                if ok:
                    vendor_done.add(v.vendor)
                    if not profile.decode_variant:
                        profile.decode_variant = v.key
            profile.decode_variants = results
    else:
        profile.decode_variants = [
            {**v.to_dict(), "ok": None, "error": "未测试"} for v in build_decode_variants([])
        ]

    if profile.decode_variant:
        label = next((r["label"] for r in profile.decode_variants
                      if r["key"] == profile.decode_variant), profile.decode_variant)
        app_core.info(f"硬件解码可用：{label}", "gpu")
    else:
        app_core.warn("没有实测可用的硬件解码路径，分析将使用软件解码", "gpu")
    return profile


def detect(ffmpeg_path: str | None = None, force: bool = False,
           probe_decoders: bool = True, use_cache: bool = True) -> GpuProfile:
    """探测并返回 GpuProfile（缓存命中条件：ffmpeg 路径+版本+设备名一致）。"""
    global _PROFILE_MEM
    with _PROFILE_LOCK:
        if not force and _PROFILE_MEM is not None and use_cache:
            return _PROFILE_MEM

    cached = app_core.load_gpu_profile() if (use_cache and not force) else {}
    if cached:
        prof = GpuProfile.from_dict(cached)
        try:
            exe = resolve_ffmpeg_path(ffmpeg_path)
            info = ffmpeg_info(exe)
            cur_torch = torch_status()
            device = _device_name(prof.vendor, cur_torch)
            key = _cache_key(exe, info.get("version", ""), device, _torch_sig(cur_torch))
            if cached.get("_key") == key:
                prof.torch = cur_torch
                prof.from_cache = True
                with _PROFILE_LOCK:
                    _PROFILE_MEM = prof
                return prof
        except FileNotFoundError:
            pass

    prof = _probe_impl(ffmpeg_path, probe_decoders=probe_decoders)
    data = prof.to_dict()
    data["_key"] = _cache_key(prof.ffmpeg_path, prof.ffmpeg_version, prof.device_name,
                              _torch_sig(prof.torch))
    app_core.save_gpu_profile(data)
    with _PROFILE_LOCK:
        _PROFILE_MEM = prof
    return prof


def detect_async(callback, ffmpeg_path: str | None = None, force: bool = False,
                 probe_decoders: bool = True) -> threading.Thread:
    """异步探测（UI 用）。callback(profile, error) 在后台线程调用。"""
    def _worker():
        try:
            prof = detect(ffmpeg_path, force=force, probe_decoders=probe_decoders)
        except Exception as exc:
            app_core.error(f"硬件探测失败: {type(exc).__name__}: {exc}", "gpu")
            callback(None, exc)
            return
        callback(prof, None)

    t = threading.Thread(target=_worker, daemon=True, name="gpu-probe")
    t.start()
    return t


def describe(profile: GpuProfile) -> list[str]:
    """能力摘要（多行，供 UI 显示）。"""
    if profile is None:
        return ["尚未探测硬件能力"]
    lines: list[str] = []
    vendor = VENDOR_LABEL.get(profile.vendor, profile.vendor)
    lines.append(f"{vendor} · {profile.device_name}" if profile.device_name else vendor)

    dec = "软件解码"
    if profile.decode_variant:
        dec = next((r["label"] for r in profile.decode_variants
                    if r["key"] == profile.decode_variant), profile.decode_variant)
    lines.append(f"解码加速: {dec}")

    st = profile.torch or {}
    if st.get("cuda_available"):
        lines.append(f"匹配可选: CUDA(torch {st.get('version')}) · "
                     f"OpenCL({'可用' if profile.opencl else '不可用'})")
    else:
        lines.append(f"匹配可选: OpenCL({'可用' if profile.opencl else '不可用'}) · CPU")
    if profile.cv2_cuda_devices == 0:
        lines.append("OpenCV CUDA 模块: 无（官方轮子不带，CUDA 匹配走 torch）")

    if profile.encoders_working:
        lines.append("编码: " + ", ".join(profile.encoders_working))
    elif profile.encoders_listed:
        lines.append("编码: 列出的硬件编码器均未通过实测 → 使用 libx264")
    else:
        lines.append("编码: libx264（未发现硬件编码器）")

    lines.append(f"导出并发建议: {profile.export_workers}")
    if profile.ffmpeg_path:
        lines.append(f"ffmpeg: {profile.ffmpeg_path}")
        ver = profile.ffmpeg_version.split(" Copyright")[0][:60]
        if ver:
            lines.append(f"        {ver}")
    else:
        lines.append("ffmpeg: 未找到")
    for n in profile.notes:
        lines.append(f"⚠ {n}")
    return lines


def profile_signature(profile: GpuProfile) -> str:
    """测速缓存失效判断用的指纹。"""
    return "|".join([
        profile.vendor,
        profile.device_name,
        profile.decode_variant,
        ",".join(profile.encoders_working),
        str(profile.torch.get("version", "")),
        str(profile.opencl),
    ])
