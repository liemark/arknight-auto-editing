"""轻量配置系统: YAML + `_base_` 继承 + 点号覆盖。

设计取舍:
  - 不引入 hydra/omegaconf (省依赖); 只需要 dict 深度合并 + CLI 覆盖。
  - 所有配置读点都通过 Cfg.get("a.b.c", default), 未命中的键返回默认值,
    这样 configs/*.yaml 可以只写"与 base 不同的部分"。
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml

__all__ = ["Cfg", "load_cfg"]

# 项目根 = 本文件上两级 (audio_inverse/audio_inverse/config.py -> repo root)
_HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(_HERE)          # .../audio_inverse   (包目录)
REPO_ROOT = os.path.dirname(PKG_ROOT)      # 仓库根


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str, _seen: tuple[str, ...] = ()) -> dict:
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"配置循环继承: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    parents = raw.pop("_base_", None)
    if parents is None:
        return raw
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for p in parents:
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(path), p)
        merged = _deep_merge(merged, _load_yaml(p, _seen + (path,)))
    return _deep_merge(merged, raw)


def _coerce(s: str) -> Any:
    """把 CLI 字符串覆盖值转成 YAML 标量 (数字/布尔/null/列表)。"""
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


class Cfg:
    """dict 包装 + 点号路径 + 默认值 + 覆盖。

    >>> c = load_cfg("base.yaml", ["synth.max_atoms=24", "train.lr=5e-5"])
    >>> c.get("synth.max_atoms")          # 24
    >>> c.get("nope.deep", 7)             # 7
    """

    def __init__(self, data: dict | None = None, *, path: str | None = None):
        self.data: dict = data or {}
        self.path = path

    # ---- 读取 -------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def __getitem__(self, key: str) -> Any:
        v = self.get(key, _MISSING)
        if v is _MISSING:
            raise KeyError(key)
        return v

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def keys(self):
        return self.data.keys()

    # ---- 写入 / 覆盖 ------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        cur = self.data
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    def update(self, pairs: Iterable[str]) -> "Cfg":
        for item in pairs:
            if "=" not in item:
                raise ValueError(f"覆盖项必须是 key=value 形式: {item!r}")
            k, v = item.split("=", 1)
            self.set(k.strip(), _coerce(v.strip()))
        return self

    def copy(self) -> "Cfg":
        return Cfg(copy.deepcopy(self.data), path=self.path)

    # ---- 路径便捷 ---------------------------------------------------------
    def abspath(self, key: str, default: str | None = None) -> str:
        """取一个路径配置项, 相对路径按仓库根解析。"""
        v = self.get(key, default)
        if v is None:
            raise KeyError(f"缺少路径配置: {key}")
        v = os.path.expanduser(str(v))
        return v if os.path.isabs(v) else os.path.normpath(os.path.join(REPO_ROOT, v))

    def as_dict(self) -> dict:
        return copy.deepcopy(self.data)


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()


def load_cfg(path: str | None = None, overrides: Iterable[str] | None = None, **kw) -> Cfg:
    """读取配置。path 为 None 时用 configs/base.yaml。相对路径按包目录解析。"""
    if path is None:
        path = os.path.join(PKG_ROOT, "configs", "base.yaml")
    elif not os.path.isabs(path) and not os.path.exists(path):
        cand = os.path.join(PKG_ROOT, "configs", path)
        if os.path.exists(cand):
            path = cand
    cfg = Cfg(_load_yaml(path), path=os.path.abspath(path))
    for k, v in kw.items():
        cfg.set(k, v)
    if overrides:
        cfg.update(overrides)
    return cfg
