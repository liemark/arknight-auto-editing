"""房间脉冲响应池: 覆盖"录屏时经过了房间/扬声器/麦克风"的一整类失真。

三个来源:
  1. MIT IR 库 (assets/mit_ir, 16kHz) -> 升到 48k 后作为固定 IR 复用;
  2. pyroomacoustics 在线合成 shoebox 房间 IR (RT60 0.05~1.5s, 尺寸随机);
  3. 纯合成 IR (指数衰减噪声 + 稀疏早期反射), 兜底且可极端化。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..audio import SR, read_wav, resample

try:
    import pyroomacoustics as pra  # type: ignore
    _HAS_PRA = True
except Exception:  # pragma: no cover
    _HAS_PRA = False


@dataclass
class IRSpec:
    rt60: float
    kind: str           # mit / shoebox / synth
    length: int


def _norm_ir(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float32)
    i = int(np.argmax(np.abs(h))) if h.size else 0
    if h.size and abs(h[i]) > 1e-9:
        h = h[: max(64, min(h.size, int(len(h) * 0.999)))]
    e = float(np.sqrt(np.mean(h.astype(np.float64) ** 2)))
    if e > 1e-9:
        h = h / e
    return h.astype(np.float32)


def load_mit_irs(mit_root: str, *, sr_out: int = SR, limit: int | None = None,
                 cache_path: str | None = None) -> list[np.ndarray]:
    """把 MIT IR (16k) 升到 sr_out。结果可缓存为单个 npz。"""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        n = int(z["n"])
        return [_norm_ir(z[f"h{i}"]) for i in range(n)]
    irs: list[np.ndarray] = []
    root = os.path.abspath(mit_root)
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            try:
                x = read_wav(os.path.join(dirpath, fn), mono=True)
            except Exception:
                continue
            if x.size < 64:
                continue
            x = resample(x, 16000, sr_out)
            x = x[: int(sr_out * 1.5)]
            irs.append(_norm_ir(x))
            if limit and len(irs) >= limit:
                break
        if limit and len(irs) >= limit:
            break
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(cache_path, n=len(irs), **{f"h{i}": h for i, h in enumerate(irs)})
    return irs


def synth_ir(rng: np.random.Generator, rt60: float, sr: int = SR,
             *, early: int = 6, hf_damp: float = 0.6) -> np.ndarray:
    """合成 IR: 稀疏早期反射 + 指数衰减的着色噪声尾巴 + 可选 HF 阻尼。"""
    n = max(64, int(sr * min(1.6, rt60 * 1.2)))
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-6.9078 * t / max(0.02, rt60)).astype(np.float32)     # -60dB @ rt60
    tail = rng.standard_normal(n).astype(np.float32) * env
    if hf_damp < 0.999:                       # 高频随时间衰减更快 (空气/软装吸收)
        a = float(np.clip(1.0 - hf_damp, 0.01, 0.9))
        from scipy.signal import lfilter
        tail = lfilter([a], [1.0, -(1.0 - a)], tail).astype(np.float32)
    h = tail
    if early > 0:
        hi = max(2, min(int(sr * 0.08), n))
        idx = rng.integers(1, hi, size=early)
        amp = rng.uniform(-0.9, 0.9, size=early).astype(np.float32) * \
            np.exp(-idx / max(1.0, sr * 0.03))
        h = h.copy()
        np.add.at(h, idx, amp)
    h[0] = 1.0
    return _norm_ir(h)


def shoebox_ir(rng: np.random.Generator, rt60: float, sr: int = SR
               ) -> np.ndarray | None:
    """pyroomacoustics shoebox IR; 不可用时返回 None。"""
    if not _HAS_PRA:
        return None
    try:
        dim = rng.uniform(3.0, 12.0, size=3)
        room = pra.ShoeBox(dim, fs=sr, materials=pra.Material(0.2),
                           max_order=6, absorption=None, air_absorption=False)
        room.set_rt60(float(np.clip(rt60, 0.05, 1.5)))
        src = rng.uniform(0.5, 1.0, size=3) * dim
        mic = rng.uniform(0.5, 1.0, size=3) * dim
        room.add_source(src)
        room.add_microphone_array(pra.MicrophoneArray(mic.reshape(3, 1), sr))
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float32)
        if h.size < 32:
            return None
        return _norm_ir(h[: int(sr * 1.6)])
    except Exception:
        return None


class IRPool:
    """IR 池: 从三个来源采样, 权重可调。"""

    def __init__(self, mit: list[np.ndarray] | None = None, *, sr: int = SR,
                 rng: np.random.Generator | None = None,
                 p_mit: float = 0.45, p_shoebox: float = 0.35):
        self.mit = mit or []
        self.sr = sr
        self.rng = rng or np.random.default_rng(0)
        self.p_mit = p_mit
        self.p_shoebox = p_shoebox

    def __len__(self) -> int:
        return len(self.mit)

    def sample(self, rt60: float | None = None) -> tuple[np.ndarray, IRSpec]:
        r = self.rng
        rt = float(rt60 if rt60 is not None else r.uniform(0.05, 1.5))
        u = r.random()
        if self.mit and u < self.p_mit:
            h = self.mit[int(r.integers(len(self.mit)))]
            return h, IRSpec(rt60=rt, kind="mit", length=h.size)
        if u < self.p_mit + self.p_shoebox:
            h = shoebox_ir(r, rt, self.sr)
            if h is not None:
                return h, IRSpec(rt60=rt, kind="shoebox", length=h.size)
        h = synth_ir(r, rt, self.sr)
        return h, IRSpec(rt60=rt, kind="synth", length=h.size)

    @staticmethod
    def build(mit_root: str, cache_path: str | None = None, *, sr: int = SR,
              limit: int | None = 200) -> "IRPool":
        mit = load_mit_irs(mit_root, sr_out=sr, limit=limit, cache_path=cache_path)
        return IRPool(mit, sr=sr)
