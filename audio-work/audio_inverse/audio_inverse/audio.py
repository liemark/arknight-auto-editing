"""音频 IO 与基础变换。

约定:
  * 系统内部统一 48 kHz。清单里原子的 `sr` 是【解包文件真实采样率】, 原样保留;
    真正需要 48k 时才重采样 (原子库以原始 sr 存 int16, 省 3x 磁盘)。
  * 全部用 float32 [-1,1] 处理; 落盘 int16。
  * 重采样统一走 soxr 最高质量 (VHQ), 只有合成侧的"模拟游戏重采样"才用
    torchaudio 的低质量核 —— 那是被刻意注入的失真, 不是我们的处理链。
"""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:  # soxr 是可选加速; 没有就退回 scipy
    import soxr  # type: ignore

    _HAS_SOXR = True
except Exception:  # pragma: no cover
    _HAS_SOXR = False

from scipy import signal as _sp_signal

SR = 48000
EPS = 1e-12


# --------------------------------------------------------------------------- IO
@dataclass(frozen=True)
class WavInfo:
    sr: int
    ch: int
    frames: int

    @property
    def dur(self) -> float:
        return self.frames / float(self.sr)


def wav_info(path: str) -> WavInfo:
    """只读 WAV 头 (不解析数据), 用于清单校验。"""
    with wave.open(path, "rb") as w:
        return WavInfo(w.getframerate(), w.getnchannels(), w.getnframes())


def read_wav(path: str, *, sr_out: int | None = None, mono: bool = True,
             dtype=np.float32) -> np.ndarray:
    """读 WAV -> float32。sr_out 给定时高质量重采样; mono=True 时下混。

    下混规则: 立体声取平均 (游戏内 SFX 是单声道 stem + 声像, 平均是合理近似)。
    """
    with wave.open(path, "rb") as w:
        sr, ch, n, sw = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"不支持的位深 {sw*8} bit: {path}")
    if ch > 1:
        x = x.reshape(-1, ch)
        if mono:
            x = x.mean(axis=1)
    if sr_out is not None and sr_out != sr:
        x = resample(x, sr, sr_out)
    return np.ascontiguousarray(x, dtype=dtype)


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    """float32 -> int16 WAV (带 dither-free 硬限幅, 防爆音)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(y * 32767.0).astype("<i2")
    if pcm.ndim == 1:
        ch = 1
    else:
        ch = pcm.shape[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------- resample / misc
def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """高质量重采样 (soxr VHQ, 无则 scipy polyphase)。保持 float32。"""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    if _HAS_SOXR:
        return np.asarray(soxr.resample(x, sr_in, sr_out, quality="VHQ"), dtype=np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    return _sp_signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to48k(x: np.ndarray, sr: int) -> np.ndarray:
    return resample(x, sr, SR)


def frame(x: np.ndarray, n: int, hop: int, *, pad: bool = True) -> np.ndarray:
    """切帧 -> [T, n]。pad=True 时尾部补零凑整帧。"""
    x = np.asarray(x, dtype=np.float32)
    if pad:
        need = n if len(x) < n else ((len(x) - n) % hop)
        if need:
            x = np.concatenate([x, np.zeros(need, dtype=np.float32)])
    nf = 1 + (len(x) - n) // hop
    if nf <= 0:
        return np.zeros((0, n), dtype=np.float32)
    idx = np.arange(n)[None, :] + hop * np.arange(nf)[:, None]
    return x[idx]


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def db(x: float | np.ndarray, floor: float = -120.0) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS)), floor)


def peak(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def normalize_rms(x: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    r = rms(x)
    if r < 1e-7:
        return np.asarray(x, dtype=np.float32)
    return (np.asarray(x, dtype=np.float32) * (10 ** (target_db / 20.0) / r)).astype(np.float32)


def rms_normalize_db(x: np.ndarray, target_db: float = -26.0) -> np.ndarray:
    """按 RMS 归一到目标电平; 静音样本原样返回 (避免把底噪放大成信号)。

    【全链路唯一实现】: 合成器 (synth/mixer.py)、推理渲染 (models/synthesize.py)、
    精对齐 (models/refine.py) 必须共用它, 否则模板幅度不一致, 相消会变成增强。
    """
    x = np.asarray(x)
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    if r < 1e-6:
        return np.asarray(x, dtype=np.float32)
    k = float(np.clip((10.0 ** (target_db / 20.0)) / r, 1e-4, 1e4))
    return (x * k).astype(np.float32)


def fade(x: np.ndarray, ms_in: float = 2.0, ms_out: float = 5.0, sr: int = SR) -> np.ndarray:
    """首尾淡入淡出, 避免拼接爆音。原地返回副本。"""
    y = np.array(x, dtype=np.float32, copy=True)
    ni, no = int(sr * ms_in / 1000.0), int(sr * ms_out / 1000.0)
    ni, no = min(ni, y.size), min(no, y.size)
    if ni > 1:
        y[:ni] *= np.linspace(0.0, 1.0, ni, dtype=np.float32)
    if no > 1:
        y[-no:] *= np.linspace(1.0, 0.0, no, dtype=np.float32)
    return y


def trim_silence(x: np.ndarray, sr: int = SR, thresh_db: float = -60.0,
                 pad_ms: float = 5.0) -> np.ndarray:
    """按能量裁掉首尾静音 (保留 pad_ms 余量)。全静音则原样返回。"""
    if x.size == 0:
        return x
    win = max(1, int(sr * 0.01))
    n = x.size // win
    if n < 2:
        return x
    e = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1) + EPS)
    idx = np.where(20 * np.log10(e) > thresh_db)[0]
    if idx.size == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    a = max(0, idx[0] * win - pad)
    b = min(x.size, (idx[-1] + 1) * win + pad)
    return np.ascontiguousarray(x[a:b])


def apply_gain(x: np.ndarray, gain_db: float | np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * (10.0 ** (np.asarray(gain_db, np.float32) / 20.0)))


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """tanh 软限幅 (drive=1 时弱饱和)。"""
    d = max(1e-3, float(drive))
    return np.tanh(np.asarray(x, dtype=np.float32) * d) / np.tanh(d)


def iter_chunks(n_total: int, win: int, hop: int) -> Iterator[tuple[int, int]]:
    """滑动窗口 (start, end) 序列, 覆盖 [0, n_total)。最后一窗贴右边界。"""
    if n_total <= win:
        yield 0, n_total
        return
    s = 0
    while s + win < n_total:
        yield s, s + win
        s += hop
    yield n_total - win, n_total
