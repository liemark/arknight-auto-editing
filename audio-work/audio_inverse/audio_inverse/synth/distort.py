"""失真族: 覆盖"明日方舟内置后处理 + 录制者收音质量"的所有未知情况。

设计原则
  * 不猜具体后处理类型, 而是定义一个**参数化、可开关**的族, 分层采样;
  * 每个算子要么"幅度-相位精确"(重采样/时延/EQ/增益/混响), 要么是刻意注入的
    非线性(饱和/限幅/编解码) —— 前者可逆, 后者用可学习参数逼近;
  * 拉伸 4 类并行建模 (见 StretchMode), 因为解包音频与视频音频之间的时间关系
    完全未知: 可能是游戏引擎变速、可能被剪辑软件拉伸、可能是录屏掉帧。

所有 numpy 路径用于【合成数据】(离线/子进程, 追求真实); torch 路径用于
【可微变换层】(models/transform.py), 二者参数语义严格对齐。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..audio import SR, apply_gain, resample, soft_clip

try:
    import soxr  # type: ignore
    _HAS_SOXR = True
except Exception:  # pragma: no cover
    _HAS_SOXR = False

# scipy.fft 用 pocketfft 的 C 实现, 比 numpy.fft 快 2~4 倍 (本文件 FFT 调用极密集)。
from scipy.fft import irfft as _irfft, rfft as _rfft

StretchMode = Literal["resample", "wsola", "tilt", "piecewise"]
STRETCH_MODES: tuple[str, ...] = ("resample", "wsola", "tilt", "piecewise")


# ============================================================== 参数
@dataclass
class TParams:
    """一个原子实例要经历的全部变换。

    约定:
      * `r`        : 重采样比, 播放端 sr 相对源 sr 的比例。r=1.0 表示无变化。
                     r 同时改变时长与音高 (resample 语义)。
      * `pitch_semi`: 与时长无关的独立变调 (半音)。与 r 叠加。
      * `mode`     : 拉伸类型。'resample' 用 r 实现变速变调; 'wsola' 变速不变调;
                     'tilt' 线性倾斜; 'piecewise' 分段常速。
      * `tilt`     : 线性倾斜系数, t' = t*(1 + tilt*(2t/L - 1)), |tilt| <= 0.05
      * `piece`    : 分段常速的 1~3 段比例 (仅 mode='piecewise')
      * `delay`    : 相对窗口起点的秒数 (可负 = 原子起点在窗口之前)
      * `gain_db`  : 线性增益 (dB)
      * `eq_db`    : mel 频带增益 (dB), 长度 = n_bands
      * `rir`      : 房间脉冲响应 (48k float32), None = 干声
      * `wet`      : 混响干湿比 0..1
      * `drive`    : tanh 饱和强度, 1.0 = 无饱和
      * `poly`     : 3 阶多项式非线性系数, 0 = 无
      * `bitcrush` : 位深缩减级数, 0 = 无
      * `clip`     : 削波阈值 (线性), >=1 = 不削
      * `pan`      : 声像 -1..1 (仅用于生成视频侧立体声)
      * `ch_delay` : 左右声道时延差 (秒, 0..1ms)
      * `ch_gain_db`: 左右声道增益差 (dB)

    注意: 底噪 (`noise_db`) 【不在这里】 —— 噪声电平是混音属性, 由
    Mixer 单独持有并显式作为"加性分量"输出 (见 Instance.noise_db)。
    这样渲染可完全复现, 也给模型一个明确的噪声电平参数可学。
    """
    r: float = 1.0
    pitch_semi: float = 0.0
    mode: str = "resample"
    tilt: float = 0.0
    piece: tuple[float, ...] = ()
    delay: float = 0.0
    gain_db: float = 0.0
    eq_db: np.ndarray | None = None
    n_bands: int = 0
    rir: np.ndarray | None = None
    wet: float = 0.0
    rt60: float = 0.3            # 仅记录用: rir 由外部按此值采样后填入
    drive: float = 1.0
    poly: float = 0.0
    bitcrush: int = 0
    clip: float = 1.0
    pan: float = 0.0
    ch_delay: float = 0.0
    ch_gain_db: float = 0.0

    def tags(self) -> dict:
        return {
            "r": round(self.r, 5), "mode": self.mode, "tilt": round(self.tilt, 5),
            "pitch": round(self.pitch_semi, 3), "gain_db": round(self.gain_db, 3),
            "wet": round(self.wet, 3), "drive": round(self.drive, 3),
            "poly": round(self.poly, 4), "clip": round(self.clip, 3),
            "pan": round(self.pan, 3),
        }


@dataclass
class DistortRange:
    """采样区间。数值型一律 [lo, hi]; 概率型为 0..1。"""
    r: tuple[float, float] = (1.0, 1.0)
    pitch_semi: tuple[float, float] = (0.0, 0.0)
    modes: tuple[str, ...] = ("resample",)
    tilt: tuple[float, float] = (0.0, 0.0)
    piece_prob: float = 0.0
    delay_span: float = 0.25           # 相对窗口起点的随机范围 (秒, 双向)
    gain_db: tuple[float, float] = (0.0, 0.0)
    n_bands: int = 0
    eq_db: tuple[float, float] = (0.0, 0.0)
    eq_prob: float = 0.0
    hi_rolloff_db: tuple[float, float] = (0.0, 0.0)
    reverb_prob: float = 0.0
    wet: tuple[float, float] = (0.1, 0.6)
    rt60: tuple[float, float] = (0.1, 0.8)
    sat_prob: float = 0.0
    drive: tuple[float, float] = (1.2, 3.0)
    poly_prob: float = 0.0
    poly: tuple[float, float] = (0.02, 0.15)
    crush_prob: float = 0.0
    clip_prob: float = 0.0
    clip: tuple[float, float] = (0.5, 0.95)
    noise_db: tuple[float, float] = (-120.0, -120.0)
    stereo: bool = False


# ============================================================== 拉伸实现
def stretch_tilt(x: np.ndarray, tilt: float, sr: int = SR) -> np.ndarray:
    """线性倾斜: 局部速度 1 + tilt*(2u-1), u=t/L。段首减速/段尾加速 (或反之)。

    实现: 对每条输出样本求源位置 t_src = L*(u + tilt*(u^2-u)), 再线性插值。
    tilt=0 时退化为恒等 (逐样本精确, 无插值误差)。
    """
    if abs(tilt) < 1e-9:
        return x
    n = x.size
    if n < 4:
        return x
    u = np.arange(n, dtype=np.float64) / (n - 1)
    src = (n - 1) * (u + tilt * (u * u - u))
    src = np.clip(src, 0.0, n - 1.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0).astype(np.float32)
    return (x[i0] * (1.0 - w) + x[i1] * w).astype(np.float32)


def stretch_piecewise(x: np.ndarray, pieces: tuple[float, ...], sr: int = SR) -> np.ndarray:
    """分段常速: 把 x 切成 k 段, 每段以不同比例拉伸/压缩后拼回。"""
    k = len(pieces)
    if k <= 1 or x.size < k * 32:
        return x
    ratios = np.asarray(pieces, dtype=np.float64)
    ratios = ratios / np.mean(ratios)                      # 保持总时长近似不变
    bounds = np.linspace(0, x.size, k + 1).astype(np.int64)
    outs = []
    for i in range(k):
        seg = x[bounds[i]:bounds[i + 1]]
        if seg.size < 16:
            outs.append(seg)
            continue
        m = max(16, int(round(seg.size / max(0.5, ratios[i]))))
        src = np.linspace(0.0, seg.size - 1.0, m)
        i0 = np.floor(src).astype(np.int64)
        i1 = np.minimum(i0 + 1, seg.size - 1)
        w = (src - i0).astype(np.float32)
        outs.append((seg[i0] * (1 - w) + seg[i1] * w).astype(np.float32))
    y = np.concatenate(outs) if outs else x
    return y


def _wsola(x: np.ndarray, ratio: float, sr: int, frame_ms: float = 40.0,
           overlap: float = 0.5, search_ms: float = 8.0) -> np.ndarray:
    """WSOLA 变速不变调。ratio = 输出/输入 时长比 (>1 变慢)。

    相似度搜索用 FFT 卷积算归一化互相关 —— 直接 np.correlate 在 40ms 参考上
    是 O(n*m) 纯 Python 循环, 实测占整个渲染 50% 时间。
    """
    if abs(ratio - 1.0) < 1e-3 or x.size < int(sr * 0.08):
        return x
    n = int(sr * frame_ms / 1000.0)
    n = max(64, n - (n % 2))
    hop_s = int(n * overlap)
    hop_a = max(1, int(round(hop_s / ratio)))
    search = max(1, int(sr * search_ms / 1000.0))
    win = np.hanning(n).astype(np.float32)
    out_len = int(round(x.size * ratio)) + n
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    ref = None
    a = 0
    o = 0
    # 手写 FFT 相关: 每帧 2 次 rfft + 1 次 irfft, 比 scipy.signal.fftconvolve
    # 每次都重算两路 FFT + 多次拷贝快 3~5 倍 (实测占渲染时间 50%+ 的热点)。
    nf = None
    R = None
    ref_len = 0
    while o + n < out_len and a + n + search < x.size:
        if ref is None or ref.size == 0:
            off = 0
        else:
            lo = max(0, a - search)
            hi = min(x.size - n, a + search)
            if hi <= lo:
                off = 0
            else:
                seg = x[lo:hi + n]
                need = seg.size + ref.size
                if nf is None or nf < need:
                    nf = 1 << int(np.ceil(np.log2(max(8, need))))
                    R = None
                if R is None or ref_len != ref.size:
                    R = _rfft(ref[::-1], n=nf)      # 参考 FFT 只算一次, 循环内复用
                    ref_len = ref.size
                S = _rfft(seg, n=nf)
                cc = _irfft(S * R, n=nf)[ref.size - 1:ref.size - 1 + (hi - lo) + 1]
                off = (int(np.argmax(np.abs(cc))) if cc.size else 0) + lo - a
        a2 = int(np.clip(a + off, 0, max(0, x.size - n)))
        out[o:o + n] += x[a2:a2 + n] * win
        norm[o:o + n] += win
        end = a2 + hop_s + hop_s
        ref = x[a2 + hop_s:end][::-1].copy() if end <= x.size else None
        a += hop_a
        o += hop_s
    m = norm > 1e-6
    out[m] /= norm[m]
    return out[:int(round(x.size * ratio))].astype(np.float32)


def stretch(x: np.ndarray, p: TParams, sr: int = SR) -> np.ndarray:
    """按参数施加时间/音高变换。返回长度可能与输入不同。"""
    y = x
    # 1) 独立变调 -> 用 resample 实现音高改变, 再用 WSOLA 把时长还原
    if abs(p.pitch_semi) > 1e-4:
        k = 2.0 ** (p.pitch_semi / 12.0)
        y = resample(y, sr, int(round(sr * k)))
        y = _wsola(y, 1.0 / k, sr)                        # 拉回原时长
    # 2) 主拉伸
    if p.mode == "resample":
        if abs(p.r - 1.0) > 1e-6:
            # r>1 = 变快变高: 先按 r 重采样再按 1/r 拉回时长? 不 —— resample 语义即
            # "播放采样率改变", 时长与音高同时变; 这里直接重采样, 时长随之改变。
            y = resample(y, sr, int(round(sr * max(0.25, p.r))))
    elif p.mode == "wsola":
        y = _wsola(y, p.r, sr)                            # 时长变, 音高不变
    elif p.mode == "tilt":
        y = stretch_tilt(y, p.tilt, sr)
        if abs(p.r - 1.0) > 1e-6:
            y = _wsola(y, p.r, sr)
    elif p.mode == "piecewise":
        y = stretch_piecewise(y, p.piece or (1.0,), sr)
        if abs(p.r - 1.0) > 1e-6:
            y = _wsola(y, p.r, sr)
    return np.ascontiguousarray(y, dtype=np.float32)


# ============================================================== 频响 / 非线性
def mel_band_gains(n_bands: int, sr: int = SR, fmin: float = 40.0,
                   fmax: float = 20000.0) -> np.ndarray:
    """mel 频带中心频率 (Hz), 长度 n_bands。合成与可微层共用同一套中心。"""
    if n_bands <= 0:
        return np.zeros(0, dtype=np.float32)
    m_lo, m_hi = _hz2mel(fmin), _hz2mel(fmax)
    m = np.linspace(m_lo, m_hi, n_bands + 2)
    return _mel2hz(m[1:-1]).astype(np.float32)


def _hz2mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel2hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_band_weights(centers: np.ndarray, n_fft: int, sr: int = SR,
                     sigma_ratio: float = 1.0) -> np.ndarray:
    """三角/高斯混合的软频带权重 [B, n_fft//2+1], 行和为 1 (能量守恒)。"""
    n_bins = n_fft // 2 + 1
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    if centers.size == 0:
        return np.zeros((0, n_bins), dtype=np.float32)
    c = centers.astype(np.float64)
    if c.size == 1:
        w = np.ones((1, n_bins), dtype=np.float64)
    else:
        # 相邻中心的对数间距 (对数轴上均匀)
        gaps = np.diff(np.log(c))
        sig = np.empty_like(c)
        sig[1:-1] = np.maximum(gaps[:-1], gaps[1:]) * 0.5
        sig[0] = gaps[0] * 0.5
        sig[-1] = gaps[-1] * 0.5
        sig *= sigma_ratio
        lf = np.log(np.maximum(f, 1e-6))[None, :]
        lc = np.log(c)[:, None]
        w = np.exp(-0.5 * ((lf - lc) / sig[:, None]) ** 2)
    s = w.sum(axis=1, keepdims=True)
    return (w / np.maximum(s, 1e-12)).astype(np.float32)


def apply_eq(x: np.ndarray, eq_db: np.ndarray, centers: np.ndarray,
             n_fft: int = 512, sr: int = SR) -> np.ndarray:
    """mel 频带增益 (重叠相加, 幅度-相位精确)。eq_db 全 0 时原样返回。

    n_fft=512 (约 94Hz 分辨率) 足够表达 mel 频带包络, 且 hop=128 时向量化后
    每帧开销 ~20us —— 这是数据生成路径的热点, 必须小窗。
    """
    if eq_db is None or eq_db.size == 0 or not np.any(np.abs(eq_db) > 1e-6):
        return x
    W = mel_band_weights(centers, n_fft, sr)                # [B, bins]
    g = (10.0 ** (np.asarray(eq_db, dtype=np.float32) / 20.0))[:, None]   # [B,1]
    band_gain = (W * g).sum(axis=0)                          # [bins] 恒等时为 1
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)
    n = x.size
    pad = n_fft
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad + n_fft, np.float32)])
    nf = 1 + (xp.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    S = _rfft(xp[idx] * win, axis=1)                   # [nf, bins] 一次批量 FFT
    Y = _irfft(S * band_gain, n=n_fft, axis=1) * win   # [nf, n_fft]
    out = np.zeros(xp.size, dtype=np.float32)
    np.add.at(out, idx.ravel(), Y.ravel())
    wn = np.zeros(xp.size, dtype=np.float32)
    np.add.at(wn, idx.ravel(), np.broadcast_to(win * win, (nf, n_fft)).ravel())
    m = wn > 1e-8
    out[m] /= wn[m]
    return out[pad:pad + n].astype(np.float32)


def hi_rolloff(x: np.ndarray, db: float, sr: int = SR, knee_hz: float = 9000.0) -> np.ndarray:
    """高频滚降 (一阶)。用频域掩码 + 小窗 OLA, 与 apply_eq 同款向量化实现。"""
    if abs(db) < 0.05:
        return x
    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    g = np.ones_like(f, dtype=np.float32)
    sel = f > knee_hz
    if sel.any():
        t = (f[sel] - knee_hz) / max(1.0, sr / 2 - knee_hz)
        g[sel] = (10.0 ** (db / 20.0)) ** t
    n = x.size
    pad = n_fft
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])
    nf = 1 + (xp.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    S = _rfft(xp[idx] * win, axis=1)
    Y = _irfft(S * g, n=n_fft, axis=1) * win
    out = np.zeros(xp.size, dtype=np.float32)
    np.add.at(out, idx.ravel(), Y.ravel())
    wn = np.zeros(xp.size, dtype=np.float32)
    np.add.at(wn, idx.ravel(), np.broadcast_to(win * win, (nf, n_fft)).ravel())
    m = wn > 1e-8
    out[m] /= wn[m]
    return out[pad:pad + n].astype(np.float32)


def apply_reverb(x: np.ndarray, rir: np.ndarray, wet: float) -> np.ndarray:
    """干湿混合的卷积混响。rir 已归一到单位能量。"""
    if rir is None or wet <= 0:
        return x
    from scipy.signal import fftconvolve
    y = fftconvolve(x, rir.astype(np.float32), mode="full")[:x.size].astype(np.float32)
    e = float(np.sqrt(np.mean(rir.astype(np.float64) ** 2)))
    if e > 1e-9:
        y /= e
    w = float(np.clip(wet, 0.0, 1.0))
    return ((1.0 - w) * x + w * y).astype(np.float32)


def apply_nonlinear(x: np.ndarray, drive: float = 1.0, poly: float = 0.0,
                    bitcrush: int = 0, clip: float = 1.0) -> np.ndarray:
    """饱和 + 多项式 + 位深缩减 + 削波 (顺序即游戏/转码链的典型顺序)。"""
    y = x
    if drive > 1.001:
        d = np.tanh(drive)
        y = (np.tanh(y * drive) / d).astype(np.float32)
    if abs(poly) > 1e-6:
        y = (y + poly * (y ** 3)).astype(np.float32)
    if bitcrush > 0:
        lev = 2.0 ** bitcrush
        y = (np.round(y * lev) / lev).astype(np.float32)
    if clip < 0.999:
        y = np.clip(y, -clip, clip).astype(np.float32)
    return y


def add_noise_seeded(x: np.ndarray, noise_db: float, seed: int,
                     sr: int = SR) -> np.ndarray:
    """给某实例叠加底噪。噪声序列由 seed 决定 -> 渲染完全可复现。

    底噪是【加性分量】, 不参与原子的线性变换, 因此它对反相抵消是不可消除的
    上界来源 —— 必须显式建模, 不能藏进原子参数里。
    """
    if noise_db <= -119.0 or x.size == 0:
        return x
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = rng.standard_normal(x.size).astype(np.float32)
    n = _one_pole_lp(n, 0.35)
    cur = float(np.sqrt(np.mean(n.astype(np.float64) ** 2)) + 1e-12)
    tgt = 10.0 ** (noise_db / 20.0)
    return (x + n * (tgt / cur)).astype(np.float32)


def _one_pole_lp(x: np.ndarray, a: float) -> np.ndarray:
    """一阶低通 (a 越小越暗)。用 lfilter 避免 Python 循环。"""
    from scipy.signal import lfilter
    a = float(np.clip(a, 0.0, 0.999))
    return lfilter([1.0 - a], [1.0, -a], x).astype(np.float32)


# ============================================================== 采样
def sample_params(rng: np.random.Generator, dr: DistortRange, *, n_bands: int = 0,
                  centers: np.ndarray | None = None, sr: int = SR,
                  delay_span: float | None = None) -> tuple[TParams, float]:
    """按 DistortRange 采一组变换参数。返回 (TParams, noise_db)。"""
    mode = str(rng.choice(dr.modes)) if dr.modes else "resample"
    p = TParams(mode=mode)
    p.r = float(rng.uniform(*dr.r))
    if dr.pitch_semi != (0.0, 0.0):
        p.pitch_semi = float(rng.uniform(*dr.pitch_semi))
    if dr.tilt != (0.0, 0.0):
        t = float(rng.uniform(*dr.tilt))
        p.tilt = t * float(rng.choice([-1.0, 1.0]))
    if mode == "piecewise" and rng.random() < dr.piece_prob:
        k = int(rng.integers(2, 4))
        p.piece = tuple(float(v) for v in rng.uniform(0.92, 1.08, size=k))
    span = dr.delay_span if delay_span is None else delay_span
    p.delay = float(rng.uniform(-span, span))
    p.gain_db = float(rng.uniform(*dr.gain_db))
    nb = dr.n_bands if n_bands <= 0 else n_bands
    if nb > 0 and rng.random() < max(dr.eq_prob, 1e-9):
        p.n_bands = nb
        e = rng.uniform(*dr.eq_db, size=nb).astype(np.float32)
        # 平滑 (相邻频带相关), 更接近真实频响
        if nb >= 3:
            k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            e = np.convolve(e, k, mode="same").astype(np.float32)
        if dr.hi_rolloff_db != (0.0, 0.0):
            e[-max(1, nb // 4):] += float(rng.uniform(*dr.hi_rolloff_db))
        p.eq_db = e
    if rng.random() < dr.reverb_prob:
        p.wet = float(rng.uniform(*dr.wet))
        p.rt60 = float(rng.uniform(*dr.rt60))
    if rng.random() < dr.sat_prob:
        p.drive = float(rng.uniform(*dr.drive))
    if rng.random() < dr.poly_prob:
        p.poly = float(rng.uniform(*dr.poly)) * float(rng.choice([-1.0, 1.0]))
    if rng.random() < dr.crush_prob:
        p.bitcrush = int(rng.integers(6, 10))
    if rng.random() < dr.clip_prob:
        p.clip = float(rng.uniform(*dr.clip))
    noise_db = float(rng.uniform(*dr.noise_db))
    if dr.stereo:
        p.pan = float(rng.uniform(-1.0, 1.0))
        p.ch_delay = float(rng.uniform(0.0, 0.001))
        p.ch_gain_db = float(rng.uniform(-4.0, 4.0))
    return p, noise_db


def apply_distortions(x: np.ndarray, p: TParams, sr: int = SR, *,
                      rng: np.random.Generator | None = None,
                      centers: np.ndarray | None = None,
                      noise_db: float = -120.0, noise_seed: int = 0) -> np.ndarray:
    """把 TParams 施加到原始波形 (原始 sr 输入, 输出同 sr)。

    顺序与推理侧可微变换层严格一致:
        拉伸/重采样 -> EQ -> 混响 -> 增益 -> 非线性 -> [底噪]

    底噪必须给 noise_seed; 给了 noise_db 却不给种子会导致不可复现渲染。
    """
    rng = rng or np.random.default_rng(0)
    y = stretch(np.asarray(x, dtype=np.float32), p, sr)
    if p.eq_db is not None and centers is not None and p.eq_db.size:
        y = apply_eq(y, p.eq_db, centers, sr=sr)
    if p.rir is not None and p.wet > 0:
        y = apply_reverb(y, p.rir, p.wet)
    if abs(p.gain_db) > 1e-6:
        y = apply_gain(y, p.gain_db)
    y = apply_nonlinear(y, p.drive, p.poly, p.bitcrush, p.clip)
    if noise_db > -119.0:
        y = add_noise_seeded(y, noise_db, noise_seed, sr)
    return np.ascontiguousarray(y, dtype=np.float32)
