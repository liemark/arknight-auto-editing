"""合成与评估: 把事件列表渲染成反相轨, 并计算抵消指标。

反相轨的最终形式就是  -Σ_i T_θ_i(a_i) , 与原始音频相加即得残差。
本模块同时提供:
  * render_events : 事件列表 -> 反相波形 (numpy, 推理侧权威实现)
  * psr / residual_metrics : 评估指标 (PSR = 原/残 功率比)
  * ideal_cancel : 用真值渲染的反相轨 = 理想上界 (试听导出与评估都要用)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import SR, resample
from ..synth.distort import TParams, apply_distortions, mel_band_gains


@dataclass
class Event:
    """一条被采纳的原子实例。"""
    atom_id: int
    delay: float                 # 秒 (相对窗口/整段起点)
    gain_db: float = 0.0
    r: float = 1.0
    tilt: float = 0.0
    pitch_semi: float = 0.0
    eq_db: np.ndarray | None = None
    wet: float = 0.0
    drive: float = 1.0
    rir: np.ndarray | None = None
    score: float = 1.0           # 置信度 (重排概率或验证后的 PSR 增益)
    source: str = "model"        # model | refine | truth
    level_hint: str = ""

    def to_params(self) -> TParams:
        return TParams(r=self.r, mode="resample", tilt=self.tilt,
                       pitch_semi=self.pitch_semi, delay=self.delay,
                       gain_db=self.gain_db, eq_db=self.eq_db,
                       n_bands=0 if self.eq_db is None else self.eq_db.size,
                       rir=self.rir, wet=self.wet, drive=self.drive)

    def to_dict(self) -> dict:
        return {"atom_id": int(self.atom_id), "delay": round(float(self.delay), 6),
                "gain_db": round(float(self.gain_db), 3), "r": round(float(self.r), 6),
                "tilt": round(float(self.tilt), 5),
                "pitch_semi": round(float(self.pitch_semi), 3),
                "eq_db": None if self.eq_db is None else
                [round(float(v), 2) for v in np.asarray(self.eq_db).ravel()],
                "wet": round(float(self.wet), 3), "drive": round(float(self.drive), 3),
                "score": round(float(self.score), 4), "source": self.source}

    @staticmethod
    def from_dict(d: dict) -> "Event":
        eq = d.get("eq_db")
        return Event(atom_id=int(d["atom_id"]), delay=float(d["delay"]),
                     gain_db=float(d.get("gain_db", 0.0)), r=float(d.get("r", 1.0)),
                     tilt=float(d.get("tilt", 0.0)),
                     pitch_semi=float(d.get("pitch_semi", 0.0)),
                     eq_db=None if eq is None else np.asarray(eq, dtype=np.float32),
                     wet=float(d.get("wet", 0.0)), drive=float(d.get("drive", 1.0)),
                     score=float(d.get("score", 1.0)),
                     source=str(d.get("source", "model")))


# --------------------------------------------------------------------- 渲染
def render_events(lib, events: list[Event], n_samples: int, *, sr: int = SR,
                  centers: np.ndarray | None = None, n_bands: int = 16,
                  invert: bool = True, rng=None, atom_rms_db: float = -34.0,
                  normalize: bool = True) -> np.ndarray:
    """事件列表 -> 波形。invert=True 时返回【反相轨】(可直接与原音相加)。

    这是推理侧的权威渲染实现, 与 synth/mixer.py 的合成链必须逐项一致:
      1) 原子先做 RMS 归一化 (同样的 atom_rms_db);
      2) 【Event.gain_db 是绝对增益】—— 已含混音总线主增益与防削波常数缩放。
         本函数【不再】接受 master_gain_db: 早期两者同时给, 主增益被应用两次,
         在 D0 单原子场景就造成 8.9e-3 的模板幅度错配 (相消变增强)。
         标签侧 (dataset.labels_from_result) 已统一成同一口径。
    """
    from ..audio import rms_normalize_db
    centers = mel_band_gains(n_bands, sr) if centers is None else centers
    out = np.zeros(n_samples, dtype=np.float32)
    for ev in events:
        x, asr = lib.raw(ev.atom_id)
        if normalize:
            x = rms_normalize_db(x, atom_rms_db)
        if asr != sr:
            x = resample(x, asr, sr)
        p = ev.to_params()
        if p.eq_db is None or p.n_bands != centers.size:
            p.n_bands = centers.size
        y = apply_distortions(x, p, sr, centers=centers, rng=rng)
        s0 = int(round(ev.delay * sr))
        a, b = max(0, s0), min(n_samples, s0 + y.size)
        if b > a:
            out[a:b] += y[a - s0:b - s0]
    return -out if invert else out


# --------------------------------------------------------------------- 指标
def residual_metrics(mix: np.ndarray, cancel: np.ndarray, *,
                     segments: list[tuple[float, float, float]] | None = None,
                     sr: int = SR, eps: float = 1e-12) -> dict:
    """整体 + 逐段 PSR。

    segments: [(起, 止, 该段能量占比)] 通常传真值事件的 [start, end, 1.0]。
    """
    mix = np.asarray(mix, dtype=np.float64)
    cancel = np.asarray(cancel, dtype=np.float64)
    res = mix + cancel
    p_mix = float((mix ** 2).sum()) + eps
    p_res = float((res ** 2).sum()) + eps
    out = {
        "psr_db": 10.0 * np.log10(p_mix / p_res),
        "rms_mix_db": 10.0 * np.log10(p_mix / max(1, mix.size)),
        "rms_res_db": 10.0 * np.log10(p_res / max(1, mix.size)),
        "peak_mix": float(np.max(np.abs(mix))) if mix.size else 0.0,
        "peak_res": float(np.max(np.abs(res))) if res.size else 0.0,
    }
    if segments:
        ps = []
        for a, b, _w in segments:
            i0, i1 = max(0, int(a * sr)), min(mix.size, int(b * sr))
            if i1 <= i0:
                continue
            pm = float((mix[i0:i1] ** 2).sum()) + eps
            pr = float((res[i0:i1] ** 2).sum()) + eps
            ps.append(10.0 * np.log10(pm / pr))
        if ps:
            out["psr_seg_median_db"] = float(np.median(ps))
            out["psr_seg_p10_db"] = float(np.percentile(ps, 10))
            out["psr_seg_min_db"] = float(np.min(ps))
            out["n_segments"] = len(ps)
    return out


def band_psr(mix: np.ndarray, cancel: np.ndarray, *, sr: int = SR,
             n_fft: int = 4096, bands: tuple[tuple[float, float], ...] = (
                 (20, 500), (500, 2000), (2000, 8000), (8000, 16000),
                 (16000, 24000))) -> dict[str, float]:
    """分子带 PSR —— 用来暴露"高频抵消不掉"这类问题 (语音 16k 上限)。"""
    from scipy.signal import welch
    res = np.asarray(mix, dtype=np.float64) + np.asarray(cancel, dtype=np.float64)
    f, Pm = welch(mix, fs=sr, nperseg=n_fft)
    _, Pr = welch(res, fs=sr, nperseg=n_fft)
    out = {}
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        a = float(Pm[m].sum()) + 1e-20
        b = float(Pr[m].sum()) + 1e-20
        out[f"{lo:.0f}-{hi:.0f}Hz"] = 10.0 * np.log10(a / b)
    return out


def event_segments(events: list[Event], lib, *, pad: float = 0.02,
                   sr: int = SR) -> list[tuple[float, float, float]]:
    """事件 -> [(起, 止, 1.0)] 供逐段 PSR 使用。"""
    segs = []
    for ev in events:
        d = lib.dur(ev.atom_id) / max(1e-3, ev.r)
        segs.append((max(0.0, ev.delay - pad), ev.delay + d + pad, 1.0))
    return segs


# ------------------------------------------------------------------ 理想上界
def ideal_cancel(lib, spec: dict, n_samples: int, *, sr: int = SR,
                 n_bands: int = 16) -> np.ndarray:
    """用真值 (atom_ids/delays/params) 渲染的完美反相轨 = 抵消上界。

    spec 由 dataset.py 的 pack 提供, 这里只接受序列化后的 dict。
    """
    ids = spec.get("atom_ids") or []
    delays = spec.get("delays") or []
    eqs = spec.get("eq_db") or [None] * len(ids)
    rs = spec.get("r") or [1.0] * len(ids)
    gains = spec.get("gain_db") or [0.0] * len(ids)
    tilts = spec.get("tilt") or [0.0] * len(ids)
    pitches = spec.get("pitch_semi") or [0.0] * len(ids)
    wets = spec.get("wet") or [0.0] * len(ids)
    drives = spec.get("drive") or [1.0] * len(ids)
    evs = []
    for i, aid in enumerate(ids):
        evs.append(Event(atom_id=int(aid), delay=float(delays[i]),
                         gain_db=float(gains[i]), r=float(rs[i]),
                         tilt=float(tilts[i]), pitch_semi=float(pitches[i]),
                         eq_db=None if eqs[i] is None else np.asarray(eqs[i], np.float32),
                         wet=float(wets[i]), drive=float(drives[i]), source="truth"))
    return render_events(lib, evs, n_samples, sr=sr, n_bands=n_bands)
