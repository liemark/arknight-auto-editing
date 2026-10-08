"""难度分级与混合权重漂移。

用户要求: "混杂简单的到非常复杂的例子, 这样模型可以先从简单的开始训练"。
因此不是"分阶段只训某一档", 而是:
  * 每个 batch 内【逐样本】按当前权重抽难度 -> 同 batch 天然混有 D0 与 D4;
  * 权重随训练进度线性漂移 -> 前期简单样本占多数(低方差梯度先学对前端/检索),
    后期复杂样本占多数(逼近真实难度);
  * 复杂样本从第 0 步就在场, 不会"训练到最后才第一次见到拉伸/混响";
  * 损失按难度归一化(在 losses 里做), 防 D4 的大残差主导总损失。

D0~D4 的划分依据是"同时叠加的难点数量", 不是单一维度:
  D0 单原子干净 | D1 少量+分数时延 | D2 中等+频响失真 | D3 大量+拉伸混响 | D4 极端+自重复+近邻
"""
from __future__ import annotations

import numpy as np

from .distort import DistortRange

# 难度名 -> 定义
LEVELS: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4")
N_LEVELS = len(LEVELS)


class LevelSpec:
    """一个难度档的完整采样定义。"""

    def __init__(self, name: str, *, n_atoms: tuple[int, int], dr: DistortRange,
                 bg_prob: float, bg_db: tuple[float, float],
                 interf: tuple[int, int], interf_db: tuple[float, float],
                 self_repeat: float, repeat_range: tuple[int, int],
                 hard_neighbor: float, hard_neighbor_range: tuple[int, int],
                 master_gain_db: tuple[float, float],
                 codec_prob: float, limiter_prob: float,
                 window: float = 10.0):
        self.name = name
        self.n_atoms = n_atoms
        self.dr = dr
        self.bg_prob = bg_prob
        self.bg_db = bg_db
        self.interf = interf
        self.interf_db = interf_db
        self.self_repeat = self_repeat              # 短原子自重复概率
        self.repeat_range = repeat_range            # 重复次数范围
        self.hard_neighbor = hard_neighbor          # 同组近邻混入概率 (每个原子)
        self.hard_neighbor_range = hard_neighbor_range
        self.master_gain_db = master_gain_db
        self.codec_prob = codec_prob
        self.limiter_prob = limiter_prob
        self.window = window

    def __repr__(self) -> str:
        return (f"<{self.name} n={self.n_atoms} bg={self.bg_prob} "
                f"rep={self.self_repeat} nb={self.hard_neighbor}>")


def build_levels(window: float = 10.0, n_bands: int = 16) -> dict[str, LevelSpec]:
    """五档难度定义。n_bands 是 EQ 频带数 (与模型输出维度一致)。

    【噪声/干扰幅度已按"真实录像不会太离谱"上调】早期版本偏极端：
      逐实例增益压到 -30 dB、底噪到 -3 dB、干扰电平到 -16 dB
      （干扰几乎和信号同能量）。真实录像里被反相抵消的音效是**听得见的主成分**，
      底噪也远低于信号。太极端的数据会让模型先学"怎么在噪声里存活"，
      而不是学"怎么把音效对齐"，训练又慢又难收敛。
      所以统一把 gain_db / noise_db / interf_db 的**下界**调高 8~12 dB，
      上界基本不动（保留少量困难样本）。想复现旧分布就把它调回去。
    """
    clean = DistortRange(gain_db=(-12.0, 3.0), delay_span=0.25, n_bands=0)
    light = DistortRange(
        r=(0.99, 1.01), modes=("resample",), delay_span=0.25,
        gain_db=(-14.0, 3.0), n_bands=n_bands, eq_prob=0.35, eq_db=(-3.0, 3.0),
        hi_rolloff_db=(-4.0, 0.0), noise_db=(-60.0, -40.0), stereo=True)
    mid = DistortRange(
        r=(0.97, 1.03), modes=("resample", "wsola"), delay_span=0.25,
        gain_db=(-18.0, 3.0), n_bands=n_bands, eq_prob=0.6, eq_db=(-5.0, 5.0),
        hi_rolloff_db=(-6.0, 0.0), reverb_prob=0.25, wet=(0.1, 0.4), rt60=(0.1, 0.5),
        sat_prob=0.25, drive=(1.2, 2.5), clip_prob=0.15, clip=(0.6, 0.95),
        noise_db=(-52.0, -30.0), stereo=True)
    hard = DistortRange(
        r=(0.94, 1.06), modes=("resample", "wsola", "tilt", "piecewise"),
        tilt=(0.0, 0.03), piece_prob=0.5, delay_span=0.25, gain_db=(-22.0, 3.0),
        n_bands=n_bands, eq_prob=0.8, eq_db=(-7.0, 7.0), hi_rolloff_db=(-9.0, 0.0),
        reverb_prob=0.45, wet=(0.1, 0.6), rt60=(0.1, 1.0),
        sat_prob=0.4, drive=(1.3, 3.0), poly_prob=0.2, poly=(0.02, 0.12),
        crush_prob=0.1, clip_prob=0.3, clip=(0.5, 0.9),
        noise_db=(-42.0, -24.0), stereo=True)
    extreme = DistortRange(
        r=(0.90, 1.10), modes=("resample", "wsola", "tilt", "piecewise"),
        pitch_semi=(-1.5, 1.5), tilt=(0.0, 0.05), piece_prob=0.7, delay_span=0.25,
        gain_db=(-24.0, 3.0), n_bands=n_bands, eq_prob=0.9, eq_db=(-8.0, 8.0),
        hi_rolloff_db=(-12.0, 0.0), reverb_prob=0.6, wet=(0.1, 0.75), rt60=(0.05, 1.5),
        sat_prob=0.5, drive=(1.4, 4.5), poly_prob=0.35, poly=(0.02, 0.15),
        crush_prob=0.2, clip_prob=0.4, clip=(0.4, 0.9),
        noise_db=(-38.0, -20.0), stereo=True)

    return {
        #   name  n_atoms        dr      bg   bg_db           interf      interf_db      selfrep rep    hardnb  hnb_rng  master        codec limiter
        "D0": LevelSpec("D0", n_atoms=(1, 1), dr=clean, bg_prob=0.7, bg_db=(-40.0, -26.0),
                        interf=(0, 0), interf_db=(-60.0, -60.0), self_repeat=0.0,
                        repeat_range=(2, 2), hard_neighbor=0.0, hard_neighbor_range=(1, 1),
                        master_gain_db=(-6.0, 0.0), codec_prob=0.0, limiter_prob=0.0, window=window),
        "D1": LevelSpec("D1", n_atoms=(2, 4), dr=light, bg_prob=0.8, bg_db=(-38.0, -25.0),
                        interf=(0, 1), interf_db=(-48.0, -34.0), self_repeat=0.05,
                        repeat_range=(2, 3), hard_neighbor=0.05, hard_neighbor_range=(1, 1),
                        master_gain_db=(-6.0, 0.0), codec_prob=0.1, limiter_prob=0.1, window=window),
        "D2": LevelSpec("D2", n_atoms=(6, 10), dr=mid, bg_prob=0.85, bg_db=(-38.0, -24.0),
                        interf=(0, 1), interf_db=(-44.0, -30.0), self_repeat=0.25,
                        repeat_range=(2, 4), hard_neighbor=0.5, hard_neighbor_range=(1, 2),
                        master_gain_db=(-6.0, 0.0), codec_prob=0.3, limiter_prob=0.3, window=window),
        "D3": LevelSpec("D3", n_atoms=(12, 18), dr=hard, bg_prob=0.9, bg_db=(-36.0, -22.0),
                        interf=(1, 2), interf_db=(-40.0, -26.0), self_repeat=0.45,
                        repeat_range=(2, 5), hard_neighbor=0.8, hard_neighbor_range=(1, 3),
                        master_gain_db=(-4.0, 0.0), codec_prob=0.5, limiter_prob=0.15, window=window),
        "D4": LevelSpec("D4", n_atoms=(20, 28), dr=extreme, bg_prob=0.95, bg_db=(-34.0, -20.0),
                        interf=(1, 3), interf_db=(-36.0, -22.0), self_repeat=0.6,
                        repeat_range=(2, 5), hard_neighbor=1.0, hard_neighbor_range=(1, 3),
                        master_gain_db=(-6.0, 0.0), codec_prob=0.5, limiter_prob=0.15, window=window),
    }


def level_weights(progress: float, *,
                  start: tuple[float, ...] = (0.40, 0.30, 0.18, 0.09, 0.03),
                  end: tuple[float, ...] = (0.15, 0.20, 0.25, 0.25, 0.15),
                  ) -> np.ndarray:
    """按训练进度 p∈[0,1] 线性漂移的难度权重。"""
    p = float(np.clip(progress, 0.0, 1.0))
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    w = s * (1.0 - p) + e * p
    return w / w.sum()


def sample_level(rng: np.random.Generator, weights: np.ndarray) -> str:
    return LEVELS[int(rng.choice(N_LEVELS, p=weights / weights.sum()))]


def sample_span(rng: np.random.Generator, span: float, *, bias_center: bool = True
                ) -> float:
    """按区间采样一个延迟。bias_center=True 时向窗口中心偏置 (真实事件更常落在中间)。"""
    if bias_center:
        u = rng.beta(2.0, 2.0)
    else:
        u = rng.random()
    return float((u * 2.0 - 1.0) * span)


def difficulty_histogram(weights: np.ndarray, n: int = 20000) -> dict[str, int]:
    return {lv: int(round(w * n)) for lv, w in zip(LEVELS, weights)}
