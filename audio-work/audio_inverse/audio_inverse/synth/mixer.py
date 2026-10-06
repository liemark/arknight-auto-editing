"""混合器: 把原子按难度档叠成一段"视频音频", 并给出逐实例真值。

核心难点 (用户强调): 高度混叠 + 短模板自重复
  * 每窗 ~20 条模板重叠 (D4: 20~28 条);
  * 短原子 (<0.6s) 高概率在同窗内重复 2~5 次, 含"等间隔(脚步)"与"随机抖动"两种;
  * 强制混入同组近邻 (p_imp_3 的兄弟音效) 作为困难负样本;
  * 允许完全同起点的两条相同原子 (同一帧两个单位放同一技能)。

真值输出是【变换后】的参数, 而不是原始参数 —— 因为要拿去监督模型的参数回归。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..audio import SR, resample
from ..atomlib import AtomLib
from ..manifest import Manifest
from .curriculum import LevelSpec, build_levels, sample_span
from .distort import TParams, apply_distortions, mel_band_gains, sample_params
from .rir import IRPool

# 实例标签里要回归的连续参数 (顺序即模型输出顺序)
PARAM_NAMES = (
    "delay",       # 相对窗口起点的秒数 (精对齐后应逼近)
    "gain_db",
    "r_minus1",    # r - 1
    "tilt",
    "pitch_semi",
    "log10_dur",   # 变换后时长 (log10), 辅助一致性
)
MAX_EQ_BANDS = 16


@dataclass
class Instance:
    atom_id: int
    delay: float
    p: TParams
    gain_db: float
    dur: float              # 变换后时长 (秒)
    rep_index: int = 0      # 第几次重复 (自重复序号, 0 = 首次)
    rep_total: int = 1
    is_hard_neighbor: bool = False
    group: str = ""
    noise_db: float = -120.0   # 该实例的底噪电平 (加性, 模型要回归它)
    noise_seed: int = 0        # 底噪随机种子 (保证渲染可复现)
    exact_dup: bool = False    # 是否与前一次完全同起点 (同一帧两个单位放同一技能)

    def tid(self) -> str:
        return f"{self.atom_id}#{self.rep_index}@{(self.atom_id % 9973)}"


@dataclass
class MixResult:
    mix: np.ndarray                 # [T] 48k 最终混合 (单声道, 模型输入)
    dry: np.ndarray                 # [T] 所有原子干声之和(无背景无主增益) — 理想上界用
    background: np.ndarray          # [T] 背景轨真值
    interference: np.ndarray        # [T] 干扰轨真值
    instances: list[Instance]
    level: str
    master_gain_db: float
    codec: str = "none"
    limiter: bool = False
    seed: int = 0
    meta: dict = field(default_factory=dict)


class BackgroundPool:
    """背景池: MUSAN 噪声 / IR 卷积 / 合成底噪, 预生成后以 memmap 常驻。

    磁盘: pool.f32 (所有条目首尾相接) + pool.json (每条的 offset/length/rms/kind)。
    读取只做一次切片, 不占常驻内存。
    """

    def __init__(self, root: str, *, sr: int = SR):
        self.root = os.path.abspath(root)
        self.sr = sr
        with open(os.path.join(self.root, "pool.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.entries = self.meta["entries"]
        self._mm = None
        self._cache: dict[int, np.ndarray] = {}
        self._cache_bytes = 0
        self._cache_cap = 48 << 20          # 最多缓存 48MB 环形条目

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def mm(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(os.path.join(self.root, "pool.f32"),
                                 dtype="<f4", mode="r")
        return self._mm

    def _entry(self, idx: int) -> np.ndarray:
        """取池内某条目的【已环形化】数组 (缓存, 避免每次重读 20s 数据)。"""
        c = self._cache.get(idx)
        if c is not None:
            return c
        e = self.entries[idx]
        o, L = int(e["offset"]), int(e["length"])
        seg = np.asarray(self.mm[o:o + L], dtype=np.float32)
        if self._cache_bytes + seg.nbytes <= self._cache_cap:
            self._cache[idx] = seg
            self._cache_bytes += seg.nbytes
        return seg

    def take(self, rng: np.random.Generator, n: int, *,
             kinds: tuple[str, ...] | None = None) -> np.ndarray:
        """随机取 n 个样本的背景 (环形拼接, 保证无缝)。"""
        cand = [i for i, e in enumerate(self.entries)
                if kinds is None or e["kind"] in kinds]
        if not cand:
            cand = list(range(len(self.entries)))
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        guard = 0
        while pos < n and guard < 64:
            guard += 1
            idx = int(rng.choice(cand))
            seg = self._entry(idx)
            if seg.size == 0:
                continue
            start = int(rng.integers(0, max(1, seg.size - 1)))
            m = min(seg.size - start, n - pos)
            out[pos:pos + m] = seg[start:start + m]
            pos += m
            if pos < n:                       # 接上条目开头, 形成无缝环
                m2 = min(start, n - pos)
                if m2 > 0:
                    out[pos:pos + m2] = seg[:m2]
                    pos += m2
        return out

    def take_loudness_matched(self, rng: np.random.Generator, n: int,
                              target_db: float) -> np.ndarray:
        x = self.take(rng, n)
        cur = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
        want = 10 ** (target_db / 20.0)
        return (x * (want / cur)).astype(np.float32)


@dataclass
class MixerConfig:
    window: float = 10.0
    sr: int = SR
    n_bands: int = 16
    short_atom_dur: float = 0.6      # 短于此的原子才可能自重复
    max_instances: int = 64          # 硬上限 (自重复 + 近邻后)
    atom_target_rms_db: float = -34.0  # 每个原子归一化到的电平 (混音器推子基准)
    safety_ceiling: float = 0.98     # 最终硬上限, 防爆音 (真实录像必然也受过限幅)
    codec_pool: tuple[str, ...] = ("mp3_128", "mp3_96", "aac_128", "opus_64", "none")
    # 把源的起始时刻铺满整个窗口，而不是挤在 ±delay_span 内。
    # 默认 True —— 旧行为（False）实测会让 10s 窗口里 95% 是空的，见 render() 里的注释。
    # 保留开关是为了让"复现旧数据分布"的对照实验仍然可做。
    spread_delays: bool = True
    # 只保留【可逆】变换：WSOLA 变速不变调（相位被重排）与混响（需要 RIR）都无法从
    # 波形反算、也无法在渲染端抵消，把它们当参数监督目标只会给参数头灌噪声，
    # 而且它们最贵（实测 wsola ~30 ms/次、reverb ~15 ms/次，每样本 8 个实例）。
    # True 时：mode 折成 resample、wet=0、rir=None；EQ/gain/drive/tilt/pitch 全部保留。
    trim_invertible: bool = True


def _rms_normalize(x: np.ndarray, target_db: float) -> np.ndarray:
    """按 RMS 归一到目标电平 (转调 audio.rms_normalize, 保证全链路一致)。"""
    from ..audio import rms_normalize_db
    return rms_normalize_db(x, target_db)


class Mixer:
    """合成器主体。一个实例可复用于多个 split (各自传不同的候选 id)。"""

    def __init__(self, lib: AtomLib, man: Manifest, bg: BackgroundPool | None = None,
                 ir: IRPool | None = None, cfg: MixerConfig | None = None,
                 levels: dict[str, LevelSpec] | None = None):
        self.lib = lib
        self.man = man
        self.bg = bg
        self.ir = ir
        self.cfg = cfg or MixerConfig()
        self.levels = levels or build_levels(self.cfg.window, self.cfg.n_bands)
        self.centers = mel_band_gains(self.cfg.n_bands, self.cfg.sr)
        # 每个 split 的候选 id (训练时只从 train 抽, 保证 test 原子从未出现)
        self.split_ids: dict[str, np.ndarray] = {}
        for s in ("train", "val", "test"):
            ids = np.array(man.ids(split=s), dtype=np.int64)
            self.split_ids[s] = ids
        # 同组索引: group -> ids (只在同 split 内, 避免跨 split 近邻泄漏)
        self.group_ids: dict[str, dict[str, list[int]]] = {}
        for s in ("train", "val", "test"):
            d: dict[str, list[int]] = {}
            for i in self.split_ids[s]:
                d.setdefault(man.atoms[int(i)].group, []).append(int(i))
            self.group_ids[s] = d
        # 长原子 (够长才有意义的拉伸) 与短原子
        self._dur48 = np.array([self.lib.dur(int(i)) for i in range(len(lib))], dtype=np.float32)

    # ------------------------------------------------------------------ 选材
    def pick_atoms(self, rng: np.random.Generator, spec: LevelSpec, split: str,
                   n: int) -> list[tuple[int, bool]]:
        """挑 n 条原子, 返回 [(atom_id, 是否近邻困难负样本)]。"""
        pool = self.split_ids[split]
        if pool.size == 0:
            raise ValueError(f"split {split} 没有可用原子")
        out: list[tuple[int, bool]] = []
        for _ in range(n):
            if rng.random() < spec.hard_neighbor and out:
                # 从前一条取同组兄弟
                base = out[int(rng.integers(len(out)))][0]
                g = self.man.atoms[base].group
                sib = [i for i in self.group_ids[split].get(g, []) if i != base]
                if sib:
                    out.append((sib[int(rng.integers(len(sib)))], True))
                    continue
            out.append((int(pool[int(rng.integers(pool.size))]), False))
        return out

    # ------------------------------------------------------------------ 渲染
    def render(self, rng: np.random.Generator, level: str, *, split: str = "train",
               atom_ids: list[int] | None = None,
               t_params: list[TParams] | None = None) -> MixResult:
        spec = self.levels[level]
        T = int(self.cfg.window * self.cfg.sr)
        sr = self.cfg.sr

        # 1) 选材 (自重复 + 近邻)
        if atom_ids is None:
            n_base = int(rng.integers(spec.n_atoms[0], spec.n_atoms[1] + 1))
            picks = self.pick_atoms(rng, spec, split, n_base)
        else:
            picks = [(int(a), False) for a in atom_ids]

        expanded: list[tuple[int, bool, int, int]] = []   # (id, nb, rep_idx, rep_total)
        for aid, nb in picks:
            d = float(self._dur48[aid])
            reps = 1
            if (atom_ids is None and d <= self.cfg.short_atom_dur
                    and rng.random() < spec.self_repeat):
                reps = int(rng.integers(spec.repeat_range[0], spec.repeat_range[1] + 1))
            if len(expanded) + reps > self.cfg.max_instances:
                reps = max(1, self.cfg.max_instances - len(expanded))
            for k in range(reps):
                expanded.append((aid, nb, k, reps))

        # 2) 逐实例参数 (支持外部注入 t_params 做"受控渲染")
        insts: list[Instance] = []
        dry = np.zeros(T, dtype=np.float32)
        periodic = None
        for si, (aid, nb, k, reps) in enumerate(expanded):
            if t_params is not None and si < len(t_params):
                p = t_params[si]
                noise_db = -120.0
            else:
                p, noise_db = sample_params(rng, spec.dr, n_bands=self.cfg.n_bands,
                                            centers=self.centers, sr=sr)
            if self.cfg.trim_invertible:
                # 只留可逆变换；参数真值与波形都基于这份裁剪后的 p，保证同源。
                p.mode = "resample"
                p.piece = ()
                p.wet = 0.0
                p.rir = None
            # 自重复布局: 三种典型形态
            #   a) 等间隔 (脚步/连续普攻)  b) 随机抖动  c) 完全同起点 (同帧多单位同技能)
            if k == 0:
                if self.cfg.spread_delays:
                    # 【修 bug】原来只有 `sample_span(rng, spec.dr.delay_span, bias_center=True)`，
                    # 而 build_levels() 给【所有】难度档的 delay_span 都是 0.25 —— 于是 10 秒窗口里
                    # 所有实例都堆在 ±0.25 秒内，其余 95% 是空的（实测 delay ∈ [-0.12, 0.09]，
                    # 平均 span 仅 0.21 s）。在这个数据上训练等于教模型"事件永远出现在第 0 帧附近"。
                    #
                    # 现在：源的起始时刻在【整个窗口】内均匀采样；同一个源的重复实例按
                    # step 间隔往后排（连续平A / 脚步的真实形态），超出窗口的重复直接丢弃。
                    dur = float(self._dur48[aid])
                    lo = float(spec.dr.delay_span)
                    hi = max(lo, self.cfg.window - dur - lo)
                    d0 = float(rng.uniform(lo, hi)) if hi > lo else lo
                    # 优先让整个重复串都落在窗口内（源模型的核心：能听到一串重复）
                    if reps > 1:
                        step_pre = dur * 1.25
                        avail = self.cfg.window - dur - step_pre * (reps - 1)
                        if avail > lo:
                            d0 = float(rng.uniform(lo, avail))
                else:
                    d0 = sample_span(rng, spec.dr.delay_span, bias_center=True)
                if reps > 1:
                    exact_dup = rng.random() < 0.15
                    periodic = (not exact_dup) and (rng.random() < 0.6)
                    step = float(self._dur48[aid]) * float(rng.uniform(0.9, 1.6))
                    jitter = 0.0 if (periodic or exact_dup) else \
                        float(self._dur48[aid]) * 0.25
                else:
                    exact_dup, step, jitter = False, 0.0, 0.0
            if k == 0:
                delay = d0
            elif exact_dup:
                delay = d0
            else:
                delay = d0 + step * k + (float(rng.normal(0.0, jitter)) if jitter > 0 else 0.0)
            if self.cfg.spread_delays and delay >= self.cfg.window - 1e-3:
                continue        # 越出窗口的重复直接丢掉（不裁出半个事件）
            # 重复实例: 轻微重参数化 (同一音效重复播放不会完全一致)
            if k > 0:
                p = TParams(**{**p.__dict__})
                p.gain_db += 0.0 if exact_dup else float(rng.normal(0.0, 0.8))
            p.delay = float(delay)
            if p.rir is None and p.wet > 0 and self.ir is not None and not self.cfg.trim_invertible:
                h, spec_ir = self.ir.sample(p.rt60)
                p.rir = h
            noise_seed = int(rng.integers(1 << 31))

            x, src_sr = self.lib.raw(aid)
            # 【关键】先把原子自身电平归一到统一 RMS, 再用 p.gain_db 施加"混音器推子"。
            # 否则各原子原始电平差异极大(有的 -30dBFS 有的 -6dBFS), 叠加 20 条必然爆音。
            # 归一化只用一个标量, 不破坏"线性变换 + 可学习增益"的可辨识性。
            x = _rms_normalize(x, self.cfg.atom_target_rms_db)
            x = resample(x, src_sr, sr) if src_sr != sr else x
            y = apply_distortions(x, p, sr, rng=rng, centers=self.centers,
                                  noise_db=noise_db, noise_seed=noise_seed)
            s0 = int(round(p.delay * sr))
            a, b = max(0, s0), min(T, s0 + y.size)
            if b > a:
                dry[a:b] += y[a - s0:b - s0]
            insts.append(Instance(atom_id=aid, delay=p.delay, p=p, gain_db=p.gain_db,
                                  dur=y.size / sr, rep_index=k, rep_total=reps,
                                  is_hard_neighbor=nb, group=self.man.atoms[aid].group,
                                  noise_db=noise_db, noise_seed=noise_seed,
                                  exact_dup=(k > 0 and exact_dup)))

        # 3) 背景
        bg = np.zeros(T, dtype=np.float32)
        if self.bg is not None and len(self.bg) and rng.random() < spec.bg_prob:
            target = float(rng.uniform(*spec.bg_db))
            bg = self.bg.take_loudness_matched(rng, T, target)

        # 4) 干扰 (用同 split 的原子做大变形 -> "另一段音频被混进来")
        itf = np.zeros(T, dtype=np.float32)
        k_itf = int(rng.integers(spec.interf[0], spec.interf[1] + 1))
        if k_itf > 0:
            itf = self._render_interference(rng, split, T, k_itf,
                                            float(rng.uniform(*spec.interf_db)))

        # 5) 主增益 -> 常数归一化 (保真) -> 可选软限幅 (真非线性, 概率低)
        mix = dry + bg + itf
        g = float(rng.uniform(*spec.master_gain_db))
        mix = mix * (10 ** (g / 20.0))
        # 【关键】防削波必须用【常数缩放】而不是压缩: 常数缩放对线性混合是精确
        # 可抵消的 (真值反相乘同一常数即可), 而限幅/压缩是非线性, 会让完美反相
        # 也消不掉 (实测把真值反相 PSR 从 +30dB 拖到 -14dB)。
        pk = float(np.max(np.abs(mix))) if mix.size else 0.0
        if pk > self.cfg.safety_ceiling:
            mix = (mix * (self.cfg.safety_ceiling / pk)).astype(np.float32)
            g += 20.0 * np.log10(self.cfg.safety_ceiling / pk)
        codec = "none"
        if rng.random() < spec.codec_prob:
            codec = str(rng.choice(self.cfg.codec_pool[1:]))
            mix = self._codec_sim(mix, codec, sr)
        limiter = bool(rng.random() < spec.limiter_prob)
        if limiter:
            mix = self._limiter(mix)

        return MixResult(mix=np.ascontiguousarray(mix, dtype=np.float32),
                         dry=dry, background=bg, interference=itf, instances=insts,
                         level=level, master_gain_db=g, codec=codec, limiter=limiter,
                         seed=int(rng.integers(1 << 31)),
                         meta={"n_atoms_base": len(picks), "n_instances": len(insts)})

    # ------------------------------------------------------------- 干扰与转码
    def _render_interference(self, rng: np.random.Generator, split: str, T: int,
                             k: int, target_db: float) -> np.ndarray:
        """干扰 = 同 split 的原子经过极端变形 (长拉伸/重混响/强 EQ) 后叠加。

        用原子而非真实音乐的原因: MUSAN 只有 noise 子集(无 music/speech), 且必须
        保证 test split 的素材绝不参与。极端变形让"它自己不可能被识别成模板"。
        """
        pool = self.split_ids[split]
        if pool.size == 0:
            return np.zeros(T, dtype=np.float32)
        out = np.zeros(T, dtype=np.float32)
        for _ in range(k):
            aid = int(pool[int(rng.integers(pool.size))])
            x, src_sr = self.lib.raw(aid)
            x = resample(x, src_sr, self.cfg.sr) if src_sr != self.cfg.sr else x
            p = TParams(r=float(rng.uniform(0.5, 0.75)),      # 大幅拉伸 -> 不再是原音效
                        mode="wsola" if rng.random() < 0.5 else "resample",
                        tilt=float(rng.uniform(-0.05, 0.05)),
                        gain_db=float(rng.uniform(-18.0, -3.0)),
                        eq_db=rng.uniform(-8.0, 2.0, size=self.cfg.n_bands).astype(np.float32),
                        n_bands=self.cfg.n_bands,
                        drive=float(rng.uniform(1.0, 2.0)),
                        clip=float(rng.uniform(0.7, 1.0)))
            if self.ir is not None and rng.random() < 0.5:
                p.rir, _ = self.ir.sample(float(rng.uniform(0.3, 1.2)))
                p.wet = float(rng.uniform(0.2, 0.6))
            y = apply_distortions(x, p, self.cfg.sr, rng=rng, centers=self.centers)
            s0 = int(rng.integers(0, max(1, T)))
            y = np.roll(y, s0)
            m = min(T, y.size)
            if m > 0:
                out[:m] += y[:m]
            if y.size > T:                      # 环绕到开头
                k2 = min(T, y.size - T)
                out[:k2] += y[T:T + k2]
        cur = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)) + 1e-12)
        want = 10 ** (target_db / 20.0)
        return (out * (want / cur)).astype(np.float32)

    def _codec_sim(self, x: np.ndarray, kind: str, sr: int) -> np.ndarray:
        """编解码损伤模拟。优先用真实编码器(sox/fast_mp3_augment), 否则用带限+量化近似。"""
        try:
            if kind.startswith("mp3"):
                import fast_mp3_augment  # type: ignore
                br = 128 if "128" in kind else 96
                import numpy as _np
                y = fast_mp3_augment.compress(x.astype(_np.float32), sr,
                                              bitrate=br, use_pydub=False)
                y = fast_mp3_augment.decompress(y)
                if isinstance(y, tuple):
                    y = y[0]
                y = np.asarray(y, dtype=np.float32).reshape(-1)
                if y.size == x.size:
                    return y
        except Exception:
            pass
        # 近似: 低通 + 轻微量化 + 高通 (丢掉编解码最典型的两个边)
        from .distort import hi_rolloff
        y = hi_rolloff(x, -6.0 if "64" in kind or "96" in kind else -3.0, sr,
                       knee_hz=11000.0 if "128" in kind or "96" in kind else 13000.0)
        lev = 2.0 ** 14
        return (np.round(y * lev) / lev).astype(np.float32)

    @staticmethod
    def _limiter(x: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
        """软膝限幅 (混音总线限幅器)。

        不能用硬 clip: 削波是强非线性, 会让【完美反相也无法抵消】。实测硬削波
        把真值反相的 PSR 从应有的 +30dB 拉到 -32dB (反相变成增强)。这里用包络
        跟踪的平滑增益压缩, 保真度高、可预测, 与真实混音总线行为一致。
        """
        from scipy.signal import lfilter
        a = np.abs(np.asarray(x, dtype=np.float32))
        if a.size == 0 or float(a.max()) <= ceiling:
            return x
        atk, rel = 0.001, 0.05
        env = lfilter([atk], [1.0, -(1.0 - atk)], a).astype(np.float32)
        peak_env = np.maximum(env, a)
        g = np.ones_like(peak_env)
        over = peak_env > ceiling
        g[over] = ceiling / peak_env[over]
        g = lfilter([rel], [1.0, -(1.0 - rel)], g).astype(np.float32)
        y = x * g
        pk = float(np.max(np.abs(y))) if y.size else 0.0
        if pk > 1.0:                      # 极少数残留: 温和饱和, 永不硬削
            y = np.tanh(y * (1.0 / pk)) * pk
        return y.astype(np.float32)
