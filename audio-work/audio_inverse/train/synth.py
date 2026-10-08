"""On-the-fly fully-synthetic mixture generator (no real-recording background by default).

声音按【源】组织: 一个源 = 同一个音效按随机间隔重复 (模拟一个干员在平A). 每个源有自己的
活跃区间 [ta, tb], 所以一个音效可以:

  中途插入   ta 在窗口内   -> onset 在窗口里, 正常给 onset 脉冲
  中途退出   tb 在窗口内   -> 尾巴被切掉, onset 仍在窗口里, 正常给脉冲
  已经在响   ta < 0        -> onset 在窗口【外】, 只标注 span, 【不给 onset 脉冲】
  响过末尾   tb > T        -> 尾巴被窗口右边界切掉

【已经在响】这一条很重要: 旧代码把 o<0 的事件整条丢掉, 于是模型从来没见过【声音已经响到一半】
的输入 —— 而 loc_eval 的假峰分类显示, 检测头 99% 的假峰恰好落在正在响的事件内部.
ev[..., 0] < 0 就是【onset 在窗口外】的标记 (padding 是 ev=-1 且 lb=-1, 由 lb 区分).

事件数由 max_ev 封顶(数组容量, 也是硬上限); 源数随 T 按 src_rate_ref 缩放, 保持每秒源数不变.
min_ev 目前【没有被使用】(保留参数只是不想动 eval.py 的调用); 空窗由 empty_frac 控制.

bg_mode:
  "noise"     -> white/pink noise floor at randomised level (default)
  "silence"   -> digital silence
  "mixed"     -> 80% noise / 20% silence
  "recording" -> crop of the real recording (kept for the later BGM stage; not used now)
  "long"      -> 【真实长音背景】从 long_pool.npy 取 1~2 条 BGM/环境/剧情音床, 再以
                 概率 long_voice_p 叠 1~3 条干员战斗语音. 这一档才是"录音里有 BGМ+语音"
                 的诚实测试: 所有长音都来自游戏解包, 不是噪声.
                 注意: long 档的 bg_lo/bg_hi 是【RMS dBFS】, 噪声档是【峰值 dBFS】,
                 所以同一个 -45 在 long 档下响得多(音乐本来就比噪声底密) —— 别直接横比.
"""
import os, json, math
import numpy as np
import torch
from torch.utils.data import Dataset

ML = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ML), "data", "atoms")
SR = 44100
HOP = 441                 # 谱帧 hop，必须与 core.HOP 一致（标签帧栅格）；改采样率时同步


def _recording_pool():
    """真实录音背景（--bg-mode recording）。

    文件名带采样率（bg44.npy / bg16.npy）：采样率不匹配时把 16 kHz 的缓冲当 44.1 kHz 用，
    音高与时序都会错。缓存不存在时从真实录音重新生成，优先用高采样率的录音源。
    """
    p = os.path.join(ML, "bg%d.npy" % (SR // 1000))
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    import sys
    sys.path.insert(0, D)
    import alab
    for name in ("nl_mono.wav", os.path.join("real", "ui_demo_mono48k.wav"),
                 os.path.join("samples", "nl_mono.wav"), "v82_mono.wav"):
        src = os.path.join(D, name)
        if not os.path.exists(src):
            continue
        x, sr = alab.wav_read(src, mono=True)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = alab.bandpass(x, sr, 30.0, min(sr, SR) * 0.475)
        y = np.interp(np.arange(int(len(x) * SR / sr)) / SR,
                      np.arange(len(x)) / sr, x).astype(np.float32)
        np.save(p, y)
        print("[synth] recording 背景 <- %s (%.0f s @%d Hz -> %d Hz)"
              % (name, len(y) / SR, sr, SR), flush=True)
        return np.load(p, mmap_mode="r")
    raise FileNotFoundError("找不到可用作 recording 背景的录音（nl_mono.wav 等）")


def make_pulse(shape, length=5, sigma=2.5):
    """事件边界的目标脉冲. rect=方波(平台没有梯度, 峰在哪都行 -> 定位抖); tri/gauss=尖峰.

    onset 和 offset 共用同一个形状: train.py 里 offset 的目标要按同样形状反向贴一次,
    否则两个头的目标锐度不同, 学出来的边界不一致.
    """
    L = max(int(length), 1)
    k = np.arange(0, L + 1, dtype=np.float32)
    if shape == "rect":
        return np.ones(L, dtype=np.float32)
    if shape == "tri":
        return np.maximum(0.0, 1.0 - k / L).astype(np.float32)
    if shape == "gauss":
        s = max(float(sigma), 0.25)
        return np.exp(-0.5 * (k / s) ** 2).astype(np.float32)
    raise ValueError("onset_shape 只能是 rect / tri / gauss")


def pink_noise(n, rng):
    nf = 1 << int(np.ceil(np.log2(max(n, 64))))
    w = rng.normal(0.0, 1.0, nf)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(nf, 1.0 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    W = W / np.sqrt(f)
    y = np.fft.irfft(W, nf)[:n]
    y = y / (np.abs(y).max() + 1e-9)
    return y.astype(np.float32)


def chan_fx(x, rng):
    n = len(x)
    nf = 1 << int(np.ceil(np.log2(max(n, 64))))
    X = np.fft.rfft(x, nf)
    f = np.maximum(np.fft.rfftfreq(nf, 1.0 / SR), 1.0)
    g = np.ones_like(f)
    hp = 10 ** rng.uniform(1.5, 2.6)
    lp = 10 ** rng.uniform(3.45, 3.90)
    g *= 1.0 / np.sqrt(1.0 + (hp / f) ** 2)
    g *= 1.0 / np.sqrt(1.0 + (f / lp) ** 2)
    g *= (f / 1000.0) ** rng.uniform(-0.45, 0.45)
    for _ in range(int(rng.integers(0, 3))):
        fc = 10 ** rng.uniform(2.2, 3.8)
        amp = 10 ** (rng.uniform(-6, 6) / 20.0)
        g *= 1.0 + (amp - 1.0) * np.exp(-0.5 * ((np.log(f) - math.log(fc)) / 0.5) ** 2)
    y = np.fft.irfft(X * g, nf)[:n].astype(np.float32)
    pk = float(np.abs(y).max())
    if pk > 1e-6:
        y = y / pk
    d = rng.uniform(0.6, 6.0)
    y = (np.tanh(d * y) / np.tanh(d)).astype(np.float32)
    if rng.random() < 0.25:
        y = (np.sign(y) * np.abs(y) ** rng.uniform(0.5, 1.0)).astype(np.float32)
    if rng.random() < 0.35:
        r = rng.uniform(30.0, 300.0)
        y = y + rng.normal(0, math.sqrt(float(np.mean(y ** 2))) / r, size=len(y)).astype(np.float32)
    if rng.random() < 0.30:
        rate = 1.0 + rng.uniform(-0.006, 0.006)
        y = np.interp(np.arange(int(len(y) / rate)) * rate, np.arange(len(y)), y).astype(np.float32)
    return y


class SynthDS(Dataset):
    def _mk_pulse(self):
        return make_pulse(self.onset_shape, self.onset_len, self.onset_sigma)

    def __init__(self, T=10.0, length=200000, seed=0, max_ev=128, lo=-32.0, hi=0.0,
                 bg_mode="mixed", bg_lo=-70.0, bg_hi=-40.0, nmel_frames=None,
                 label_mode="cluster", nclust=512, lmax_fx=88200, use_variants=True,
                 no_fx=False, min_ev=1, dense_frac=0.3, onset_len=5, silence_frac=0.5,
                 empty_frac=0.0, min_src=1, max_src=4, ivl_med=1.0, ivl_sig=0.5, once_len=1.0,
                 onset_shape="rect", onset_sigma=2.5, trace=False,
                 left_pad=0.0, p_mid=0.3, src_span_lo=0.25, src_rate_ref=7.5, min_ev_samp=2205,
                 long_voice_p=0.65):
        self.T = T; self.n = int(T * SR); self.length = length; self.seed = seed
        self.max_ev = max_ev; self.lo = lo; self.hi = hi
        self.bg_mode = bg_mode; self.bg_lo = bg_lo; self.bg_hi = bg_hi
        self.lmax_fx = int(lmax_fx)
        self.clusters = None
        if label_mode == "cluster":
            cp = os.path.join(ML, "clusters_%d.npy" % nclust)
            assert os.path.exists(cp), "run ml/make_clusters.py first"
            self.clusters = np.load(cp)
        self._pink = {}
        self.no_fx = bool(no_fx); self.min_ev = int(min_ev)
        self.dense_frac = float(dense_frac); self.onset_len = int(onset_len)
        self.onset_shape = str(onset_shape); self.onset_sigma = float(onset_sigma)
        self._pulse = self._mk_pulse()
        self.silence_frac = float(silence_frac); self.empty_frac = float(empty_frac)
        self.min_src = int(min_src); self.max_src = int(max_src)
        self.ivl_med = float(ivl_med); self.ivl_sig = float(ivl_sig)
        # 时长 >= once_len(秒) 的素材【每个源只播一次】, 不再按 ivl 间隔重复.
        # 依据: 重复间隔的中位是 1.0s, 而模板库的时长中位已经从 SFX 时代的 0.94s 变成
        # 含战斗语音后的 2.09s (p90 6.17s) -> 照着 1.0s 重复会让每个源自己叠自己 2~6 层,
        # 一个 240ms 池化窗里 71% 混着别的类别, 识别头拿到的输入里"类别"不是输入的函数.
        # 长语音/长音效本来也不是每秒重复一次的东西; 短的攻击音效照旧重复(那才是平A).
        self.once_len = float(once_len)
        self.trace = bool(trace); self.trace_rec = None
        # 源可以【中途插入 / 中途退出】; 事件可以跨窗口边界被切掉. 见 __getitem__ 里的说明.
        # left_pad<=0 -> 取整个窗口长度：素材可以从窗口前开始播，窗口里听到的是中段
        self.left_pad = float(left_pad) if float(left_pad) > 0 else float(T)
        self.p_mid = float(p_mid)                # 一个源"t=0 时已经在响"的概率 (onset 落在窗口外)
        self.src_span_lo = float(src_span_lo)    # 源存活时长至少占剩余窗口的比例
        # 源数 = 抽到的 1~max_src 乘 max(1, T/该值). 7.5 是 edge_probe.py 实测校准出来的 (T=10 -> ×1.33):
        #   ref=7.5 -> 1.93 事件/秒   ref=10 -> 1.42   ref=4 -> 3.40   (T=4 时不管 ref 都是 1.87)
        # 想要更密的训练数据就调小 ref.
        # 之所以对齐【事件密度】而不是【源数/秒】: 源现在只占窗口的一部分, 按源数等比放大会把
        # 密度顶高 1.5 倍, 那样 T 和密度两个变量就缠在一起了, 没法只看窗口长度的影响.
        self.src_rate_ref = float(src_rate_ref)
        self.min_ev_samp = int(min_ev_samp)      # 窗口内可听见长度短于这个采样数就不算一个事件
        if self.no_fx:
            use_variants = False
        self.nmel = nmel_frames or (1 + self.n // HOP)
        self.lens = np.load(os.path.join(ML, "bank_lens.npy"))
        # NOTE: do NOT hold the memmaps here.  On Windows the DataLoader pickles the Dataset to
        # every spawned worker, and pickling a 526 MB memmap materialises it -> 3 banks x N workers
        # is several GB of pointless copies (and MemoryError under a constrained parent).
        # Keep the paths only, and open the memmaps lazily inside the worker.
        # 模板库是【变长流】：bank_pool.npy(fp16 拼接) + bank_offs/lens.npy
        self.bank_pool_path = os.path.join(ML, "bank_pool.npy")
        assert os.path.exists(self.bank_pool_path), \
            "no template bank found - run build_bank.py first"
        self.offs = np.load(os.path.join(ML, "bank_offs.npy"))
        self._pool = None
        self._rec = None
        self.long_voice_p = float(long_voice_p)
        # 长音池(BGM / 环境 / 干员战斗语音). 同样只存路径 + 懒加载: 把 memmap 挂在 self 上
        # 会让 DataLoader 每 spawn 一个 worker 就整块 pickle 一遍(见上面 bank 的注释).
        self._lp = None; self._lp_path = None
        self._llen = None; self._loff = None; self._lmeta = None
        self._long_bed = None; self._long_vox = None
        vp = os.path.join(ML, "variants.npy")
        self.variants_path = vp if (use_variants and os.path.exists(vp)) else None
        self.variants = None
        if self.variants_path:
            self.vlens = np.load(os.path.join(ML, "variants_lens.npy"))
            self.V = int(self.vlens.shape[0] // len(self.lens))
        else:
            self.vlens = None; self.V = 0

    def __len__(self):
        return self.length

    def _content_end(self, chunk=1 << 22):
        """long_pool.npy 里【真实内容的结束位置】.

        build_long.py 用 open_memmap 先按完整长度把文件建好、再逐条填. 中途被打断的话,
        头里声明的长度是完整的, 但后半段永远读成 0 —— np.load 一声不响.
        实测这份池子: 声明 932.6M 采样, 真实内容只到 ~798M (尾部 14% 全零),
        而 long_offs 是按 1684 分钟算的 -> 2709 条长音床里只剩个位数还能取到东西.
        从尾部按 chunk 倒着扫找最后一个非零块, 再二分定位精确边界.
        """
        N = int(self._lp.shape[0])
        j = N
        while j > 0 and not np.any(self._lp[max(0, j - chunk):j]):
            j = max(0, j - chunk)
        if j == 0:
            return 0
        lo, hi = max(0, j - chunk), j
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if np.any(self._lp[mid:hi]):
                lo = mid
            else:
                hi = mid
        return hi

    def _long_bank(self):
        """Lazily open ml/long_pool.npy; returns None if the pool was never built."""
        if self._lp_path is None:
            p = os.path.join(ML, "long_pool.npy")
            self._lp_path = p if os.path.exists(p) else ""
            if not self._lp_path:
                self._long_bed = np.zeros(0, dtype=np.int64)
                self._long_vox = np.zeros(0, dtype=np.int64)
        if not self._lp_path:
            return None
        if self._lp is None:
            self._lp = np.load(self._lp_path, mmap_mode="r")
            self._llen = np.load(os.path.join(ML, "long_lens.npy"))
            self._loff = np.load(os.path.join(ML, "long_offs.npy"))
            self._lmeta = json.load(open(os.path.join(ML, "long_meta.json"), encoding="utf-8"))
            BED = ("music", "ambience", "dialog", "long_se")
            # 池子可能是【残缺】的: long_offs/long_lens 记的是完整索引的偏移, 而 long_pool.npy
            # 只写进去了前面一部分. 必须【在抽样前】把死条目剔掉, 否则绝大多数抽样都落在死条目上,
            # 长音床静默变成"一片数字静音" —— 训练分布和真实录音对不上, 而且一点错都不报.
            N = int(self._lp.shape[0])
            _ce = self._content_end()
            _L = np.asarray(self._llen, dtype=np.int64)
            _O = np.asarray(self._loff, dtype=np.int64)
            _usable = (_L > 0) & (_O >= 0) & (_O < _ce)
            self._long_bed = np.array([i for i, m in enumerate(self._lmeta)
                                       if m.get("kind") in BED and _usable[i]], dtype=np.int64)
            self._long_vox = np.array([i for i, m in enumerate(self._lmeta)
                                       if m.get("kind") == "voice_battle" and _usable[i]], dtype=np.int64)
            _nbed = sum(1 for m in self._lmeta if m.get("kind") in BED)
            if len(self._long_bed) < _nbed or _ce < N:
                print("[synth] long_pool 残缺: 声明 %.0f 分钟, 真实内容只到 %.0f 分钟; "
                      "长音床可用 %d/%d 条 (BGM/环境), 干员语音可用 %d 条. "
                      "要补全 BGM 得重跑 x_extract_long.py all + build_long.py"
                      % (N / SR / 60, _ce / SR / 60, len(self._long_bed), _nbed,
                         len(self._long_vox)), flush=True)
        return self._lp

    def _lseg(self, i):
        """条目 i 在池子里【实际可用】的那一段; 越界或全零返回 None.

        long_offs/long_lens 记的是完整索引的偏移, 而 long_pool.npy 可能只写进去一部分 ——
        实测 16797 条里 2517 条的起点已越过池尾, 越过内容边界的那些整段读出来全是 0.
        以前 _lcrop 会切出空数组, 紧接着 rng.integers(0, 1-n) 抛
        `ValueError: high <= 0`, 把整个 --bg-mode long 直接打崩.
        """
        L = int(self._llen[i])
        off = int(self._loff[i])
        N = int(self._lp.shape[0])
        if L <= 0 or off < 0 or off >= N:
            return None
        x = np.asarray(self._lp[off:off + min(L, N - off)], dtype=np.float32)
        if x.size == 0 or not x.any():      # 全零 = 落在没写进去的那一段, 当成无效
            return None
        return x

    def _lcrop(self, rng, i, n):
        """one n-sample crop of item i; short items are tiled so the bed never gaps."""
        x = self._lseg(i)
        if x is None:
            return None
        if x.size < n:
            x = np.tile(x, int(np.ceil(n / x.size)))
        if x.size < n:
            return None
        o = int(rng.integers(0, x.size - n + 1))
        return x[o:o + n]

    @staticmethod
    def _fit(x, db):
        """RMS-normalise to db dBFS."""
        r = math.sqrt(float(np.mean(x ** 2)))
        if r <= 1e-9:
            return x
        return x * (10 ** (db / 20.0) / r)

    def _background(self, rng, n):
        mode = self.bg_mode
        if mode == "mixed":
            mode = "silence" if rng.random() < self.silence_frac else "noise"
        if mode == "silence":
            return np.zeros(n, dtype=np.float32)
        if mode == "recording":
            if self._rec is None:
                self._rec = _recording_pool()
            if len(self._rec) > n + 1:
                o = int(rng.integers(0, len(self._rec) - n))
                bg = np.asarray(self._rec[o:o + n], dtype=np.float32)
                r = math.sqrt(float(np.mean(bg ** 2)))
                return bg * (10 ** (rng.uniform(self.bg_lo, self.bg_hi) / 20.0) / r) if r > 1e-7 else bg
            return np.zeros(n, dtype=np.float32)
        if mode == "long":
            if self._long_bank() is not None:
                db = float(rng.uniform(self.bg_lo, self.bg_hi))
                bg = np.zeros(n, dtype=np.float32)
                for _ in range(int(rng.integers(1, 3))):        # 1~2 条长音床 (BGM/环境)
                    if len(self._long_bed) == 0:
                        break
                    i = int(self._long_bed[rng.integers(0, len(self._long_bed))])
                    x = self._lcrop(rng, i, n)
                    if x is not None:
                        bg += self._fit(x, db)
                if not bg.any():
                    # 床一条都取不到 (池子残缺) -> 用噪声底当床兜住. 【绝不能】静默变成数字静音:
                    # "背景=绝对安静" 会让模型把任何一点声响都当成事件, 和录音差得最远.
                    bg = self._noise_floor(rng, n)
                if len(self._long_vox) and rng.random() < self.long_voice_p:
                    for _ in range(int(rng.integers(1, 4))):    # 1~3 条干员战斗语音
                        i = int(self._long_vox[rng.integers(0, len(self._long_vox))])
                        seg = self._lseg(i)
                        if seg is None:
                            continue
                        if seg.size > n:      # 语音比窗口长 -> 从语音里随机挑一段
                            s = int(rng.integers(0, seg.size - n + 1))
                            seg = seg[s:s + n]; o = 0
                        else:                 # 比窗口短 -> 放到窗口里的随机位置
                            o = int(rng.integers(0, n - seg.size + 1))
                        bg[o:o + seg.size] += self._fit(seg, db + float(rng.uniform(4.0, 14.0)))
                return bg.astype(np.float32)
            mode = "noise"                     # 池子没建 -> 退化成噪声, 不静默变成静音
        return self._noise_floor(rng, n)

    def _noise_floor(self, rng, n):
        """白噪/粉噪底, 峰值归一化后按 bg_lo..bg_hi (峰值 dBFS) 定电平."""
        if rng.random() < 0.5:
            bg = rng.normal(0.0, 1.0, n).astype(np.float32)
        else:
            if n not in self._pink:
                self._pink.clear()
                self._pink[n] = pink_noise(n, np.random.default_rng(12345))
            bg = self._pink[n].copy()
        pk = float(np.abs(bg).max())
        if pk > 1e-9:
            bg = bg / pk
        return (bg * (10 ** (rng.uniform(self.bg_lo, self.bg_hi) / 20.0))).astype(np.float32)

    def __getitem__(self, i):
        if self.variants_path is not None and self.variants is None:
            self.variants = np.load(self.variants_path, mmap_mode="r")
        if self._pool is None:
            self._pool = np.load(self.bank_pool_path, mmap_mode="r")
        rng = np.random.default_rng((self.seed * 1000003 + i) & 0x7FFFFFFF)
        n = self.n; K = len(self.lens)
        bg = self._background(rng, n)
        if rng.random() < 0.20:
            bg = (bg * (1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * np.arange(n) / SR))).astype(np.float32)

        mix = bg.copy()
        sig = np.zeros(n, dtype=np.float32) if self.trace else None
        rec = [] if self.trace else None
        spans = []; labels = []; tpls = []
        if rng.random() >= self.empty_frac:
            # "源"模型: 每个源 = 同一个音效按【随机间隔】重复 (模拟一个干员在平A)
            # 同一个源共用同一个变体 -> 音色一致, 这正是真实录音里的重复性线索
            # 源数按 T/src_rate_ref 缩放 -> 片段变长时【每秒的源数】不变, 否则 10s 反而比 4s 稀疏
            _sf = max(1.0, self.T / self.src_rate_ref)
            nsrc = int(rng.integers(self.min_src, self.max_src + 1))
            if rng.random() < self.dense_frac:
                nsrc = self.max_src
            nsrc = max(1, int(round(nsrc * _sf)))
            for _s in range(nsrc):
                k = int(rng.integers(0, K))
                wi = None
                if self.variants is not None:
                    vi = k * self.V + int(rng.integers(0, self.V))
                    L = int(self.vlens[vi])
                    if L < int(0.05 * SR):
                        continue
                    w = np.asarray(self.variants[vi, :L], dtype=np.float32)
                    wi = vi
                else:
                    L = int(self.lens[k])
                    if L < int(0.05 * SR):          # 短于 50 ms 的模板不足以定位
                        continue
                    off = int(self.offs[k])
                    w = np.asarray(self._pool[off:off + L], dtype=np.float32)
                    if L > self.lmax_fx:
                        w = w[:self.lmax_fx]; L = self.lmax_fx
                    if not self.no_fx:
                        if rng.random() < 0.30 and L > int(0.1 * SR):
                            L = int(rng.uniform(0.35, 0.95) * L); w = w[:L]
                        if rng.random() < 0.10:
                            w = np.interp(np.arange(int(L / 2.0)) * 2.0, np.arange(L), w).astype(np.float32); L = len(w)
                        w = chan_fx(w, rng)
                        # [本项目补的一行] chan_fx 末尾那段 ±0.6% 变速 (110-112 行) 会改变波形长度
                        # (实测 14908 -> 14871), 而这里原本没有同步 L -> La 仍按旧 L 算,
                        # seg = w[cut:cut+La] 比 La 短 -> mix[o:o+La] += seg*g 广播失败,
                        # 第一条样本就崩 (rate>1 的约 15% 样本都会崩).
                        # 上一行同样改过 w 的两个分支 (392/394) 都同步了 L, 这里漏了.
                        L = len(w)
                # 长素材只播一次(见 __init__ 里 once_len 的说明): 短攻击音效照旧按 ivl 重复
                once = self.once_len > 0 and (L / SR) >= self.once_len
                gain = 10 ** (rng.uniform(self.lo, self.hi) / 20.0)
                # ---- 源的活跃区间 [ta, tb]: 允许中途插入 / 中途退出 ----
                # ta 落在窗口左边之外 -> t=0 时这个音效【已经在响】, 它的 onset 不在窗口内(被切掉)
                # tb 落在窗口右边之外 -> 尾巴被窗口切掉, onset 仍在窗口内
                # 旧行为是 ta 只在 [-0.25,0.35] 抖一下、而且 o<0 的事件被整条丢弃 -> 永远见不到
                # "已经在响"和"中途停下"这两种真实情况.
                if self.p_mid > 0 and rng.random() < self.p_mid:
                    _cmax = int(min(self.left_pad * SR, max(0, L - self.min_ev_samp)))
                    ta = -(int(rng.integers(1, _cmax + 1)) / SR) if _cmax >= 1 else 0.0
                else:
                    ta = float(rng.uniform(0.0, self.T))
                tb = ta + float(rng.uniform(self.src_span_lo, 1.0)) * (self.T + L / SR - ta)
                t = ta
                while t < tb and len(spans) < self.max_ev:
                    o_raw = int(round(t * SR))                # 真值 onset(采样); 负 = 在窗口左边之外
                    ivl = float(np.clip(np.exp(rng.normal(math.log(self.ivl_med), self.ivl_sig)), 0.20, 4.0))
                    t += ivl
                    if o_raw >= n:
                        break
                    cut = -o_raw if o_raw < 0 else 0          # 左边界切掉的采样数
                    o = o_raw + cut                           # 混音里的实际起点(>=0)
                    La = min(L - cut, n - o)                  # 窗口内可听见的采样数
                    if La < self.min_ev_samp:
                        if once:
                            break          # 只播一次: 这一次没落进窗口, 就不再重试
                        continue
                    seg = w[cut:cut + La]
                    g = gain * 10 ** (rng.uniform(-1.5, 1.5) / 20.0)
                    mix[o:o + La] += seg * g
                    if self.trace:
                        sig[o:o + La] += seg * g
                        rec.append((o, La, k, float(g), seg.copy()))
                    # f0(真值 onset 帧, 可为负) 一路传下去: 负值 = onset 在窗口外 -> 只标 span, 不给脉冲
                    f0 = o_raw // HOP
                    f1 = min((o + La) // HOP + 1, self.nmel)
                    if f1 > max(f0, 0):
                        spans.append((f0, f1)); tpls.append(k)
                        labels.append(int(self.clusters[k]) if self.clusters is not None else k)
                    if once:
                        break              # 长素材: 播完这一次就换下一个源, 不叠自己
        scale = 10 ** (rng.uniform(-6, 4) / 20.0)
        mix = mix * scale
        clipped = False
        pk = float(np.abs(mix).max())
        if pk > 0.99:
            clipped = True
            mix = np.tanh(mix / pk * 1.3).astype(np.float32)
        if self.trace:
            self.trace_rec = dict(bg=bg * scale, sig=sig * scale, evs=rec,
                                  scale=scale, clipped=clipped, tpls=tpls)

        Tm = self.nmel
        ev = np.full((self.max_ev, 2), -1, dtype=np.int64)
        lb = np.full(self.max_ev, -1, dtype=np.int64)
        po = np.zeros(Tm, dtype=np.float32)
        for j, (f0, f1) in enumerate(spans[:self.max_ev]):
            a = min(f0, Tm - 1); b = min(max(f1, a + 1), Tm)
            ev[j] = (a, b); lb[j] = labels[j]
            if a < 0:
                continue          # onset 在窗口外(声音早就响了) -> 标 span, 但不给 onset 脉冲
            p = self._pulse
            e2 = min(Tm, a + len(p))
            po[a:e2] = np.maximum(po[a:e2], p[:e2 - a])   # ONSET pulse, not the whole span
        self.last_tpl = tpls[:self.max_ev]      # template ids of this sample (for diagnostics)
        return mix.astype(np.float32), po, ev, lb


def collate(batch):
    return (torch.from_numpy(np.stack([b[0] for b in batch])),
            torch.from_numpy(np.stack([b[1] for b in batch])),
            torch.from_numpy(np.stack([b[2] for b in batch])),
            torch.from_numpy(np.stack([b[3] for b in batch])))
