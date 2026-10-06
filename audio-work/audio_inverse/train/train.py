"""Train the coarse-class SFX detector.  argparse lives inside main() so spawned workers never see it."""
import os, sys, time, json, math, argparse, traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
from core import FrontEnd, Detector, PoolAttn, OffsetHead
from synth import SynthDS, collate, make_pulse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BAR = 28


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bs", type=int, default=8,
                   help="批大小. 默认 16 是和 --T 10 配对了实测过的: 本进程峰值 1.30GB, 每步音频 160s. "
                        "改回 32 会变 2.50GB 并在 8GB 卡上 cudaErrorUnknown")
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lr-sched", default="const", choices=["const", "cosine"],
                   help="const=恒定学习率(旧行为); cosine=从 --lr 余弦衰减到 0 (配 --warmup). "
                        "长跑(十万步量级)时恒定 lr 会在最优点附近来回跳, 汇总曲线一直抖、best 也难刷新")
    p.add_argument("--warmup", type=int, default=500, help="cosine 的线性 warmup 步数 (const 时忽略)")
    p.add_argument("--workers", type=int, default=12)
    # prefetch_factor 以前硬编码 4. 实测 (2026-09-15) 单窗口合成只要 9.3ms 热 / 35.3ms 冷,
    # 单 worker 就能产 108 窗/s, 而训练最多只要 ~10.5 窗/s —— 12 workers x 4 prefetch
    # = 48 个 batch 在途, 白占 ~3.5GB 内存, 把 mmap 的 bank 页缓存挤出去, 合成退化成冷读,
    # 吞吐从 8.4 it/s 一路掉到 5.5. 降到 4x2 后内存松出来, 速度反而稳在高位.
    p.add_argument("--prefetch", type=int, default=2,
                   help="每个 worker 预取几个 batch. 合成很便宜(9ms/窗), 不需要大预取; "
                        "调大会吃光内存并把页缓存挤掉, 长跑越跑越慢. workers>0 时生效")
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--nh", type=int, default=4,
                   help="注意力头数. 以前写死在 Detector 里, 所以想加宽 d 时没法同步调大 "
                        "(每个头的维度 = d/nh, 必须整除)")
    p.add_argument("--resize", action="store_true",
                   help="【扩大模型时必加】续跑默认【丢弃】命令行里的 d/emb/layers/nh, 采用 checkpoint 的值, "
                        "所以 `--layers 8` 续跑会静默变回 4 (日志里那行 '[resume] layers: 8 -> 4' 就是它). "
                        "加上 --resize 才真按命令行改形状: 形状相符的张量暖启动, 其余重新初始化, 优化器动量清零")
    p.add_argument("--T", type=float, default=20.0,
                   help="输入窗口秒数(默认 10). 长窗口让一个音效可以中途插入/中途退出, 注意力上下文也更长. "
                        "实测 step_cost.py 本进程峰值: T=10/bs=16 -> 1.30GB; T=10/bs=32 -> 2.50GB "
                        "(真跑 train.py 会 cudaErrorUnknown); T=4/bs=32 -> 1.06GB. 注意 T>34 前端 PCEN 会溢出")
    p.add_argument("--max-ev", type=int, default=192,
                   help="每个窗口最多几个事件(数组容量). 10s 里的真实事件数可达几十个, 24 会大量截断")
    p.add_argument("--min-src", type=int, default=1)
    p.add_argument("--max-src", type=int, default=5)
    p.add_argument("--ivl-med", type=float, default=1.0, help="同一来源的普攻间隔中位数(秒)")
    p.add_argument("--ivl-sig", type=float, default=1.2, help="间隔的对数正态 sigma (越大越随机)")
    p.add_argument("--once-len", type=float, default=1.0,
                   help="时长 >= 该值(秒)的素材每个源只播一次, 不按 --ivl-med 重复. "
                        "理由: 重复间隔中位 1.0s 而模板时长中位 2.09s(含战斗语音) -> 长素材会自己叠自己, "
                        "识别池化窗被同类/异类混满, 识别 CE 降不下去. 0=关掉(全部按间隔重复)")
    p.add_argument("--arch", default="v2", choices=["v1", "v2"],
                   help="v1=原架构(无位置编码, 只有 onset 头); "
                        "v2=RoPE 相对位置编码 + 独立的 offset 边界头. "
                        "v2 的 Detector 参数形状和 v1 完全一样(RoPE 不引入参数), 所以可以互相续训")
    p.add_argument("--enc", default="attn", choices=["attn", "crnn", "cnn", "bcresnet"],
                   help="编码器(四者输出形状相同, 头/评测/推理链不用改): "
                        "attn=原注意力栈(O(T^2), 20s 窗口显存随窗口平方涨); "
                        "crnn=卷积 stem + 扩张残差卷积 + BiGRU; cnn=纯卷积(无循环); "
                        "bcresnet=BC-ResNet(2D卷积与时间向1D卷积广播相加, 参数最小)")
    p.add_argument("--off-w", type=float, default=1.0,
                   help="offset(事件结束)边界损失的权重. 只在 --arch v2 生效")
    p.add_argument("--bnd-loss", default="bce", choices=["bce", "focal"],
                   help="边界(脉冲)损失的形状. bce=pos_weight BCE(旧行为); "
                        "focal=arXiv 2601.04178 的 Onset-Offset-Loss (gamma=2, 直接作用在概率上)")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="focal 的 gamma (论文用 2)")
    p.add_argument("--ool-w", type=float, default=15.0,
                   help="focal 边界项的总体权重(只对 --bnd-loss focal 生效). "
                        "实测同一批数据上 focal 比 pos_weight BCE 小约 15 倍 "
                        "(onset 1.040->0.069, offset 1.125->0.068), 所以不加这个权重的话 "
                        "边界监督会从占识别项的 35%% 掉到 1.2%%, 等于静默关掉. 论文用 lambda_ool=100")
    p.add_argument("--pos-weight", type=float, default=10.0)
    p.add_argument("--pool-mode", default="mean", choices=["mean", "attn"],
                   help="mean=固定窗算术平均(默认); attn=让模型自己学每帧权重 (子集包含 mean 和整段均值). "
                        "加权后有效窗长会打在汇总行里, 可以直接看出模型想要多长")
    p.add_argument("--pool-frames", type=int, default=12,
                   help="识别池化窗长(输出帧/20ms). 0=整段(旧行为); 12=240ms. 实测 pool_sweep.py: "
                        "240ms 在四档全面优于旧的 600ms, dense +9.2 点. 训练和 infer.py 必须用同一个值")
    p.add_argument("--peak-w", type=float, default=0.0,
                   help="【onset 必须高过后续持续段】hinge 损失的权重 (0=关). "
                        "针对实测病灶: 99%% 的假峰落在正在响的事件内部、中位相对位置 0.49")
    p.add_argument("--peak-win", type=int, default=25,
                   help="往后检查多长(输出帧/20ms); 25 = 500ms")
    p.add_argument("--peak-margin", type=float, default=2.0,
                   help="onset logit 要比后面最高的那个高出多少; 2.0 约等于几率 7.4 倍")
    p.add_argument("--le-w", type=float, default=1.0,
                   help="识别损失权重: loss = 检测BCE + le_w * 识别CE. 尖峰 onset 目标会让检测项变难, "
                        "硬相加时它会压制共享主干上的识别头 (实测 gauss 后 top1 -8)")
    p.add_argument("--left-pad", type=float, default=0.0,
                   help="源的 onset 最早能比窗口起点提前多久(秒), 也就是【已经在响】能响到第几秒. "
                        "这种事件只标 span 不给 onset 脉冲")
    p.add_argument("--p-mid", type=float, default=0.3,
                   help="一个源【t=0 时已经在响】的概率 (onset 落在窗口外). 0=关掉, 退回旧行为")
    p.add_argument("--src-rate-ref", type=float, default=7.5,
                   help="源数 = 抽到的源数 x max(1, T/该值). 7.5 = 实测校准到事件密度与 T 无关 "
                        "(T=10 -> 1.93 事件/秒, 和 T=4 的 1.87 持平). 调到 4.0 会加密到 3.4 事件/秒")
    p.add_argument("--src-span-lo", type=float, default=0.25,
                   help="一个源存活时长至少占【剩余窗口】的比例 (1.0 = 一定响到窗口末尾, 即旧行为)")
    p.add_argument("--bg-mode", default="mixed",
                   choices=["mixed", "noise", "silence", "recording", "long"],
                   help="训练背景. mixed=噪声/静音(旧行为). long=真实长音(BGM+环境+干员战斗语音), "
                        "需要先跑 x_extract_long.py all + build_long.py. 默认不改, 要换背景得显式指定")
    p.add_argument("--bg-db", type=float, default=None,
                   help="long 档背景电平 (RMS dBFS, bg_lo=bg_hi=该值). 不给就用 SynthDS 默认 -70..-40")
    p.add_argument("--long-voice-p", type=float, default=0.65,
                   help="long 档下一个窗口里出现干员战斗语音的概率")
    p.add_argument("--dense-frac", type=float, default=0.6)
    p.add_argument("--silence-frac", type=float, default=0.2)
    p.add_argument("--empty-frac", type=float, default=0.2)
    p.add_argument("--out", default=os.path.join(ML, "ckpt"))
    p.add_argument("--resume", default="", help="checkpoint 路径, 或 'latest' 取最新的")
    p.add_argument("--tag", default="", help="本次运行的标签(默认时间戳); 续跑时默认沿用源 ckpt 的标签")
    p.add_argument("--logevery", type=int, default=50)
    p.add_argument("--summary-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast 前向/反向 (~1.5-2x); 余弦头保持 fp32")
    p.add_argument("--label-mode", default="cluster", choices=["cluster", "template"],
                   help="cluster=512 粗类; template=直接 6576 个模板 (方法① 全库 GEMM)")
    p.add_argument("--best-metric", default="top1", choices=["top1", "set1", "f1", "none"],
                   help="按训练批 EMA 另存一份 best_<tag>.pt (不覆盖); none=关闭")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="每 N 步另存 snap_<tag>_sN.pt 快照 (0=关闭, 不覆盖)")
    p.add_argument("--shuffle-block", type=int, default=2048,
                   help="按块打乱样本顺序的块大小; 0=顺序读(旧行为)")
    p.add_argument("--onset-shape", default="rect", choices=["rect", "tri", "gauss"],
                   help="onset 目标脉冲形状. rect=50ms 方波(现状; 平台没有梯度, 峰在哪都行 -> 定位抖); "
                        "tri/gauss=尖峰, 定位更锐. 注意帧级 F1 会因真值变窄而下降, 要看 loc_eval 的事件级指标")
    p.add_argument("--onset-len", type=int, default=5, help="脉冲长度(10ms 帧), rect/tri 用")
    p.add_argument("--onset-sigma", type=float, default=2.5, help="gauss 的 sigma(10ms 帧)")
    p.add_argument("--tir-conf", type=float, default=0.0,
                   help="识别损失改用【集合目标】: 与真实标签波形相似度 >= 该值的其它模板也算对. 0=关(退化成普通 CE)")
    p.add_argument("--tir-max", type=int, default=8,
                   help="每个标签最多并入几个近重复模板, 防集合坍塌")
    p.add_argument("--tir-file", default=os.path.join(ML, "tir_6576.npz"),
                   help="混淆图, build_confusion.py 的产物 (仅 label-mode=template)")
    return p.parse_args()


def set_ce(logits, target, nbr, val, conf, kmax):
    """集合目标 CE: 概率质量落在 {target} ∪ {近重复邻居} 内就算对.

    conf=0 (或邻居表为空) 时集合只剩 target 自己, 与 F.cross_entropy 逐位相同.
    动机: 大量模板在听觉上互为近重复, 强迫模型在它们之间选一个等于在拟合标签噪声.
    """
    logp = F.log_softmax(logits, dim=-1)
    nb = nbr[target][:, :kmax]
    ok = (val[target][:, :kmax] >= conf) & (nb != target[:, None])
    own = logp.gather(1, target[:, None])
    neg = torch.full(ok.shape, float("-inf"), dtype=logp.dtype, device=logp.device)
    sel = torch.where(ok, logp.gather(1, nb), neg)
    return -torch.logsumexp(torch.cat([own, sel], 1), dim=1).mean()


def focal_prob(p, y, gamma=2.0, eps=1e-6):
    """arXiv 2601.04178 的 Onset-Offset-Loss: focal 直接作用在【边界概率】上, alpha(=gamma)=2.

        y=1: (1-p)^gamma * (-log p)        y=0: p^gamma * (-log(1-p))

    gamma=0 时逐位退化成普通 BCE(p, y). 我们的边界目标是脉冲(软标签), 所以写成
    |y-p|^gamma * [-y log p - (1-y) log(1-p)]; y in {0,1} 时与论文 Eq.4 完全一致.
    论文用 lambda_ool 代替 pos_weight 来平衡稀疏脉冲 —— 这正是它取代 pos_weight BCE 的理由.
    """
    p = p.clamp(eps, 1.0 - eps)
    ce = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
    return ((y - p).abs() ** gamma * ce).mean()


def bnd_loss(logit, target, kind, gamma, pos_weight):
    """边界(脉冲)损失. bce=旧行为(pos_weight BCE); focal=论文的 OOL."""
    if kind == "focal":
        return focal_prob(torch.sigmoid(logit), target, gamma)
    return F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=torch.tensor(pos_weight, device=logit.device))


def pulse_at(frames, T, pulse, valid, backward=False):
    """把 pulse 写到 (B,T) 的输出网格上; backward=True 时脉冲在 frames 处【结束】(offset 用).

    无效事件把 frames 置到 >=T 即可: 越界的 idx 会被 ok 掩掉, 写进去的是 0, 对 amax 无影响.
    """
    B = frames.shape[0]; L = pulse.numel()
    out = torch.zeros(B, T, device=frames.device, dtype=pulse.dtype)
    start = (frames - (L - 1)) if backward else frames
    idx = start[..., None] + torch.arange(L, device=frames.device)
    ok = valid[..., None] & (idx >= 0) & (idx < T)
    src = pulse.view(1, 1, L).expand_as(idx).to(pulse.dtype) * ok.to(pulse.dtype)
    out.scatter_reduce_(1, idx.clamp_(0, T - 1).reshape(B, -1), src.reshape(B, -1), reduce="amax")
    return out


def onset_margin(logit, ev, valid, W, margin):
    """onset 必须在【事件开头】显著高于它后面那一段持续段.

    为什么是这个形状 (而不是窗口内局部 softmax): loc_eval 的假峰分类显示, 99% 的假峰落在
    正在响的事件区间【内部】, 相对位置中位 0.49, 只有 13~21% 靠近某个 onset —— 也就是说
    onset 头在"一个声音已经响到一半"的地方也会报事件. 逐帧 BCE 教不会它这件事.
    +-100ms 的局部 softmax 也管不到 (那里离 onset 有 500ms), 只能把"开头要高过后面"写成 hinge:
        relu( max_{t in (a, a+W]} z[t]  -  z[a]  +  margin )
    窗口里若还有别的真值 onset, 那些帧要排除, 否则等于强迫模型漏检.

    ev[..., 0] < 0 = onset 在窗口【外】(合成器里"声音已经在响"的事件, 见 synth.py). 这种事件
    只标了 span, 没有 onset 脉冲, 所以也不能要求它在帧 0 处出峰 —— vo 把它们排除掉.
    """
    B, Tm = logit.shape
    vo = valid & (ev[..., 0] >= 0)                                           # (B,G) 有 onset 落在窗口内
    a = (ev[..., 0] // 2).clamp(min=0, max=Tm - 1)                           # (B,G) onset 输出帧
    b = ((ev[..., 1].clamp(min=0) + 1) // 2).clamp(min=1, max=Tm)            # (B,G)
    i = torch.arange(Tm, device=logit.device)[None, None, :]                 # (1,1,Tm)
    later = (i > a[..., None]) & (i <= (a + W)[..., None]) & (i < b[..., None])
    other = (((i - a[..., None]).abs() <= 2) & vo[..., None]).any(1)         # (B,Tm) 任一真 onset 的 +-2 帧
    keep = later & (~other[:, None, :]) & vo[..., None]
    zl = logit[:, None, :].masked_fill(~keep, -1e9).max(dim=-1).values       # (B,G)
    za = logit.gather(1, a)                                                  # (B,G)
    return F.relu(zl - za + margin)[vo].mean()


def hms(s):
    s = int(max(0.0, s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


class BlockShuffleSampler(torch.utils.data.Sampler):
    """按块打乱样本顺序.

    没有它, DataLoader 顺序读 + SynthDS.__getitem__(i) 是 i 的确定函数 (rng 由 seed 和 i 决定)
    => 每个进程都从样本 0 按同一条序列重播, 于是 "第 N 步" 和 "数据位置" 完全混淆: 重启起点不同
    的两个进程, 同一个 step 看到的是完全不同的数据; 而同一个数据位置总是出现在重启后同样的步数上,
    所以根本无法区分指标变化是来自模型还是来自数据.

    块内保持顺序 (保住 variants.npy 2.4GB memmap 的页缓存局部性), 只在块之间打乱. 种子挂 step0,
    这样每次重启拿到的顺序都不同.
    """

    def __init__(self, n, block=2048, seed=0):
        self.n = int(n); self.block = max(int(block), 1); self.seed = int(seed)

    def __iter__(self):
        g = np.random.default_rng(self.seed)
        blocks = np.arange(0, self.n, self.block)
        g.shuffle(blocks)
        for b in blocks:
            yield from range(int(b), min(int(b) + self.block, self.n))

    def __len__(self):
        return self.n


def resolve_resume(v, out):
    """-> (ckpt 路径或 None, 该 ckpt 的 tag)"""
    if not v:
        return None, ""
    if v == "latest":
        c = [os.path.join(out, f) for f in os.listdir(out)
             if f.startswith("ckpt_") and f.endswith(".pt")]
        if not c:
            raise SystemExit("--resume latest: %s 下没有 ckpt_*.pt" % out)
        v = max(c, key=os.path.getmtime)
    if not os.path.exists(v):
        raise SystemExit("找不到 checkpoint: %s" % v)
    ck = torch.load(v, map_location="cpu")
    return v, str(ck.get("tag", "") or "")


def main():
    args = build_args()
    os.makedirs(args.out, exist_ok=True)
    src_ckpt, src_tag = resolve_resume(args.resume, args.out)
    run_id = args.tag or src_tag or time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out, "train_%s.log" % run_id)
    ckpt_path = os.path.join(args.out, "ckpt_%s.pt" % run_id)
    # 名字刻意不带 ckpt_ 前缀: resolve_resume("latest") 只 glob ckpt_*.pt, 不会被 best/snap 干扰
    best_path = os.path.join(args.out, "best_%s.pt" % run_id)
    logf = open(log_path, "a", encoding="utf-8")

    def out(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # CUDA 预检. 以前这里直接 torch.device("cuda"), 环境里装的是 CPU 版 torch 时会在
    # 后面 .to(dev) 处抛 AssertionError, 而崩溃日志又被静默吞掉 -> 表现成"训练莫名其妙没了".
    # 现在提前报清楚, 并把诊断同时写进训练日志和 stderr.
    if not torch.cuda.is_available():
        _v = torch.__version__
        _cpu = _v.endswith("+cpu") or torch.version.cuda is None
        _lines = [
            "",
            "=" * 74,
            "训练无法开始:  torch.cuda.is_available() == False",
            "  torch 版本        %s%s" % (_v, "    <- CPU 版 (PyPI 默认轮子)" if _cpu else ""),
            "  torch 编译的 CUDA %s" % (torch.version.cuda or "None (CPU-only build)"),
            "  当前解释器        %s" % sys.executable,
            "",
        ]
        if _cpu:
            _lines += [
                "  装 CUDA 版即可 (约 2.5GB):",
                "    \"%s\" -m pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124" % sys.executable,
                "",
            ]
        _lines += [
            "  装完先自检 (两条都要过):",
            "    nvidia-smi",
            "    \"%s\" -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"" % sys.executable,
            "  注意: 在 Windows 上直接 `pip install torch` 装到的是 CPU 版, 会把 CUDA 版覆盖掉.",
            "=" * 74,
            "",
        ]
        for _l in _lines:
            out(_l)
        raise SystemExit(2)

    dev = torch.device("cuda")
    torch.manual_seed(0)
    # TF32: fp32 matmul/conv go through tensor cores.  Free speed, no API change.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = bool(getattr(args, "amp", False))
    ck = None
    _shape_keys = ("d", "emb", "layers", "nh")
    if src_ckpt:
        ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        sa = ck.get("args", {}) or {}
        # label_mode 必须跟 checkpoint 走: 换了标签模式 K 就变了, 识别头形状对不上会直接崩
        if "label_mode" in sa and args.label_mode != sa["label_mode"]:
            out("  [resume] label_mode: %s -> %s  (取 checkpoint 的值)"
                % (args.label_mode, sa["label_mode"]))
            args.label_mode = sa["label_mode"]
        if args.resize:
            _chg = [(k, getattr(args, k), sa[k]) for k in _shape_keys
                    if k in sa and getattr(args, k) != sa[k]]
            if _chg:
                out("  [resume] --resize: 按本次命令行改形状  %s"
                    % "  ".join("%s %s->%s" % (k, o, n) for k, n, o in _chg))
                out("            形状对不上的参数重新初始化, 其余暖启动; Adam 动量清零")
            else:
                out("  [resume] --resize: 形状没变, 等同于普通续跑")
        else:
            for k in _shape_keys:
                if k in sa and getattr(args, k) != sa[k]:
                    out("  [resume] %s: %s -> %s  (取 checkpoint 的值; 想改形状要显式加 --resize)"
                        % (k, getattr(args, k), sa[k]))
                    setattr(args, k, sa[k])
    if args.label_mode == "template":
        K = int(len(np.load(os.path.join(ML, "bank_lens.npy"))))
    else:
        cj = json.load(open(os.path.join(ML, "clusters_512.json"), encoding="utf-8"))
        K = int(cj["C"])
    rnd = 1.0 / K

    out("音效检测器 (纯合成训练)   类别 C=%d   top-1 随机基线 %.2f%%   目标 top-1>=70%%, 帧F1>=0.85" % (K, rnd * 100))
    out("数据  静音底%.0f%%/噪底%.0f%%  %.0f%% 空窗  %.0f%% 拉到最大源数" % (
        args.silence_frac * 100, (1 - args.silence_frac) * 100, args.empty_frac * 100,
        args.dense_frac * 100))
    # 这行以前没有: 背景模式不打印的话, 事后翻日志根本看不出这次到底训的是噪声底还是真实长音
    # (checkpoint 里的 args.bg_mode 是唯一线索, 而旧 checkpoint 连这个键都没有)
    _bgr = ("bg_lo=bg_hi=%.1f dBFS" % args.bg_db) if args.bg_db is not None else "电平用 SynthDS 默认"
    out("      背景模式: %s  (%s)%s" % (
        args.bg_mode, _bgr,
        "   干员战斗语音出现概率 %.2f" % args.long_voice_p if args.bg_mode == "long" else ""))
    if args.bg_mode == "long" and not os.path.exists(os.path.join(ML, "long_pool.npy")):
        out("  【警告】--bg-mode long 但 ml/long_pool.npy 不存在 -> SynthDS 会静默退化成噪声底! "
            "先跑 x_extract_long.py all + build_long.py")
    out("      源模型: 每段 %d~%d 个源 (按 T/%.1f 缩放), 每个源 = 同一音效按【随机间隔】重复 "
        "(中位 %.2fs, 对数正态 sigma %.2f); 时长 >= %.2fs 的素材只播一次%s" % (
            args.min_src, args.max_src, args.src_rate_ref, args.ivl_med, args.ivl_sig,
            args.once_len, "" if args.once_len > 0 else "（已关掉：全部重复）"))
    # left_pad<=0 时 SynthDS 内部会取整个窗口长度, 这里必须打印【生效值】而不是原始入参,
    # 否则日志上写着"最多提前 0.00s"而实际是 20s, 读日志的人会以为中段播放没开
    _lp = args.left_pad if args.left_pad > 0 else args.T
    out("      源活跃区间: 插入时刻 U(0,T), %.0f%% 的源在 t=0 时【已经在响】(最多提前 %.2fs, left_pad=%g); "
        "退出时刻 >= 剩余窗口的 %.0f%%  -> 音效可中途插入/中途退出" % (
            args.p_mid * 100, _lp, args.left_pad, args.src_span_lo * 100))
    out("      跨窗口边界: onset 落在窗口外的事件只标注 span (ev[...,0]<0), 不产生 onset 脉冲")
    if args.T * args.bs > 480:      # 本机 4060(8GB) 实测, T=20s + attn + bf16 + 前向+反传:
        out("  【显存提醒】T=%.0fs x bs=%d 超出了实测安全区" % (args.T, args.bs))
        out("      T=20s 实测: bs=8→1.85GB  bs=12→2.74GB  bs=16→3.62GB  bs=20→4.50GB"
            "（6G 预算建议 bs<=16；crnn/cnn/bcresnet 编码器比 attn 更省）")
    out("目标  检测头学【事件起始脉冲】(起点后 50ms), 不是整段存在 -> 连续平A也能逐个分辨")
    out("字段  lo=检测BCE  le=识别CE(两者硬相加成反传的 loss)   F1=检测帧级F1   top1/top5=识别准确率(训练批, 滑动平均)")

    fe = FrontEnd().to(dev).eval()
    _rope = (args.arch == "v2")
    args.rope = _rope     # 显式写进 checkpoint: 否则下游只能看到 arch, 猜错就会静默按无位置编码前向
    model = Detector(K, d=args.d, emb=args.emb, nl=args.layers, nh=args.nh, rope=_rope,
                     enc=args.enc).to(dev)
    if args.pool_mode == "attn":
        model.pooler = PoolAttn(args.emb).to(dev)      # 挂成子模块 -> state_dict 自动带上
    off_head = OffsetHead(args.d).to(dev) if _rope else None
    npar = sum(q.numel() for q in model.parameters()) / 1e6
    _pp = list(model.parameters()) + (list(off_head.parameters()) if off_head else [])
    opt = torch.optim.AdamW(_pp, lr=args.lr, weight_decay=0.01)
    pulse2 = None
    if off_head is not None:      # offset 目标脉冲, 和 onset 的 po2 走同一套降采样
        _pl = torch.from_numpy(make_pulse(args.onset_shape, args.onset_len, args.onset_sigma)).to(dev)
        pulse2 = F.max_pool1d(_pl[None, None], 2, 2)[0, 0]
    _off_sd = None                # 每个 checkpoint 都带上 offset 头 (没启用就是 None)
    step0 = 0
    if ck is not None:
        _sd = ck.get("model") or {}
        if args.resize:
            # 形状不符的键【不能】直接喂给 load_state_dict: strict=False 只放过"缺失/多余",
            # 尺寸不一致仍然抛 RuntimeError. 所以先按形状筛一遍.
            _mine = model.state_dict()
            _keep = {k: v for k, v in _sd.items()
                     if k in _mine and tuple(_mine[k].shape) == tuple(v.shape)}
            _fresh = sorted(k for k in _mine if k not in _keep)
            _drop = sorted(k for k in _sd if k not in _keep)
            model.load_state_dict(_keep, strict=False)
            out("  [resume] 暖启动 %d/%d 个张量; 重新初始化 %d 个 (新增或形状变了)"
                % (len(_keep), len(_mine), len(_fresh)))
            if _drop:
                out("            丢弃旧张量 %d 个: %s%s"
                    % (len(_drop), ", ".join(_drop[:6]), " ..." if len(_drop) > 6 else ""))
            if _fresh:
                out("            重新初始化: %s%s"
                    % (", ".join(_fresh[:6]), " ..." if len(_fresh) > 6 else ""))
        else:
            _miss, _unexp = model.load_state_dict(_sd, strict=False)
            if _miss:
                out("  [resume] 本次新增的参数(旧 checkpoint 里没有, 用初始值): %s"
                    % ", ".join(list(_miss)[:8]) + (" ..." if len(_miss) > 8 else ""))
        _carch = str(sa.get("arch", "v1"))
        if _carch != args.arch:
            out("  [resume] 架构: 记录 %s -> 本次 %s  (v2 是在 v1 权重上【加】RoPE 和 offset 头, "
                "形状不变, 所以这是一次合法的暖启动)" % (_carch, args.arch))
        if off_head is not None and ck.get("offset_head") is not None:
            off_head.load_state_dict(ck["offset_head"])
            out("  [resume] offset 边界头已从 checkpoint 载入")
        if "opt" in ck:                      # absent when warm-starting with a resized proto head
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as _e:          # 参数集合变了(比如 v1->v2 多出 offset 头)
                out("  [resume] 优化器状态载不进去 (%s), Adam 动量从零开始" % str(_e).split("(")[0])
        step0 = int(ck["step"])
        if args.steps <= step0:
            raise SystemExit("--steps %d 必须大于 checkpoint 的 step %d, 否则一步都不会跑 "
                             "(旧版本会静默跳过并报出一个假的 it/s)" % (args.steps, step0))
        out("  续跑自 %s  (起点 step %d, 沿用标签 %s)" % (os.path.basename(src_ckpt), step0, run_id))
    out("  标签 tag      %s" % run_id)
    out("  日志          %s" % log_path)
    out("  检查点        %s   (每 %d 步覆盖写)" % (ckpt_path, args.ckpt_every))
    if args.best_metric != "none":
        out("  最佳存档      %s   (按最近 %d 步的%s均值, 每 %d 步判一次, 不覆盖)"
            % (best_path, args.summary_every, args.best_metric, args.summary_every))
    if args.snapshot_every:
        out("  周期快照      snap_%s_s<N>.pt  每 %d 步" % (run_id, args.snapshot_every))
    out("%s | %.2fM 参数 | bs=%d T=%.1fs workers=%d prefetch=%d | TF32 on | AMP %s" % (
        torch.cuda.get_device_name(0), npar, args.bs, args.T, args.workers,
        args.prefetch if args.workers > 0 else 0,
        "bf16" if amp else "off (fp32)"))
    out("  形状          d=%d  layers=%d  nh=%d (头维 %d)  emb=%d   K=%d" % (
        args.d, args.layers, args.nh, args.d // max(args.nh, 1), args.emb, K))
    out("  学习率        %s%s" % (
        ("%.1e -> 0 余弦 (warmup %d 步)" % (args.lr, args.warmup)) if args.lr_sched == "cosine"
        else ("%.1e 恒定 (旧行为; 十万步以上的长跑建议换 cosine)" % args.lr),
        ""))
    out("开始 %d 步    %s" % (args.steps, time.strftime("%H:%M:%S")))
    out()

    tir_nbr = tir_val = None
    if args.tir_conf > 0:
        if args.label_mode != "template":
            raise SystemExit("--tir-conf 目前只支持 --label-mode template (混淆图是按 6576 个模板建的)")
        if not os.path.exists(args.tir_file):
            raise SystemExit("找不到混淆图 %s, 先跑 build_confusion.py" % args.tir_file)
        _z = np.load(args.tir_file)
        tir_nbr = torch.from_numpy(_z["nbr"].astype(np.int64)).to(dev)
        tir_val = torch.from_numpy(_z["val"].astype(np.float32)).to(dev)
        _sz = (tir_val[:, :args.tir_max] >= args.tir_conf).sum(1).float().cpu()
        out("  集合目标      --tir-conf %.2f  --tir-max %d   %s" % (
            args.tir_conf, args.tir_max, os.path.basename(args.tir_file)))
        out("                每个标签平均并入 %.2f 个近重复 (p90 %.0f, 最大 %.0f); top1 之外额外统计 set1"
            % (float(_sz.mean()), float(_sz.quantile(0.9)), float(_sz.max())))

    ds = SynthDS(T=args.T, length=10 ** 7, seed=1, max_ev=args.max_ev,
                 dense_frac=args.dense_frac, silence_frac=args.silence_frac,
                 empty_frac=args.empty_frac, min_src=args.min_src, max_src=args.max_src,
                 ivl_med=args.ivl_med, ivl_sig=args.ivl_sig, once_len=args.once_len,
                 label_mode=args.label_mode,
                 onset_shape=args.onset_shape, onset_len=args.onset_len, onset_sigma=args.onset_sigma,
                 left_pad=args.left_pad, p_mid=args.p_mid, src_rate_ref=args.src_rate_ref,
                 src_span_lo=args.src_span_lo, bg_mode=args.bg_mode, long_voice_p=args.long_voice_p,
                 **({"bg_lo": args.bg_db, "bg_hi": args.bg_db} if args.bg_db is not None else {}))
    _oshape = {"rect": "rect %d帧(%.0fms) 方波" % (args.onset_len, args.onset_len * 10),
               "tri": "tri %d帧(%.0fms)" % (args.onset_len, args.onset_len * 10),
               "gauss": "gauss sigma=%.1f帧(%.0fms)" % (args.onset_sigma, args.onset_sigma * 10)
               }[args.onset_shape]
    out("  onset 目标    %s" % _oshape)
    out("  识别池化      %s" % ("整段 (旧行为)" if args.pool_frames == 0
                              else "onset 起 %d 帧 (%.0f ms)" % (args.pool_frames, args.pool_frames * 20))
        + ("   [attn: 窗长由模型自己学]" if args.pool_mode == "attn" else ""))
    out("  边界损失      %s%s" % (
        ("focal (gamma=%.1f, 论文 OOL, 总体权重 x%.0f)" % (args.focal_gamma, args.ool_w)) if args.bnd_loss == "focal"
        else ("pos_weight BCE (pos_weight=%.0f)" % args.pos_weight),
        "" if off_head is not None else "   [arch v1: 没有 offset 头, 权重忽略]"))
    out("  损失权重      loss = 检测(%s)%s + %.2f x offset(%s) + %.2f x 识别CE" % (
        args.bnd_loss,
        (" + %.2f x onset-margin(后 %d帧 需高 %.1f)" % (args.peak_w, args.peak_win, args.peak_margin))
        if args.peak_w > 0 else "",
        args.off_w if off_head is not None else 0.0, args.bnd_loss, args.le_w))
    out("  架构          %s / 编码器 %s%s" % (args.arch, args.enc,
        "   RoPE 相对位置编码 (修掉'打乱帧序输出只是跟着打乱'那个洞) + 独立 offset 边界头"
        if off_head is not None else "   (无位置编码, 只有 onset 头)"))
    if args.shuffle_block > 0:
        sampler = BlockShuffleSampler(len(ds), args.shuffle_block, seed=1000 + step0)
        out("  采样顺序      块打乱 (block=%d, seed=%d) -> 每次重启看到的数据都不同"
            % (args.shuffle_block, 1000 + step0))
    else:
        sampler = None
        out("  采样顺序      顺序读 (step 与数据位置混淆, 不建议)")
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=args.workers > 0, sampler=sampler,
                    prefetch_factor=args.prefetch if args.workers > 0 else None)

    t_start = time.time(); t_win = t_start
    _off_sd = off_head.state_dict() if off_head is not None else None
    acc = dict(loss=0.0, pk=0.0, le=0.0, off=0.0, f1=0.0, t1=0.0, t5=0.0, s1=0.0, pw=0.0, n=0,
               last_loss=0.0, last_pk=0.0, last_le=0.0, last_f1=0.0,
               last_t1=0.0, last_t5=0.0, last_s1=0.0)
    rate = 0.0
    best_val = -1.0; best_step = 0
    # 续跑时必须先把已经存下来的记录读回来. 否则重启后的第一次汇总一定"破纪录",
    # 把之前真正的峰值直接覆盖掉 —— 实测 rect 的 0.6898 就是这样被 gauss 的 0.6071 冲掉的.
    if args.best_metric != "none" and os.path.exists(best_path):
        try:
            _b = torch.load(best_path, map_location="cpu", weights_only=False)
            _bb = _b.get("best") or {}
            _rs = int(_bb.get("step", 0))
            _ba = _b.get("args") or {}
            _t = _ba.get("T")
            _oa = str(_ba.get("arch", "v1"))
            _ok = (_bb.get("metric") == args.best_metric and int(_b.get("K", -1)) == K
                   and str(_b.get("tag", "")) == run_id)
            # 可比性签名: 窗口长度 + 架构 + 形状 (d/emb/layers/nh). 少了形状这一项的话,
            # 扩大模型后新模型的 top1 会被拿去和【小模型】的记录比, 低的永远压不过去.
            _dim_old = (int(_ba.get("d", 192)), int(_ba.get("emb", 256)),
                        int(_ba.get("layers", 4)), int(_ba.get("nh", 4)))
            _dim_now = (int(args.d), int(args.emb), int(args.layers), int(args.nh))
            _same = ((_t is None) or (abs(float(_t) - float(args.T)) <= 1e-6 and _oa == args.arch
                                      and _dim_old == _dim_now))
            if _ok and not _same:
                # 窗口长度/架构/形状变了 -> top1 是在【不同难度/不同模型族】上算的, 根本不可比.
                # 不处理的话, 新配置下更低的分数可能永远压不过旧记录, best 再也不更新.
                _sig = "T%.1f_%s_d%de%dl%d" % (float(_t), _oa, _dim_old[0], _dim_old[1], _dim_old[2])
                _arch = os.path.join(args.out, "best_%s_s%d_%s.pt" % (run_id, _rs, _sig))
                if not os.path.exists(_arch):
                    os.replace(best_path, _arch)
                # 说清楚到底哪一项不可比. 旧版只印 T 和 arch, 于是形状变了的时候会印出
                # "记录是 v2 (T=10.0s), 本次是 v2 (T=10.0s)" —— 看着像什么都没变.
                _why = []
                if _t is not None and abs(float(_t) - float(args.T)) > 1e-6:
                    _why.append("窗口 T %.1fs -> %.1fs" % (float(_t), float(args.T)))
                if _oa != args.arch:
                    _why.append("架构 %s -> %s" % (_oa, args.arch))
                if _dim_old != _dim_now:
                    _why.append("形状 d%de%dl%d -> d%de%dl%d"
                                % (_dim_old[0], _dim_old[1], _dim_old[2],
                                   _dim_now[0], _dim_now[1], _dim_now[2]))
                out("  历史最佳      与旧记录不可比 (%s) -> top1 不横比" % "；".join(_why))
                out("                旧记录已留档为 %s, 本次从 -1 重新计"
                    % os.path.basename(_arch))
            elif _ok:
                if step0 >= _rs:
                    best_val = float(_bb.get("value", -1.0)); best_step = _rs
                    out("  历史最佳      %s %.4f @step %d   (继承自 %s, 只有超过它才覆盖)"
                        % (args.best_metric, best_val, best_step, os.path.basename(best_path)))
                else:
                    # 从比记录更早的点重开 = 新的一条血脉. 旧记录改名留档, 不许被这次的
                    # 低分直接冲掉 (实测: 从头开训的第一条汇总就把 step40000/0.6148 换成了 step2000/0.3095)
                    _arch = os.path.join(args.out, "best_%s_s%d.pt" % (run_id, _rs))
                    if not os.path.exists(_arch):
                        os.replace(best_path, _arch)
                    out("  历史最佳      本条从 step %d 重开, 记录却在 step %d -> 旧记录已留档为 %s, 本次从 -1 重新计"
                        % (step0, _rs, os.path.basename(_arch)))
            else:
                out("  历史最佳      忽略 %s: metric/K/tag 不匹配 (%s / K=%s / %s)"
                    % (os.path.basename(best_path), _bb.get("metric"), _b.get("K"), _b.get("tag")))
        except Exception as _e:
            out("  历史最佳      读取 %s 失败 (%s), 从 -1 开始" % (os.path.basename(best_path), _e))
    win = {"t1": 0.0, "s1": 0.0, "f1": 0.0, "n": 0}
    ema = {"loss": 0.0, "pk": 0.0, "le": 0.0, "f1": 0.0, "t1": 0.0, "t5": 0.0, "s1": 0.0, "n": 0}
    it = iter(dl)
    for step in range(step0 + 1, args.steps + 1):
        # 学习率调度按【绝对 step】算 (step 不是本次运行的相对步数), 所以中断后续跑曲线是连续的,
        # 不会重新 warmup. 想改总预算就改 --steps, 曲线会随之拉伸.
        if args.lr_sched == "cosine":
            if args.warmup > 0 and step <= args.warmup:
                _f = step / float(args.warmup)
            else:
                _f = 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, args.steps))))
            for _g in opt.param_groups:
                _g["lr"] = args.lr * max(_f, 0.0)
        try:
            mix, po, ev, lb = next(it)
        except StopIteration:
            it = iter(dl); mix, po, ev, lb = next(it)
        mix = mix.to(dev, non_blocking=True)
        po = po.to(dev, non_blocking=True); ev = ev.to(dev); lb = lb.to(dev)
        with torch.no_grad():
            mel = fe(mix)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _o = model(mel, return_all=(off_head is not None))
            _offl = off_head(_o["h"]) if off_head is not None else None
        # EVERYTHING below stays fp32.  autocast casts matmul-like ops regardless of input dtype,
        # so torch.einsum inside the context would come back bf16 and the fp32 cosine head would
        # raise "expected mat1 and mat2 to have the same dtype".
        logit = (_o["onset"] if off_head is not None else _o[0]).float()
        e = (_o["emb"] if off_head is not None else _o[1]).float()
        offl = _offl.float() if _offl is not None else None
        Tm = logit.shape[1]
        po2 = F.max_pool1d(po.unsqueeze(1), 2, 2).squeeze(1)[:, :Tm]
        # focal 的量级比 BCE 小一个数量级, 用 --ool-w 补回来, 否则不是 A/B 而是"关掉边界监督"
        _bw = args.ool_w if args.bnd_loss == "focal" else 1.0
        lo = _bw * bnd_loss(logit[:, :po2.shape[1]], po2, args.bnd_loss, args.focal_gamma, args.pos_weight)
        # max_ev 只是【容量上限】, 不该按它付费: 一个 batch 里 G 个槽位大部分是 padding,
        # 但识别头是 pooled @ proto.t() 这种 (B,G,K) 的 GEMM, G=128 时空槽位一样要算/要显存
        # (B=32,K=6576,G=128 -> 107MB 一个张量). 先按本批真实事件数裁掉尾巴.
        G = max(1, int((lb >= 0).sum(1).max()))
        if G < lb.shape[1]:
            ev = ev[:, :G]; lb = lb[:, :G]
        B, Tm2, E = e.shape
        st = (ev[..., 0].clamp(min=0) // 2).clamp(max=Tm2 - 1)
        en_full = ((ev[..., 1].clamp(min=0) + 1) // 2).clamp(min=1, max=Tm2)
        # offset 边界目标: 在事件【结束帧】反向贴一个同形状脉冲 (arch=v2)
        if offl is not None:
            _T2 = po2.shape[1]
            po_off = pulse_at(en_full - 1, _T2, pulse2, lb >= 0, backward=True)
            lo_off = _bw * bnd_loss(offl[:, :_T2], po_off, args.bnd_loss, args.focal_gamma, args.pos_weight)
        # 池化窗: 0=整段(旧行为); >0 = 只从 onset 起取 N 个输出帧. 训练/推理必须一致.
        en = torch.minimum(en_full, st + args.pool_frames) if args.pool_frames > 0 else en_full
        ar = torch.arange(Tm2, device=dev)[None, None, :]
        msk = ((ar >= st[..., None]) & (ar < en[..., None])) & (lb[..., None] >= 0)
        cnt = msk.sum(-1, keepdim=True).clamp(min=1)
        if args.pool_mode == "attn":
            _p, _w = model.pooler(e, msk)
            pooled = F.normalize(_p, dim=-1)
            # 有效窗长 = 权重的参与率(帧). masked_fill 后外面那圈权重为 0, 不影响.
            acc["pw"] += float((1.0 / (_w ** 2).sum(-1).clamp(min=1e-9))[(lb >= 0)].mean().detach())
        else:
            pooled = F.normalize(torch.einsum("bgt,btd->bgd", msk.float(), e) / cnt, dim=-1)
        valid = lb >= 0
        cls = pooled @ F.normalize(model.proto, dim=-1).t() / model.tau
        if tir_nbr is None:
            le = F.cross_entropy(cls[valid], lb[valid])
        else:
            le = set_ce(cls[valid], lb[valid], tir_nbr, tir_val, args.tir_conf, args.tir_max)
        von = valid & (ev[..., 0] >= 0)      # 只有 onset 在窗口内的事件才参与 onset-margin
        if args.peak_w > 0 and bool(von.any()):
            lp = onset_margin(logit[:, :Tm2], ev, valid, args.peak_win, args.peak_margin)
        else:
            lp = torch.zeros((), device=dev)
        loss = lo + args.peak_w * lp + args.le_w * le
        if offl is not None:
            loss = loss + args.off_w * lo_off
        opt.zero_grad(set_to_none=True); loss.backward()
        # 裁剪要带上 offset 头: 它不在 model.parameters() 里 (单独一个模块), 之前一直没被裁
        torch.nn.utils.clip_grad_norm_(_pp, 1.0)
        opt.step()
        with torch.no_grad():
            pr = logit[:, :po2.shape[1]] >= 0.0
            tg = F.max_pool1d((po2 > 0.5).float().unsqueeze(1), 5, 1, 2).squeeze(1) > 0.5  # +-2 frame tol
            tp = float((pr & tg).sum()); fp = float((pr & ~tg).sum()); fn = float((~pr & tg).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
            if valid.any():
                a1 = (cls[valid].argmax(-1) == lb[valid]).float().mean().item()
                a5 = (cls[valid].topk(5, dim=-1).indices == lb[valid][:, None]).any(-1).float().mean().item()
                if tir_nbr is None:
                    s1 = a1
                else:
                    _tv = lb[valid]; _top = cls[valid].argmax(-1)
                    _nb = tir_nbr[_tv][:, :args.tir_max]
                    _ok = tir_val[_tv][:, :args.tir_max] >= args.tir_conf
                    s1 = float(((_top == _tv) | ((_top[:, None] == _nb) & _ok).any(1)).float().mean())
            else:
                a1 = a5 = s1 = 0.0
        acc["loss"] += lo.item(); acc["pk"] += float(lp.detach()); acc["le"] += le.item()
        if offl is not None:
            acc["off"] += float(lo_off.detach())
        acc["f1"] += f1; acc["t1"] += a1; acc["t5"] += a5
        acc["s1"] += s1; acc["n"] += 1
        # 最佳判据用整个 summary 窗口的原始均值. acc["last_*"] 只是 b=0.05 的 EMA (等效 ~14 步),
        # 拿它判最佳会在噪声上抽样.
        win["t1"] += a1; win["s1"] += s1; win["f1"] += f1; win["n"] += 1
        ema["n"] += 1
        b = 0.05 if ema["n"] > 20 else 1.0 / ema["n"]
        for kk, vv in (("loss", lo.item()), ("pk", float(lp.detach())), ("le", le.item()),
                       ("f1", f1), ("t1", a1), ("t5", a5), ("s1", s1)):
            ema[kk] = (1 - b) * ema[kk] + b * vv
        acc["last_loss"], acc["last_pk"], acc["last_le"] = ema["loss"], ema["pk"], ema["le"]
        acc["last_f1"], acc["last_t1"], acc["last_t5"], acc["last_s1"] = \
            ema["f1"], ema["t1"], ema["t5"], ema["s1"]

        if step % args.logevery == 0:
            now = time.time(); dt = now - t_win; t_win = now
            inst = args.logevery / max(dt, 1e-6)
            rate = inst if rate == 0 else 0.7 * rate + 0.3 * inst
            frac = step / args.steps
            fill = int(BAR * frac)
            bar = "#" * fill + "." * (BAR - fill)
            el = now - t_start
            eta = (args.steps - step) / max(rate, 1e-6)
            n = max(acc["n"], 1)
            extra = ("  set1 %5.1f%%" % (100 * acc["s1"] / n)) if tir_nbr is not None else ""
            _pk = ("  pk %.3f" % (acc["pk"] / n)) if args.peak_w > 0 else ""
            _off = ("  off %.3f" % (acc["off"] / n)) if off_head is not None else ""
            out("[%s] %5.1f%%  %6d/%-6d | 已用 %-6s 剩余 %-6s | lo %.3f%s%s  le %.3f  F1 %.3f | "
                "top1 %5.1f%%  top5 %5.1f%%%s | %.1f it/s" % (
                    bar, frac * 100, step, args.steps, hms(el), hms(eta),
                    acc["loss"] / n, _pk, _off, acc["le"] / n, acc["f1"] / n,
                    100 * acc["t1"] / n, 100 * acc["t5"] / n, extra, rate))
            if args.pool_mode == "attn":
                out("              有效池化窗长 %.1f 帧 (%.0f ms)" % (acc["pw"] / n, acc["pw"] / n * 20))
            acc["loss"] = acc["pk"] = acc["le"] = acc["off"] = 0.0
            acc["f1"] = acc["t1"] = acc["t5"] = acc["s1"] = 0.0
            acc["pw"] = 0.0; acc["n"] = 0

        if step % args.summary_every == 0:
            now = time.time(); frac = step / args.steps
            eta = (args.steps - step) / max(rate, 1e-6)
            _s = ("  set1 %.1f%%" % (100 * acc["last_s1"])) if tir_nbr is not None else ""
            if best_step:
                _s += "  记录 %.2f%%@%d" % (100 * best_val, best_step)
            out("---------- %.1f%%  %d/%d 步 | 已用 %s  剩余 %s | lo %.3f  pk %.3f  le %.3f  F1 %.3f  top1 %.1f%% (随机 %.2f%% 的 %.0f 倍)%s | %.1f it/s  %.2fGB ----------" % (
                frac * 100, step, args.steps, hms(now - t_start), hms(eta),
                acc["last_loss"], acc["last_pk"], acc["last_le"], acc["last_f1"], 100 * acc["last_t1"], rnd * 100,
                (acc["last_t1"] / rnd) if rnd else 0, _s, rate,
                torch.cuda.max_memory_allocated() / 1e9))
            out()
            # 判据 = 最近 args.summary_every 步的原始均值 (win), 不是瞬时值也不是快 EMA
            if args.best_metric != "none":
                wn = max(win["n"], 1)
                cand_t1, cand_f1 = win["t1"] / wn, win["f1"] / wn
                cand_s1 = win["s1"] / wn
                cur = float({"top1": cand_t1, "set1": cand_s1, "f1": cand_f1}[args.best_metric])
                if cur > best_val:
                    _old = ("%.4f@%d" % (best_val, best_step)) if best_step else "无"
                    best_val, best_step = cur, step
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                "args": vars(args), "offset_head": _off_sd, "K": K, "tag": run_id,
                                "best": {"metric": args.best_metric, "value": cur,
                                         "window_top1": cand_t1, "window_set1": cand_s1,
                                         "window_f1": cand_f1,
                                         "window_steps": int(wn), "step": step}}, best_path)
                    out("       新最佳 %s %.4f (窗口 %d 步: top1 %.4f / set1 %.4f / F1 %.4f)  "
                        "覆盖旧记录 %s -> %s"
                        % (args.best_metric, cur, wn, cand_t1, cand_s1, cand_f1, _old,
                           os.path.basename(best_path)))
            win["t1"] = win["s1"] = win["f1"] = 0.0; win["n"] = 0

        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(args), "offset_head": _off_sd, "K": K, "tag": run_id}, ckpt_path)

        if args.snapshot_every and step % args.snapshot_every == 0:
            sp = os.path.join(args.out, "snap_%s_s%d.pt" % (run_id, step))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(args), "offset_head": _off_sd, "K": K, "tag": run_id,
                        "best": {"ema_top1": float(acc["last_t1"]),
                                 "ema_f1": float(acc["last_f1"])}}, sp)
            out("       快照 -> %s" % os.path.basename(sp))

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps,
                "args": vars(args), "offset_head": _off_sd, "K": K, "tag": run_id}, ckpt_path)
    tot = time.time() - t_start
    out()
    out("完成  %d 步 / %s / 平均 %.1f it/s    loss %.3f  F1 %.3f  top1 %.1f%%  top5 %.1f%%" % (
        args.steps - step0, hms(tot), (args.steps - step0) / max(tot, 1e-6),
        acc["last_loss"], acc["last_f1"], 100 * acc["last_t1"], 100 * acc["last_t5"]))
    out("ckpt -> %s" % ckpt_path)
    if args.best_metric != "none":
        out("best -> %s   (%s %.4f @step %d)" % (best_path, args.best_metric, best_val, best_step))
    out("日志 -> %s" % log_path)
    out("下一步: 任务 'SFX · 6 合成集评测' / 'SFX · 7 全片推理'")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        # 以前这里只往 ML/ckpt/ 写, 而那个目录默认不存在 -> open 抛错又被 except 吞掉,
        # 于是所有崩溃的 traceback 都被静默丢掉 (训练目录是 ckpt_tmpl/, 不是 ckpt/).
        # 现在两个位置都写, 并确保目录存在.
        for _p in (os.path.join(ML, "crash.log"), os.path.join(ML, "ckpt", "crash.log")):
            try:
                os.makedirs(os.path.dirname(_p), exist_ok=True)
                with open(_p, "a", encoding="utf-8") as f:
                    f.write("==== %s ====\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), tb))
            except Exception:
                pass
        print(tb, flush=True)
        raise
