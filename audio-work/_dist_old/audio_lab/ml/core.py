"""Feature front-end (log-mel + PCEN, vectorised) and the detector model (SDPA + time stride 2)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 64
TIME_STRIDE = 2


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=30.0, fmax=7600.0):
    def h2m(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def m2h(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    pts = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        r = max(r, c + 1); c = max(c, l + 1)
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / (r - c)
    return torch.from_numpy(fb)


class FrontEnd(nn.Module):
    """waveform (B,N) -> per-clip-normalised PCEN (B, N_MELS, T).  IIR smoothing done by cumsum."""

    def __init__(self, alpha=0.8, delta=10.0, r=0.25, s=0.025, eps=1e-6, pcen=True):
        super().__init__()
        self.register_buffer("fb", mel_filterbank())
        self.register_buffer("win", torch.hann_window(N_FFT))
        self.alpha, self.delta, self.r, self.s, self.eps = alpha, delta, r, s, eps
        self.pcen = pcen

    def forward(self, x):
        S = torch.stft(x, N_FFT, HOP, window=self.win, center=True, return_complex=True)
        M = torch.matmul(self.fb, S.real ** 2 + S.imag ** 2)
        T = M.shape[2]
        a = 1.0 - self.s
        k = torch.arange(T, device=M.device, dtype=torch.float32)
        ak = torch.exp(-k * np.log(a))
        P = torch.cumsum(M * ak.view(1, 1, T), dim=2)
        if not self.pcen:
            out = torch.log(M + self.eps)
        else:
            Ss = self.s * (1.0 / ak).view(1, 1, T) * P
            out = (M / (self.eps + Ss) ** self.alpha + self.delta) ** self.r - self.delta ** self.r
        m = out.mean(dim=(1, 2), keepdim=True)
        sd = out.std(dim=(1, 2), keepdim=True) + 1e-5
        return (out - m) / sd


# ---------------------------------------------------------------- 位置编码
# 2026 年的 TaRoPE 一类工作(时间戳/连续时间位置编码)把 RoPE 从"离散下标"推广到"真实时间"。
# 帧是等间隔的(20ms), 标准 RoPE 即可; 关键性质是【相对位置】:
#   q·k 只依赖 (内容, 帧距), 与绝对位置无关 -> 可以跨窗口长度外推(T=4 训的开始能跑 10s)。
# 没有它时, 打乱帧序输出只会跟着打乱(等于 Set Transformer, 而不是序列模型)。
def rope_tables(T, dh, device, base=10000.0):
    """(T, dh/2) 的 cos/sin 表; dh 必须是偶数。"""
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def rope_apply(x, cos, sin):
    """x: (B,H,T,dh) -> 同形状。GPT-NeoX 的半维配对约定。cos/sin: (T, dh/2)。"""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos[None, None]; s = sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Block(nn.Module):
    def __init__(self, d, nh, drop, ffn=4, rope=False):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn * d, d))
        self.drop = nn.Dropout(drop); self.nh = nh
        self.rope = bool(rope)

    def forward(self, x, rt=None):
        B, T, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nh, D // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope:
            cos, sin = rt if rt is not None else rope_tables(T, D // self.nh, x.device)
            q = rope_apply(q, cos, sin); k = rope_apply(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a))
        return x + self.mlp(self.n2(x))


class Enc(nn.Module):
    def __init__(self, n_mels=N_MELS, d=192, nl=3, nh=4, drop=0.1, rope=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, d, 3, padding=1), nn.GELU(),
        )
        fq = n_mels // 8
        self.proj = nn.Linear(d * fq, d)
        self.blocks = nn.ModuleList([Block(d, nh, drop, rope=rope) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.out_stride = TIME_STRIDE
        self.d = d; self.nh = nh; self.rope = bool(rope)

    def forward(self, mel):
        h = self.stem(mel.unsqueeze(1))          # (B, d, 8, T/2)
        h = h.permute(0, 3, 1, 2).flatten(2)
        h = self.proj(h)
        rt = rope_tables(h.shape[1], self.d // self.nh, mel.device) if self.rope else None
        for b in self.blocks:
            h = b(h, rt)
        return self.norm(h)


class PoolAttn(nn.Module):
    """可学习的池化: 给事件区间内的每一帧打分, softmax 加权求和.

    零初始化最后一层 -> 训练开始时 logits 全为 0, 也就是【均匀平均】(等价旧的"整段均值").
    训练只会让它【偏离】均值: 该压的帧压下, 该重的地方加重.
    所以它天然包含了"固定 240ms 窗"和"整段均值"两个特例, 不需要人去猜窗长.

    有效窗长 = 权重的参与率 1/sum(w^2), 单位是帧. 直接把这个数打出来就知道模型想要多长.
    """
    def __init__(self, emb, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(emb, hid), nn.GELU(), nn.Linear(hid, 1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, e, msk):
        # e: (B,T,D)   msk: (B,G,T) bool -> (B,G,D), (B,G,T)
        s = self.f(e)[..., 0].unsqueeze(1).expand(-1, msk.shape[1], -1)
        w = torch.softmax(s.masked_fill(~msk, -1e9), dim=-1)
        return torch.einsum("bgt,btd->bgd", w, e), w


class OffsetHead(nn.Module):
    """再过一层 Linear 出【事件结束】logit (2026 边界感知那条路线: 显式建模 onset 和 offset).

    用途: onset 头只在"事件开始"处被监督, 因此容易在持续音的中间产生假峰.
    把"事件结束"显式建模之后, 推理端可以用 offset 关掉 onset 打开的事件,
    不必再靠不应期(refr)这种后处理超参.

    刻意【不】塞进 Detector: 那样 Detector 的 state_dict 会多出键, 仓库里十几个按
    strict=True 载入的评测脚本会全部报错. 它跟 pooler 一样作为独立模块单独存进 checkpoint.
    """
    def __init__(self, d):
        super().__init__()
        self.f = nn.Linear(d, 1)

    def forward(self, h):
        return self.f(h).squeeze(-1)


class Detector(nn.Module):
    def __init__(self, K, d=192, emb=128, nl=3, drop=0.1, tau=0.07, nh=4, rope=False):
        super().__init__()
        self.enc = Enc(d=d, nl=nl, drop=drop, nh=nh, rope=rope)
        self.onset = nn.Linear(d, 1)
        self.head = nn.Linear(d, emb)
        self.proto = nn.Parameter(torch.randn(K, emb) * 0.02)
        self.tau = tau
        self.d = d; self.nh = nh; self.rope = bool(rope)

    def forward(self, mel, return_all=False):
        h = self.enc(mel)
        on = self.onset(h).squeeze(-1)
        emb = F.normalize(self.head(h), dim=-1)
        if return_all:
            return {"onset": on, "emb": emb, "h": h}
        return on, emb


def ckpt_rope(ar):
    """这个 checkpoint 是不是带 RoPE. 优先读显式的 rope; 老 checkpoint 没有这个键就看 arch.

    只按 arch 推断时, v2 的 checkpoint 可能被按 rope=False 前向, 结果错误但不报错.
    """
    if "rope" in (ar or {}):
        return bool(ar["rope"])
    return str((ar or {}).get("arch", "v1")) == "v2"


def build_detector(ck, K=None, dev="cpu"):
    """按 checkpoint 里记的 args 建模型并载入(d/emb/layers/nh/rope), strict=False.

    评测/推理脚本都该走这里: 漏掉 rope 的话, v2 的 checkpoint 会用一个【没带位置编码】的
    函数去前向, 结果错误而不是报错.
    """
    ar = ck.get("args", {}) or {}
    k = int(K if K is not None else ck["K"])
    m = Detector(k, d=int(ar.get("d", 192)), emb=int(ar.get("emb", 128)),
                 nl=int(ar.get("layers", 3)), nh=int(ar.get("nh", 4)),
                 rope=ckpt_rope(ar))
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    if miss or unexp:
        print("[core] 载入 checkpoint: 缺 %s / 多 %s (旧架构 checkpoint 属正常)"
              % (list(miss)[:4], list(unexp)[:4]), flush=True)
    return m.to(dev).eval()
