"""音频指纹（spectral-peak-pair landmark hashing, Shazam/Wang 2003 那一类）在本题上的可行性。

为什么值得试：波形 NCC 对失真极度敏感（重采样/编码/增益/微小时钟差都能把它打到 0.1），
而指纹的设计目标恰恰是"在有损、重采样、增益、背景、轻微变速下认出同一段音频"，并自带 offset 投票。

流程：
  1) 离线索引：库里每条模板 -> log-mel -> 谱峰 -> 锚点-目标对 -> hash(f1,f2,dt) -> (clip, t_anchor)
  2) 查询：录音同样提 hash -> 查索引 -> 在 (clip, Δt) 上投票 -> 票数最高的就是 (哪条, 什么偏移)
  3) 取 top 候选，用采样级互相关细化偏移 -> 报 NCC（能不能真抵消看这个）

先跑合成验证（真值已知：能否认回 clip、offset 误差多少），再跑真实录音 v82。
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402

dev = torch.device("cuda")
SR = core.SR
NF = core.N_MELS
DT_MAX, DF_MAX, FAN = 60, 64, 5          # 目标最多往后 600ms, 频差 <=64 bin, 每锚点最多 5 个目标
MAXSEC = 3.0                             # 每条模板最多索引 3 s
fe = core.FrontEnd(pcen=False).to(dev).eval()


def logmel(x44):
    with torch.no_grad():
        return fe(torch.from_numpy(np.asarray(x44, np.float32))[None].to(dev))[0].cpu().numpy()


def peaks(S, pct=88.0, nf=9, nt=9):
    """log-mel 上的局部极大值 -> (f, t)。"""
    mf = maximum_filter(S, size=(nf, nt))
    thr = np.percentile(S, pct)
    m = (S >= mf) & (S > thr)
    f, t = np.where(m)
    return f.astype(np.int32), t.astype(np.int32)


def hashes(f, t):
    """锚点-目标对 -> (hash, t_anchor)。"""
    hs, ts = [], []
    n = len(t)
    for i in range(n):
        k = 0
        for j in range(i + 1, n):
            dt = int(t[j] - t[i])
            if dt > DT_MAX:
                break
            df = int(f[j]) - int(f[i])
            if abs(df) > DF_MAX or df == 0:
                continue
            hs.append(((int(f[i]) * NF + int(f[j])) * (DT_MAX + 1) + dt))
            ts.append(int(t[i]))
            k += 1
            if k >= FAN:
                break
    return np.asarray(hs, np.int64), np.asarray(ts, np.int32)


# ------------------------------------------------------------------ 建索引
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)
print("建索引：K=%d（每条最多 %.1fs）" % (K, MAXSEC))
t0 = time.time()
H, C, T = [], [], []
for k in range(K):
    L = int(min(int(lens[k]), int(MAXSEC * SR)))
    if L < int(0.05 * SR):
        continue
    x = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
    f, t = peaks(logmel(x))
    if len(t) < 2:
        continue
    h, ta = hashes(f, t)
    H.append(h); C.append(np.full(len(h), k, np.int32)); T.append(ta)
    if (k + 1) % 5000 == 0:
        print("  %5d/%d  %.0fs  累计 hash %d" % (k + 1, K, time.time() - t0, sum(len(v) for v in H)), flush=True)
H = np.concatenate(H); C = np.concatenate(C); T = np.concatenate(T)
o = np.argsort(H, kind="stable")
H, C, T = H[o], C[o], T[o]
print("索引完成：%d 个 hash，用时 %.0fs" % (len(H), time.time() - t0))


def query(x44, topn=8, ignore_clip=None):
    """录音 -> [(clip, offset帧, 票数, 该候选的 hash 命中数)]"""
    f, t = peaks(logmel(x44))
    hq, tq = hashes(f, t)
    votes = {}
    for h, t_a in zip(hq, tq):
        i = np.searchsorted(H, h, "left"); j = np.searchsorted(H, h, "right")
        if i == j:
            continue
        for k in range(i, j):
            c = int(C[k])
            if ignore_clip is not None and c == ignore_clip:
                continue
            key = (c, int(t_a) - int(T[k]))
            votes[key] = votes.get(key, 0) + 1
    out = sorted(votes.items(), key=lambda kv: -kv[1])[:topn]
    return [(c, dt, v) for (c, dt), v in out]


def ncc_best(x, seg, srch=600, step=4):
    """采样级对齐（±srch 采样）后的最大 |NCC|。"""
    xc = x - x.mean(); nx = np.linalg.norm(xc)
    best = (0.0, 0)
    for lag in range(-srch, srch + 1, step):
        a = len(seg) // 2 + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        w = seg[a:a + len(x)]; wc = w - w.mean()
        r = float(xc @ wc) / (nx * np.linalg.norm(wc) + 1e-12)
        if abs(r) > abs(best[0]):
            best = (r, lag)
    return best


# ------------------------------------------------------------- A) 合成验证
ds_cls = None
print()
print("=" * 84)
print("A) 合成验证：把 8 条已知模板按已知时间混进噪声底, 看指纹能不能认回来")
print("=" * 84)
rng = np.random.default_rng(0)
sel = rng.choice(np.where(lens > int(0.3 * SR))[0], 8, replace=False)
Tmix = 60.0
mix = rng.standard_normal(int(Tmix * SR)).astype(np.float32) * 0.002
truth = []
for i, k in enumerate(sel):
    L = int(min(int(lens[k]), int(MAXSEC * SR)))
    w = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
    t_ev = 2.0 + i * 6.0
    a = int(t_ev * SR)
    mix[a:a + L] += w * 10 ** (rng.uniform(-12, -4) / 20)
    truth.append((int(k), t_ev))
res = query(mix, topn=200)
hit = 0
for k, t_ev in truth:
    # 找这条 clip 的最高票候选
    cand = [(dt, v) for c, dt, v in res if c == k]
    if cand:
        dt, v = max(cand, key=lambda z: z[1])
        off_err_ms = ((dt + 0.0) * 0.01 - t_ev) * 1000
        print("  clip %6d  真值 t=%6.2fs  指纹票 %4d  估计 t=%6.2fs  误差 %+6.1f ms"
              % (k, t_ev, v, (dt * 0.01), off_err_ms))
        hit += 1
    else:
        print("  clip %6d  真值 t=%6.2fs  ✗ 未出现在 top200" % (k, t_ev))
print("  合成验证命中 %d/8" % hit)

# ------------------------------------------------------------- B) 真实录音
print()
print("=" * 84)
print("B) 真实录音：库里有没有能指纹命中的片段？")
print("=" * 84)
for name in ("v82_mono", "nl_mono"):
    path = os.path.join(os.path.dirname(TRAIN), "data", "atoms", name + ".wav")
    import alab
    sys.path.insert(0, os.path.join(os.path.dirname(TRAIN), "data", "atoms"))
    x, sr = alab.wav_read(path, mono=True)
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if int(sr) != SR:
        x = alab.bandpass(x, int(sr), 30.0, min(int(sr), SR) * 0.475)
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x).astype(np.float32)
    print("%s: %.1fs" % (name, len(x) / SR))
    t0 = time.time()
    res = query(x, topn=12)
    print("  查询用时 %.0fs；票数最高的 12 个 (clip, 时间, 票数)：" % (time.time() - t0))
    bi = json.load(open("bank_index.json", encoding="utf-8"))
    for c, dt, v in res:
        print("    clip %6d  t=%8.2fs  票 %5d   %s" % (c, dt * 0.01, v, bi[c].get("name", "")[:28]))
