"""标定：NCC 到底能容忍哪种失真？—— 回答"失真了就完全对不上吗"。

A) 拿干净模板人为施加各种失真, 量我的 NCC 度量还剩多少。
B) 对 v82 的真实事件做【精细速率搜索】(0.1% 级) —— 之前的搜索步长是 1%, 抓不到微小速率差。
C) 分块对齐：把 2s 切成 4 块各找一个最佳延迟; 若延迟随块号线性漂移 => 存在速率差(斜率即速率误差),
   若延迟是乱的 => 那一段根本不相关。这个方法在 NCC 被拉低时仍能看出"是速率差还是别的"。
纯 CPU。
"""
import os
import sys

import numpy as np

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from scipy.signal import resample as sp_resample                            # noqa: E402
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav, resample as pkg_resample          # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 48000
SRC = 44100


def ncc_at(x, seg, lag):
    a = lag
    if a < 0 or a + len(x) > len(seg):
        return 0.0
    xc = x - x.mean(); w = seg[a:a + len(x)]; wc = w - w.mean()
    d = np.linalg.norm(xc) * np.linalg.norm(wc)
    return float(xc @ wc) / d if d > 1e-12 else 0.0


def best_ncc(x, seg, c, srch, step=4):
    """在 c 附近 ±srch 采样找最大 |NCC|。"""
    best = 0.0
    for lag in range(c - srch, c + srch + 1, step):
        best = max(best, abs(ncc_at(x, seg, lag)))
    return best


# ----------------------------------------------------------------- A) 标定
rng = np.random.default_rng(0)
cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
lens = np.load(os.path.join(AI, "train", "bank_lens.npy"))
offs = np.load(os.path.join(AI, "train", "bank_offs.npy"))
bank = np.load(os.path.join(AI, "train", "bank_pool.npy"), mmap_mode="r")

cand = [i for i in range(len(lens)) if 0.5 * SRC < lens[i] < 2.0 * SRC]
pick = rng.choice(cand, 4, replace=False)


def load44(i):
    L = int(min(lens[i], 2 * SRC))
    return np.asarray(bank[int(offs[i]):int(offs[i]) + L], np.float64)


def lowpass(x, fc, sr):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f > fc] = 0
    return np.fft.irfft(X, len(x))


print("=" * 92)
print("A) 我的 NCC 度量对已知失真的容忍度（干净模板 vs 失真后的同一条）")
print("=" * 92)
print("%-46s %10s" % ("失真", "NCC"))
res = {}
for i in pick:
    x = load44(i)
    seg = np.concatenate([np.zeros(2000), x, np.zeros(2000)])
    c = 2000
    res.setdefault("0. 干净（对照）", []).append(best_ncc(x, seg, c, 120))
    # 1) 44.1k -> 48k -> 44.1k（视频/引擎重采样）
    y = pkg_resample(np.asarray(x, np.float32), SRC, 48000)
    y = np.asarray(pkg_resample(y, 48000, SRC), np.float64)
    res.setdefault("1. 重采样 44.1k→48k→44.1k", []).append(best_ncc(y, seg, c, 120))
    # 2) AAC 味：低通 15k + 30dB 噪声
    y = lowpass(x, 15000, SRC)
    y = y + rng.standard_normal(len(y)) * np.sqrt((y ** 2).mean()) * 10 ** (-30 / 20)
    res.setdefault("2. 低通15k + 30dB 噪声（有损味）", []).append(best_ncc(y, seg, c, 120))
    # 3) 硬限幅（压到 0.3 峰值）
    p = np.abs(x).max()
    y = np.clip(x, -0.3 * p, 0.3 * p)
    res.setdefault("3. 硬限幅到 0.3 峰值", []).append(best_ncc(y, seg, c, 120))
    # 4) tanh 软限幅（重压缩）
    y = np.tanh(x / (0.3 * p)) * (0.3 * p)
    res.setdefault("4. tanh 软限幅（重压缩）", []).append(best_ncc(y, seg, c, 120))
    # 5) 混响（0.3s 衰减噪声 IR）
    ir = rng.standard_normal(int(0.3 * SRC)) * np.exp(-np.arange(int(0.3 * SRC)) / (0.05 * SRC))
    ir /= np.linalg.norm(ir)
    y = np.convolve(x, ir)[:len(x)]
    res.setdefault("5. 混响 0.3s", []).append(best_ncc(y, seg, c, 120))
    # 6+) 微小速率误差（这才是长窗 NCC 的杀手）
    for rel in (0.0001, 0.0005, 0.001, 0.003, 0.01, 0.03):
        n = int(len(x) / (1 + rel))
        y = np.asarray(sp_resample(x, n), np.float64)
        res.setdefault("6. 速率 +%.2f%%" % (100 * rel), []).append(best_ncc(y, seg, c, 120))
for k, v in res.items():
    print("%-46s %10.3f" % (k, float(np.mean(v))))
print("  （模板 4 条, 2s 窗; 速率误差那一组就是把同一条音效按 1+rel 变速后再对齐）")

# ------------------------------------------------------- B/C) 真实事件
import json                                                                  # noqa: E402
mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", "v82_mono.wav"), mono=True),
                 np.float32)
tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_v82_v3d.json"), encoding="utf-8"))
evs = sorted([e for e in tl if e["cands"][0]["cat"] != "其他"],
             key=lambda e: -e["cands"][0]["gain"])[:3]
print()
print("=" * 92)
print("B/C) v82 真实事件：精细速率搜索 + 分块延迟漂移")
print("=" * 92)
for e in evs:
    c0 = e["cands"][0]
    aid = mp.get(int(c0["class"]))
    x = np.asarray(lib.get48k(int(aid))[:2 * SR], np.float64)
    t0 = int(round(e["t"] * SR))
    pad = int(0.12 * SR)
    seg = mix[max(0, t0 - pad): t0 + len(x) + pad].astype(np.float64)
    cc = t0 - max(0, t0 - pad)
    if len(seg) < len(x) + 2 * pad or cc < 0:
        continue
    # B) 精细速率搜索：粗 0.5% 网格 + 在最优点附近 0.02% 精搜
    r_base = best_ncc(x, seg, cc, 900, 16)
    best = (r_base, 1.0)
    for rel in np.arange(-0.05, 0.0501, 0.005):
        y = np.asarray(sp_resample(x, max(256, int(len(x) / (1 + rel)))), np.float64)
        r = best_ncc(y, seg, cc, 900, 16)
        if r > best[0]:
            best = (r, 1 + rel)
    for rel in best[1] - 1 + np.arange(-0.005, 0.0051, 0.0002):
        y = np.asarray(sp_resample(x, max(256, int(len(x) / (1 + rel)))), np.float64)
        r = best_ncc(y, seg, cc, 900, 16)
        if r > best[0]:
            best = (r, 1 + rel)
    # C) 分块延迟漂移：4 块各找最佳延迟
    K = 4
    cl = len(x) // K
    lags = []
    for k in range(K):
        xk = x[k * cl:(k + 1) * cl]
        cand_lags = []
        a0 = cc + k * cl
        for lag in range(a0 - 900, a0 + 901, 4):
            if lag < 0 or lag + cl > len(seg):
                continue
            cand_lags.append((abs(ncc_at(xk, seg, lag)), lag - a0))
        if cand_lags:
            cand_lags.sort(reverse=True)
            lags.append(cand_lags[0])
        else:
            lags.append((0.0, 0))
    drift = (lags[-1][1] - lags[0][1]) / SR / (len(x) * (K - 1) / K / SR) * 1e6
    print("  t=%7.2fs %-24s  NCC(原速) %.3f  ->  精细速率搜索后 %.3f @%.3f%%"
          % (e["t"], c0["template"][:24], r_base, best[0], 100 * (best[1] - 1)))
    print("        分块最佳延迟(采样, 相对): %s  -> 等效速率误差 %.0f ppm（|r|=%s）"
          % ([f"{l:+d}({r:.2f})" for r, l in lags], drift,
             [f"{r:.2f}" for r, _ in lags]))
