"""经典模板匹配的【更强】版本：不做时间平均，直接拿 240ms 的频谱 patch 当模板,
并允许 ±2 谱帧位移搜索（这才是教科书式的"特征模板匹配"）。

对照上一版（均值池化余弦）与训练好的模型。
"""
import os
import sys
import time

import numpy as np
import torch

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402
import synth                                                      # noqa: E402
from core import FrontEnd                                         # noqa: E402

dev = torch.device("cuda")
WIN = 24                    # 240 ms 比较窗
PAD = 2                     # ±2 谱帧 = ±20 ms 位移搜索
WIDE = WIN + 2 * PAD        # 每条存 28 帧
NS = int(0.6 * core.SR)
BATCH = 512
NWIN = 60
KS = [1, 5, 20, 50]

fe_p = FrontEnd().to(dev).eval()
fe_l = FrontEnd(pcen=False).to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)

t0 = time.time()
G = {"pcen": np.zeros((K, 128, WIDE), np.float32),
     "logmel": np.zeros((K, 128, WIDE), np.float32)}
for i0 in range(0, K, BATCH):
    xs = []
    for j in range(i0, min(i0 + BATCH, K)):
        L = int(min(int(lens[j]), NS))
        x = np.asarray(bank[int(offs[j]):int(offs[j]) + L], np.float32)
        if len(x) < NS:
            x = np.pad(x, (0, NS - len(x)))
        xs.append(x)
    X = torch.from_numpy(np.stack(xs)).to(dev)
    with torch.no_grad():
        mp = fe_p(X)[:, :, :WIDE].cpu().numpy()
        ml = fe_l(X)[:, :, :WIDE].cpu().numpy()
    G["pcen"][i0:i0 + len(xs)] = mp
    G["logmel"][i0:i0 + len(xs)] = ml
print("画廊 patch 完成 %.0fs（%d × 128 × %d）" % (time.time() - t0, K, WIDE))

ds = synth.SynthDS(T=20.0, length=NWIN, seed=4242 + 404, dense_frac=0.6,
                   min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0,
                   empty_frac=0.0, silence_frac=0.2, min_src=1, max_src=5,
                   ivl_med=1.0, ivl_sig=1.2, once_len=1.0, label_mode="template",
                   onset_shape="gauss", onset_len=5, onset_sigma=2.5, bg_mode="mixed")
Q = {"pcen": [], "logmel": []}
lab = []
for i in range(NWIN):
    mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
    v = lb >= 0
    if not v.any():
        continue
    f0 = ev[v][:, 0]; cl = lb[v]
    X = torch.from_numpy(mix[None]).to(dev)
    with torch.no_grad():
        mp = fe_p(X)[0]; ml = fe_l(X)[0]
    for j in range(len(cl)):
        a = int(f0[j])
        if a - PAD < 0 or a + WIN + PAD > mp.shape[1]:
            continue
        Q["pcen"].append(mp[:, a - PAD:a - PAD + WIDE].cpu().numpy())
        Q["logmel"].append(ml[:, a - PAD:a - PAD + WIDE].cpu().numpy())
        lab.append(int(cl[j]))
lab = np.array(lab)
print("查询事件 %d 个" % len(lab))


def rank_scores(name):
    """对全库算 max-over-shift 的 patch 余弦 -> 每查询一个全库分数向量。"""
    out = np.empty((len(lab), K), np.float32)
    CH = 4096
    for qi, q in enumerate(Q[name]):
        qc = q[:, PAD:PAD + WIN].reshape(-1).astype(np.float32)
        qc = qc / (np.linalg.norm(qc) + 1e-12)
        s = np.empty(K, np.float32)
        for c0 in range(0, K, CH):
            c1 = min(c0 + CH, K)
            blk = G[name][c0:c1]                                   # (n,128,WIDE)
            best = None
            for k in range(0, 2 * PAD + 1):
                g = blk[:, :, k:k + WIN].reshape(c1 - c0, -1)
                g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
                sc = g @ qc
                best = sc if best is None else np.maximum(best, sc)
            s[c0:c1] = best
        out[qi] = s
    return out


print()
print("%-26s %8s %8s %8s %8s" % ("方法", "R@1", "R@5", "R@20", "R@50"))
print("%-26s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("模型 v3d（oracle onset）", 91.9, 95.8, 97.6, 98.1))
for name in ("pcen", "logmel"):
    for tag, shift in (("patch 余弦（无位移搜索）", 1), ("patch 余弦（±20ms 位移搜索）", 2 * PAD + 1)):
        if shift == 1:
            # 只算 k=PAD 一个位置
            out = np.empty((len(lab), K), np.float32)
            CH = 4096
            for qi, q in enumerate(Q[name]):
                qc = q[:, PAD:PAD + WIN].reshape(-1).astype(np.float32)
                qc = qc / (np.linalg.norm(qc) + 1e-12)
                s = np.empty(K, np.float32)
                for c0 in range(0, K, CH):
                    c1 = min(c0 + CH, K)
                    g = G[name][c0:c1, :, PAD:PAD + WIN].reshape(c1 - c0, -1)
                    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
                    s[c0:c1] = g @ qc
                out[qi] = s
        else:
            print("  计算 %s %s ..." % (name, tag), flush=True)
            out = rank_scores(name)
        order = np.argsort(-out, axis=1)[:, :max(KS)]
        row = [float((order[:, :k] == lab[:, None]).any(1).mean()) for k in KS]
        print("%-26s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name + " " + tag, *[100 * r for r in row]))
