"""粗筛对比实验：log-mel / MFCC / PCEN 的 patch 余弦 vs 训练好的模型 —— recall@k。

问题：不做 transformer、改用经典特征模板匹配，能行到什么程度？
设定（与 eval.py 的 dense 档同口径）：
  * 画廊 = 全库 K=22326 条干净模板，取【onset 起 24 谱帧(240ms)】的特征均值向量
  * 查询 = 合成混音里【每个真实事件】在同一位置的同一特征
  * 指标 = recall@k（k=1/5/20/50），事件取 oracle onset（只看识别，不掺检测误差）
对照 = 训练好的 best_v3d 在同样 oracle onset 下的 top-1/top-5。
纯 GPU 前端 + numpy 余弦，几分钟。
"""
import json
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
from scipy.fftpack import dct                                     # noqa: E402
from core import FrontEnd, Detector                               # noqa: E402

dev = torch.device("cuda")
WIN = 24                     # 谱帧窗口 = 12 个模型帧 = 240 ms
NS = int(0.6 * core.SR)      # 每条模板取前 0.6 s 算特征
BATCH = 512
NWIN = 60                    # 查询窗口数（dense 档, 每窗 8~12 个事件）
CKPT = "ckpt_tmpl/best_v3d-9月18日.pt"

fe_p = FrontEnd().to(dev).eval()
fe_l = FrontEnd(pcen=False).to(dev).eval()
lens = np.load("bank_lens.npy")
offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)
print("画廊 K=%d  窗口=%d 谱帧(%.0fms)" % (K, WIN, WIN * 10))

# ---------------------------------------------------------------- 画廊特征
t0 = time.time()
G = {k: np.zeros((K, d), np.float32) for k, d in
     (("logmel", 128), ("pcen", 128), ("mfcc20", 20), ("mfcc40", 40))}
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
        mp = fe_p(X)[:, :, :WIN]
        ml = fe_l(X)[:, :, :WIN]
    n = len(xs)
    G["pcen"][i0:i0 + n] = mp.mean(-1).cpu().numpy()
    lm = ml.cpu().numpy()
    G["logmel"][i0:i0 + n] = lm.mean(-1)
    M = dct(lm, type=2, axis=1, norm="ortho")[:, :20]           # (n,20,WIN)
    G["mfcc20"][i0:i0 + n] = M.mean(-1)
    G["mfcc40"][i0:i0 + n] = np.concatenate([M.mean(-1), np.diff(M, axis=-1).mean(-1)], 1)
    if (i0 // BATCH) % 8 == 0:
        print("  画廊 %5d/%d  %.0fs" % (i0 + n, K, time.time() - t0), flush=True)
for k in G:
    v = G[k]
    G[k] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
print("画廊特征完成 %.0fs" % (time.time() - t0))

# ---------------------------------------------------------------- 模型
ck = torch.load(CKPT, map_location=dev, weights_only=False)
ar = ck.get("args", {})
model = Detector(int(ck["K"]), d=ar.get("d", 192), emb=ar.get("emb", 256),
                 nl=ar.get("layers", 4), nh=int(ar.get("nh", 4)),
                 rope=bool(ar.get("rope", True))).to(dev)
model.load_state_dict(ck["model"], strict=False); model.eval()
PF = int(ar.get("pool_frames", 12) or 12)
proto = torch.nn.functional.normalize(model.proto, dim=-1).detach()

# ---------------------------------------------------------------- 查询
ds = synth.SynthDS(T=20.0, length=NWIN, seed=4242 + 404,
                   dense_frac=0.6, min_ev=8, max_ev=12, lo=-40.0, hi=-28.0,
                   bg_lo=-45.0, bg_hi=-45.0, empty_frac=0.0, silence_frac=0.2,
                   min_src=1, max_src=5, ivl_med=1.0, ivl_sig=1.2, once_len=1.0,
                   label_mode="template", onset_shape="gauss", onset_len=5,
                   onset_sigma=2.5, bg_mode="mixed")
Q = {k: [] for k in G}
lab = []
for i in range(NWIN):
    o = ds[i]
    mix, po, ev, lb = o[0], o[1], o[2], o[3]
    v = lb >= 0
    if not v.any():
        continue
    f0 = ev[v][:, 0]
    cl = lb[v]
    X = torch.from_numpy(mix[None]).to(dev)
    with torch.no_grad():
        mp = fe_p(X)[0]; ml = fe_l(X)[0]
        logit, emb = model(fe_p(X))
    Mnp = dct(ml.cpu().numpy(), type=2, axis=0, norm="ortho")[:20]   # (20, T)
    for j in range(len(cl)):
        a = int(f0[j])
        if a < 0 or a + WIN > mp.shape[1]:
            continue
        Q["pcen"].append(mp[:, a:a + WIN].mean(-1).cpu().numpy())
        Q["logmel"].append(ml[:, a:a + WIN].mean(-1).cpu().numpy())
        seg = Mnp[:, a:a + WIN]
        Q["mfcc20"].append(seg.mean(-1))
        Q["mfcc40"].append(np.concatenate([seg.mean(-1), np.diff(seg, axis=-1).mean(-1)]))
        st = a // 2
        e = emb[0, st:min(st + PF, emb.shape[1])]
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-9)
        Q.setdefault("model", []).append(
            torch.argsort(-(e.mean(0) @ proto.t())).cpu().numpy()[:50])
        lab.append(int(cl[j]))
lab = np.array(lab)
print("查询事件 %d 个" % len(lab))

# ---------------------------------------------------------------- recall@k
KS = [1, 5, 20, 50]
print()
print("%-12s %8s %8s %8s %8s" % ("特征", "R@1", "R@5", "R@20", "R@50"))
for name in ("pcen", "logmel", "mfcc20", "mfcc40"):
    q = np.stack(Q[name])
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    order = np.argsort(-(q @ G[name].T), axis=1)
    row = [float((order[:, :k] == lab[:, None]).any(1).mean()) for k in KS]
    print("%-12s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, *[100 * r for r in row]))
# 模型：直接用它的 top-50 排名
order = np.stack(Q["model"])
row = [float((order[:, :k] == lab[:, None]).any(1).mean()) for k in KS]
print("%-12s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("模型(v3d)", *[100 * r for r in row]))
