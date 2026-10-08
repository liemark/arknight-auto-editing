"""在 RF 前面接一个 PCEN 层：PCEN 特征 + 随机森林。

动机：前面所有 RF 变体都输给"不训练的 PCEN 余弦"（28.1%），而 PCEN 正好是我手工不变化失败的东西 ——
对数压缩 + 逐通道自适应增益 + 逐窗标准化，既保留频谱包络又消掉整体增益。
那直接让 RF 吃 PCEN 特征，看树能不能在好表示上赢过余弦。

口径严格对齐（保证可比）：
  * 数据用缓存 out/rf_patches_800.npz（7786 个事件, 800 个混音窗）
  * 512 簇标签空间用与之前完全相同的确定性 k-means（raw 特征 + seed 0）
  * 同一套按窗口划分（seed 0）-> 测试集与前面几次完全一致（1207 事件）
对照：原始谱特征 RF 5.9% / 不变特征 RF 2.1% / PCEN 余弦 28.1% / 训练好的模型 91.9%
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.fftpack import dct
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
OUT = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\out"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402

dev = torch.device("cuda")
SR, WIN, NCLUST = core.SR, 24, 512
NFR = WIN * core.HOP
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
fe_p = core.FrontEnd().to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)


def feats_raw(S):
    idx = np.linspace(0, S.shape[0] - 1, 40).astype(int)
    M = dct(S, type=2, axis=0, norm="ortho")[:20]
    e = S.mean(1); e = e - e.min() + 1e-6
    cent = (np.arange(len(e)) * e).sum() / e.sum()
    bw = np.sqrt((((np.arange(len(e)) - cent) ** 2) * e).sum() / e.sum())
    cs = np.cumsum(e) / e.sum()
    roll = float(np.searchsorted(cs, 0.85))
    flat = float(np.exp(np.log(e + 1e-9).mean()) / (e.mean() + 1e-9))
    zcr = float(np.abs(np.diff(S.mean(0))).mean())
    return np.concatenate([S[idx].mean(1), M.mean(1), np.diff(M, axis=1).mean(1),
                           [cent, bw, roll, flat, zcr],
                           np.abs(np.diff(S, axis=1)).mean(1)[idx]]).astype(np.float32)


def pcen_of(xs):
    with torch.no_grad():
        return fe_p(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].mean(-1).cpu().numpy()


def mel_of(xs):
    with torch.no_grad():
        return fe_l(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].cpu().numpy()


# ---------------------------------------------- 标签空间（与之前几次完全相同）
gidx = np.array([k for k in range(K) if lens[k] > int(0.15 * SR)])
GB = np.zeros((len(gidx), 125), np.float32)
GP = np.zeros((len(gidx), 128), np.float32)
for i0 in range(0, len(gidx), 1024):
    c = gidx[i0:i0 + 1024]
    xs = []
    for k in c:
        L = int(min(int(lens[k]), NFR))
        x = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
        if len(x) < NFR:
            x = np.pad(x, (0, NFR - len(x)))
        xs.append(x)
    S = mel_of(xs)
    for j, s in enumerate(S):
        GB[i0 + j] = feats_raw(s)
    GP[i0:i0 + len(xs)] = pcen_of(xs)
GP /= (np.linalg.norm(GP, axis=1, keepdims=True) + 1e-12)
km = MiniBatchKMeans(n_clusters=NCLUST, batch_size=4096, n_init=5, random_state=0)
clu = km.fit_predict(GB)
g2c = {int(k): int(c) for k, c in zip(gidx, clu)}

# ---------------------------------------------- 数据 + PCEN 特征
d = np.load(os.path.join(OUT, "rf_patches_800.npz"))
X = d["X"]; Yb = d["Y"]; W = d["W"]
keep = np.array([int(y) in g2c for y in Yb])
X, Y, W = X[keep], np.array([g2c[int(y)] for y in Yb[keep]]), W[keep]
F = np.zeros((len(X), 128), np.float32)
t0 = time.time()
for i0 in range(0, len(X), 512):
    F[i0:i0 + 512] = pcen_of(X[i0:i0 + 512])
print("PCEN 特征 %s  %.0fs" % (F.shape, time.time() - t0), flush=True)

u = np.unique(W); rng = np.random.default_rng(0); rng.shuffle(u)
n1, n2 = int(0.7 * len(u)), int(0.85 * len(u))
tr = np.isin(W, u[:n1]); va = np.isin(W, u[n1:n2]); te = np.isin(W, u[n2:])
print("划分: 训练 %d / 验证 %d / 测试 %d（与之前几次一致）" % (tr.sum(), va.sum(), te.sum()), flush=True)

ps = PredefinedSplit(np.concatenate([np.full(tr.sum(), -1), np.zeros(va.sum(), int)]))


def run(tag, Ftr_all, grid):
    Xv = np.vstack([Ftr_all[tr], Ftr_all[va]]); Yv = np.concatenate([Y[tr], Y[va]])
    gs = GridSearchCV(RandomForestClassifier(max_features="sqrt", n_jobs=-1, random_state=0),
                      grid, scoring="accuracy", cv=ps, n_jobs=1)
    t = time.time(); gs.fit(Xv, Yv)
    best = gs.best_estimator_
    pred = best.predict(Ftr_all[te]); proba = best.predict_proba(Ftr_all[te])
    a1 = accuracy_score(Y[te], pred)
    a5 = float(np.mean([Y[te][i] in best.classes_[np.argsort(-proba[i])[:5]] for i in range(len(pred))]))
    print("  %-34s top-1 %5.1f%%  top-5 %5.1f%%   (%s, %.0fs)" % (
        tag, 100 * a1, 100 * a5, gs.best_params_, time.time() - t), flush=True)
    return best, a1


print()
print("=" * 86)
print("PCEN 层 + 随机森林（同一测试集 %d 事件）" % te.sum())
print("=" * 86)
grid = {"n_estimators": [100, 300], "min_samples_leaf": [3, 5]}
best, a1 = run("PCEN128 + RF", F, grid)
p = PCA(n_components=32, random_state=0).fit(F[tr])
Fp = np.zeros((len(F), 32), np.float32)
Fp[tr] = p.transform(F[tr]); Fp[va] = p.transform(F[va]); Fp[te] = p.transform(F[te])
best2, _ = run("PCEN128 -> PCA32 + RF", Fp, grid)

# 同特征、不训练的余弦（关键对照：树有没有比余弦强）
Ft = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
cent = {c: Ft[tr][Y[tr] == c].mean(0) for c in np.unique(Y[tr])}
cc = np.array(sorted(cent)); CM = np.stack([cent[c] for c in cc])
CM /= (np.linalg.norm(CM, axis=1, keepdims=True) + 1e-12)
cos_cls = cc[np.argmax(Ft[te] @ CM.T, axis=1)]
print("  %-34s top-1 %5.1f%%   （同特征、不训练）" % (
    "PCEN128 + 余弦(簇心)", 100 * float((cos_cls == Y[te]).mean())), flush=True)
nn = np.argmax(Ft[te] @ GP.T, axis=1)
print("  %-34s top-1 %5.1f%%   （同特征、不训练, 全库最近邻）" % (
    "PCEN128 + 余弦(全库)", 100 * float((clu[nn] == Y[te]).mean())), flush=True)
print("=" * 86)
print("对照：原始谱特征 RF 5.9% / 不变特征 RF 2.1% / 训练好的 v3d 实例级 91.9% / 随机 0.20%")
imp = best.feature_importances_
print("PCEN+RF 特征重要性: top1 %.4f  中位 %.4f  均分 %.4f（平的=没结构）"
      % (imp.max(), np.median(imp), 1.0 / len(imp)))
