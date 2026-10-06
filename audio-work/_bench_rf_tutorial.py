"""按菜鸟教程的随机森林流程重做一遍：同分布划分 + 超参搜索 + 特征重要性。

上一轮的三个问题（所以 1.7% 不公平）：
  1. 训练集是"干净模板 + 增广", 测试集是混音片段 —— 分布不一致（教程专门有一章讲训练/测试划分）
  2. 完全没搜超参（n_estimators / max_depth / max_features / min_samples_split / min_samples_leaf）
  3. 没看特征重要性

这次：训练/验证/测试**全部从同一个混音分布里按窗口划分**（无泄漏），网格搜索, 报特征重要性。
口径：dense 档（一间 8~12 个事件）、oracle onset、0.24s 窗、512 簇标签。
对照：上一轮 RF 1.7% / PCEN 余弦 26.6% / 训练好的 v3d 实例级 91.9%。
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy.fftpack import dct
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
OUT = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\out"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402
import synth                                                      # noqa: E402

dev = torch.device("cuda")
SR, WIN, NCLUST = core.SR, 24, 512
NFR = WIN * core.HOP
NWIN = 800                                   # 生成的混音窗数, 按窗划分 train/val/test
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
fe_p = core.FrontEnd().to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
bi = json.load(open("bank_index.json", encoding="utf-8"))
K = len(lens)


def feats(S):
    """S: (128, W) log-mel -> 125 维特征（40 带 + 20 MFCC + 20 ΔMFCC + 5 统计 + 40 谱通量）。"""
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


def mel_of(xs):
    with torch.no_grad():
        return fe_l(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].cpu().numpy()


def pcen_of(xs):
    with torch.no_grad():
        return fe_p(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].mean(-1).cpu().numpy()


# ------------------------------------------------ 512 簇（标签空间）
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
        GB[i0 + j] = feats(s)
    GP[i0:i0 + len(xs)] = pcen_of(xs)
GP /= (np.linalg.norm(GP, axis=1, keepdims=True) + 1e-12)
km = MiniBatchKMeans(n_clusters=NCLUST, batch_size=4096, n_init=5, random_state=0)
clu = km.fit_predict(GB)
g2c = {int(k): int(c) for k, c in zip(gidx, clu)}
print("512 簇: 大小中位 %d  最小 %d" % (np.median(np.bincount(clu)), np.bincount(clu).min()), flush=True)

# ------------------------------------------------ 数据：全部来自同一个混音分布
ds = synth.SynthDS(T=20.0, length=NWIN, seed=777, dense_frac=0.6,
                   min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0,
                   empty_frac=0.0, silence_frac=0.2, min_src=1, max_src=5,
                   ivl_med=1.0, ivl_sig=1.2, once_len=1.0, label_mode="template",
                   onset_shape="gauss", onset_len=5, onset_sigma=2.5, bg_mode="mixed")
X, Y, W = [], [], []
t0 = time.time()
for i in range(NWIN):
    mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
    v = lb >= 0
    if not v.any():
        continue
    for j in range(int(v.sum())):
        k = int(lb[v][j]); a = int(ev[v][j, 0])
        if a < 0 or k not in g2c:
            continue
        s = a * core.HOP
        if s + NFR > len(mix):
            continue
        X.append(mix[s:s + NFR]); Y.append(g2c[k]); W.append(i)
    if (i + 1) % 200 == 0:
        print("  生成 %3d/%d 窗  %.0fs  事件 %d" % (i + 1, NWIN, time.time() - t0, len(X)), flush=True)
X = np.stack(X); Y = np.array(Y); W = np.array(W)
F = np.zeros((len(X), 125), np.float32)
for i0 in range(0, len(X), 512):
    S = mel_of(X[i0:i0 + 512])
    for j, s in enumerate(S):
        F[i0 + j] = feats(s)
P = np.zeros((len(X), 128), np.float32)
for i0 in range(0, len(X), 512):
    P[i0:i0 + 512] = pcen_of(X[i0:i0 + 512])
P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
print("样本 %s  用时 %.0fs" % (F.shape, time.time() - t0), flush=True)

# 按【窗口】划分，同一窗的事件不会跨集合（防泄漏）
u = np.unique(W)
rng = np.random.default_rng(0); rng.shuffle(u)
n1, n2 = int(0.7 * len(u)), int(0.85 * len(u))
tr = np.isin(W, u[:n1]); va = np.isin(W, u[n1:n2]); te = np.isin(W, u[n2:])
print("划分: 训练 %d / 验证 %d / 测试 %d（按窗口）" % (tr.sum(), va.sum(), te.sum()), flush=True)

# ------------------------------------------------ (b) 未调参 baseline（同样的同分布数据）
base = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, n_jobs=-1, random_state=0)
base.fit(F[tr], Y[tr])
acc_base = accuracy_score(Y[te], base.predict(F[te]))
print("(b) 未调参、但同分布训练: 测试准确率 %.1f%%" % (100 * acc_base), flush=True)

# ------------------------------------------------ (c) 按教程做网格搜索
grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "max_features": ["sqrt", 0.3],
    "min_samples_leaf": [5, 10],
}
Xv = np.vstack([F[tr], F[va]]); Yv = np.concatenate([Y[tr], Y[va]])
ps = PredefinedSplit(np.concatenate([np.full(tr.sum(), -1), np.zeros(va.sum(), int)]))
gs = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), grid,
                  scoring="accuracy", cv=ps, n_jobs=1, verbose=0)
t0 = time.time()
gs.fit(Xv, Yv)
print("(c) 网格搜索 %d 组 x 1 折验证, %.0fs" % (len(gs.cv_results_["params"]), time.time() - t0), flush=True)
print("    最佳参数: %s   验证准确率 %.1f%%" % (gs.best_params_, 100 * gs.best_score_), flush=True)
best = gs.best_estimator_
pred = best.predict(F[te])
acc = accuracy_score(Y[te], pred)
proba = best.predict_proba(F[te])
top5 = float(np.mean([Y[te][i] in best.classes_[np.argsort(-proba[i])[:5]] for i in range(len(pred))]))
print("    测试集: top-1 %.1f%%   top-5 %.1f%%" % (100 * acc, 100 * top5), flush=True)

# ------------------------------------------------ (d) 同测试集上的 PCEN 余弦（同粒度）
nn = np.argmax(P[te] @ GP.T, axis=1)
acc_cos = float((clu[nn] == Y[te]).mean())
print("(d) 同测试集 PCEN 余弦（不训练、同 512 簇粒度）: %.1f%%" % (100 * acc_cos), flush=True)
print("    随机基线 1/512 = %.2f%%" % (100 / 512), flush=True)

# ------------------------------------------------ 特征重要性
names = ([f"mel{i}" for i in range(40)] + [f"mfcc{i}" for i in range(20)] +
         [f"dmfcc{i}" for i in range(20)] + ["centroid", "bandwidth", "rolloff", "flatness", "zcr"] +
         [f"flux{i}" for i in range(40)])
imp = best.feature_importances_
order = np.argsort(-imp)[:12]
print("    特征重要性 top12: " + "  ".join("%s=%.3f" % (names[i], imp[i]) for i in order), flush=True)
os.makedirs(OUT, exist_ok=True)
pickle.dump({"rf": best, "km": km, "gidx": gidx, "clu": clu, "params": gs.best_params_},
            open(os.path.join(OUT, "rf_cluster512_tuned.pkl"), "wb"))
print("模型 -> out/rf_cluster512_tuned.pkl", flush=True)
