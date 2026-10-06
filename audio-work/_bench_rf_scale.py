"""随机森林在【多类别】上的可训练性：时间 / 内存 / 分辨率的实测曲线。

关键事实：sklearn 的 RandomForest 每个叶节点存一个长度=类别数 的分布向量 ->
内存 ≈ 树数 × 叶节点数 × 类别数 × 8B。所以类别数一涨, 要么内存爆, 要么被迫把
min_samples_leaf 调大 -> 叶子覆盖几百个类 -> 结构性做不了实例级区分。

做法：每类 16 个增广样本, 50 棵树, 给 1GB 的树内存预算, 反推 min_samples_leaf,
量 (a) 训练耗时 (b) 真实树内存 (c) 每叶平均覆盖多少类 (d) 留出集上的 top1/top5。
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402

dev = torch.device("cuda")
SR, WIN, NAUG, NTREE = core.SR, 24, 16, 50
BUDGET = 1e9                                    # 树内存预算 1GB
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)
cand = np.array([k for k in range(K) if lens[k] > int(0.2 * SR)])
print("可用类 %d" % len(cand))


def mel_batch(xs):
    X = torch.from_numpy(np.stack(xs)).to(dev)
    with torch.no_grad():
        S = fe_l(X)[:, :, :WIN].cpu().numpy()
    return S


def feats(S):
    M = dct(S, type=2, axis=1, norm="ortho")[:, :20]
    e = S.mean(1)
    return np.concatenate([S.mean(2), M.mean(2), np.diff(M, axis=2).mean(2)], 1).astype(np.float32)


def make(C, seed):
    """C 个类 x NAUG 个增广 -> 特征/标签。"""
    rng = np.random.default_rng(seed)
    cls = rng.choice(cand, C, replace=False)
    N = int(WIN * core.HOP)
    X, y = [], []
    for n, k in enumerate(cls):
        L = int(min(int(lens[k]), N))
        w = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
        if len(w) < N:
            w = np.pad(w, (0, N - len(w)))
        for a in range(NAUG):
            z = w.copy()
            if a % 3 == 1:
                z = z * 10 ** (rng.uniform(-8, 8) / 20)
            if a % 3 == 2:
                z = z + rng.standard_normal(len(z)).astype(np.float32) * np.sqrt((z ** 2).mean() + 1e-12) * rng.uniform(0.05, 0.4)
            if a >= 6:
                z = np.roll(z, int(rng.integers(-40, 41)))
            X.append(z); y.append(n)
    X = np.stack(X)
    S = np.concatenate([mel_batch(X[i:i + 1024]) for i in range(0, len(X), 1024)], 0)
    return feats(S), np.array(y)


print()
print("%-8s %8s %10s %10s %10s %12s %8s %8s" % (
    "类别数", "样本数", "min叶", "训练秒", "树内存MB", "每叶覆盖类", "top1", "top5"))
for C in (100, 1000, 5000, 22326):
    t0 = time.time()
    Xtr, ytr = make(C, 1)
    Xte, yte = make(C, 2)
    N = len(Xtr)
    leaf_budget = max(1, int(BUDGET / (8 * NTREE * C)))       # 1GB 预算下允许的叶节点数
    min_leaf = max(2, int(np.ceil(N / leaf_budget)))
    rf = RandomForestClassifier(n_estimators=NTREE, min_samples_leaf=min_leaf,
                                n_jobs=-1, random_state=0)
    t1 = time.time()
    rf.fit(Xtr, ytr)
    t2 = time.time()
    mem = sum(t.tree_.value.nbytes for t in rf.estimators_) / 1e6
    nleaf = sum(int((t.tree_.children_left == -1).sum()) for t in rf.estimators_)
    per_leaf = C / max(nleaf / NTREE, 1)                       # 每叶平均覆盖多少类
    p = rf.predict_proba(Xte[:4000])
    top1 = float((rf.classes_[np.argmax(p, 1)] == yte[:4000]).mean())
    k5 = min(5, p.shape[1])
    top5 = float(np.mean([yte[i] in rf.classes_[np.argsort(-p[i])[:k5]] for i in range(len(p))]))
    print("%-8d %8d %10d %10.1f %10.1f %12.1f %7.1f%% %7.1f%%" % (
        C, N, min_leaf, t2 - t1, mem, per_leaf, 100 * top1, 100 * top5))
print()
print("对照：我们的原型/最近邻方案 —— 每类只存 1 个 256 维向量(22326x256 = 23MB), 实例级 R@1 91.9%")
