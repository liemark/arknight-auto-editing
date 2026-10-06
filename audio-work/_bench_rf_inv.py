"""把特征改成【不变量】后再上随机森林 —— 上一轮的结论是"该动的是特征不是分类器"。

上一轮 RF 只有 5.9%，而不训练的 PCEN 余弦有 28.1%，根因是树在【绝对能量】上做轴对齐切分：
同一音效每次播放增益不同、还叠了别的音 -> 样本直接跳到别的分支。特征重要性全平(0.013~0.016)
也印证了"没有可用结构"。

不变特征（三件事）：
  ① 逐带去时间均值  S -= mean_t(S)        -> 消掉通道/EQ 倾斜
  ② 逐帧去跨带均值  S -= mean_f(S)        -> 消掉整体增益（等价于丢掉 MFCC 的 c0）
  ③ 帧内归一化谱上的形状统计（质心/带宽/滚降/平坦度/ZCR）-> 本身就是尺度不变量
特征 = 40 相对带能量 + 19 MFCC(c1..c19) 均值 + 19 ΔMFCC + 40 谱通量 + 10 形状统计
对照：原始特征 RF 5.9% / PCEN 余弦 28.1% / 训练好的 v3d 91.9%
数据与划分沿用上一轮（窗口种子 777、划分种子 0），保证测试集完全一致；音频片段会缓存复用。
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
SR, WIN, NCLUST, NWIN = core.SR, 24, 512, 800
NFR = WIN * core.HOP
CACHE = os.path.join(OUT, "rf_patches_800.npz")
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
fe_p = core.FrontEnd().to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 特征
def mel_of(xs):
    with torch.no_grad():
        return fe_l(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].cpu().numpy()


def pcen_of(xs):
    with torch.no_grad():
        return fe_p(torch.from_numpy(np.stack(xs)).to(dev))[:, :, :WIN].mean(-1).cpu().numpy()


def feats_inv(S):
    """S: (128, W) log-mel -> 不变特征。"""
    Sn = S - S.mean(axis=1, keepdims=True)          # ① 逐带去时间均值
    Sf = Sn - Sn.mean(axis=0, keepdims=True)        # ② 逐帧去跨带均值
    idx = np.linspace(0, S.shape[0] - 1, 40).astype(int)
    band = Sf[idx].mean(1)
    M = dct(Sf, type=2, axis=0, norm="ortho")[1:20]  # 丢掉 c0（=整体电平）
    mm = M.mean(1); dm = np.diff(M, axis=1).mean(1)
    flux = np.abs(np.diff(Sf, axis=1)).mean(1)[idx]
    # ③ 帧内归一化谱的形状统计（尺度不变）
    E = np.exp(S - S.max())                          # 稳定化
    P = E / (E.sum(0, keepdims=True) + 1e-12)
    f = np.arange(S.shape[0])
    cent = (P * f[:, None]).sum(0)
    bw = np.sqrt(((f[:, None] - cent) ** 2 * P).sum(0))
    cs = np.cumsum(P, axis=0)
    roll = np.array([np.searchsorted(cs[:, i], 0.85) for i in range(P.shape[1])], float)
    flat = np.exp(np.log(P + 1e-12).mean(0)) / (P.mean(0) + 1e-12)
    zcr = np.abs(np.diff(Sf, axis=0)).mean(0)
    shp = np.array([cent.mean(), cent.std(), bw.mean(), bw.std(), roll.mean(),
                    roll.std(), flat.mean(), flat.std(), zcr.mean(), zcr.std()])
    return np.concatenate([band, mm, dm, flux, shp]).astype(np.float32)


# ---------------------------------------------------------------- 数据（带缓存）
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=False)
    X, Y, W = d["X"], d["Y"], d["W"]
    print("载入缓存 %s" % CACHE, flush=True)
else:
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
            if a < 0:
                continue
            s = a * core.HOP
            if s + NFR > len(mix):
                continue
            X.append(mix[s:s + NFR]); Y.append(k); W.append(i)
        if (i + 1) % 200 == 0:
            print("  生成 %3d/%d 窗  %.0fs  事件 %d" % (i + 1, NWIN, time.time() - t0, len(X)), flush=True)
    X = np.stack(X); Y = np.array(Y); W = np.array(W)
    np.savez(CACHE, X=X, Y=Y, W=W)
    print("缓存 -> %s" % CACHE, flush=True)

# ---------------------------------------------------------------- 512 簇（与上一轮同构）
gidx = np.array([k for k in range(K) if lens[k] > int(0.15 * SR)])
GB = np.zeros((len(gidx), 128), np.float32); GP = np.zeros((len(gidx), 128), np.float32)
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
        GB[i0 + j] = feats_inv(s)
    GP[i0:i0 + len(xs)] = pcen_of(xs)
GP /= (np.linalg.norm(GP, axis=1, keepdims=True) + 1e-12)
km = MiniBatchKMeans(n_clusters=NCLUST, batch_size=4096, n_init=5, random_state=0)
clu = km.fit_predict(GB)
g2c = {int(k): int(c) for k, c in zip(gidx, clu)}
keep = np.array([int(y) in g2c for y in Y])
X, Y, W = X[keep], np.array([g2c[int(y)] for y in Y[keep]]), W[keep]
print("样本 %d（簇 %d, 大小中位 %d）" % (len(Y), NCLUST, np.median(np.bincount(clu))), flush=True)

# 与上一轮完全相同的按窗口划分
u = np.unique(W); rng = np.random.default_rng(0); rng.shuffle(u)
n1, n2 = int(0.7 * len(u)), int(0.85 * len(u))
tr = np.isin(W, u[:n1]); va = np.isin(W, u[n1:n2]); te = np.isin(W, u[n2:])
print("划分: 训练 %d / 验证 %d / 测试 %d" % (tr.sum(), va.sum(), te.sum()), flush=True)

t0 = time.time()
Fi = np.zeros((len(X), 128), np.float32)          # 40 相对带能量 + 19 MFCC + 19 ΔMFCC + 40 通量 + 10 形状
for i0 in range(0, len(X), 512):
    S = mel_of(X[i0:i0 + 512])
    for j, s in enumerate(S):
        Fi[i0 + j] = feats_inv(s)
P = np.zeros((len(X), 128), np.float32)
for i0 in range(0, len(X), 512):
    P[i0:i0 + 512] = pcen_of(X[i0:i0 + 512])
P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
print("不变特征 %s  %.0fs" % (Fi.shape, time.time() - t0), flush=True)

# ---------------------------------------------------------------- RF（小网格）
Xv = np.vstack([Fi[tr], Fi[va]]); Yv = np.concatenate([Y[tr], Y[va]])
ps = PredefinedSplit(np.concatenate([np.full(tr.sum(), -1), np.zeros(va.sum(), int)]))
grid = {"n_estimators": [100, 300], "min_samples_leaf": [3, 5]}
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(max_features="sqrt", n_jobs=-1, random_state=0),
                  grid, scoring="accuracy", cv=ps, n_jobs=1)
gs.fit(Xv, Yv)
best = gs.best_estimator_
pred = best.predict(Fi[te]); proba = best.predict_proba(Fi[te])
a1 = accuracy_score(Y[te], pred)
a5 = float(np.mean([Y[te][i] in best.classes_[np.argsort(-proba[i])[:5]] for i in range(len(pred))]))
print()
print("=" * 84)
print("不变特征 + 随机森林（%d 组网格, %.0fs）最佳 %s" % (len(grid["n_estimators"]) * len(grid["min_samples_leaf"]), time.time() - t0, gs.best_params_))
print("  测试集: top-1 %.1f%%   top-5 %.1f%%" % (100 * a1, 100 * a5))
# 同特征、只用余弦最近邻（分离"特征变好"和"分类器变好"）
Ftr = Fi[tr] / (np.linalg.norm(Fi[tr], axis=1, keepdims=True) + 1e-12)
Fte = Fi[te] / (np.linalg.norm(Fi[te], axis=1, keepdims=True) + 1e-12)
cent = {}
for c in np.unique(Y[tr]):
    cent[c] = Ftr[Y[tr] == c].mean(0)
cc = np.array(sorted(cent)); CM = np.stack([cent[c] for c in cc])
CM /= (np.linalg.norm(CM, axis=1, keepdims=True) + 1e-12)
nnc = cc[np.argmax(Fte @ CM.T, axis=1)]
print("  同特征 + 余弦最近邻（不训练）: %.1f%%" % (100 * float((nnc == Y[te]).mean())))
print("  同测试集 PCEN 余弦（对照）  : 28.1%%")
print("=" * 84)
imp = best.feature_importances_
names = ([f"band{i}" for i in range(40)] + [f"mfcc{i}" for i in range(1, 20)] +
         [f"dmfcc{i}" for i in range(1, 20)] + [f"flux{i}" for i in range(40)] +
         ["cent_m", "cent_s", "bw_m", "bw_s", "roll_m", "roll_s", "flat_m", "flat_s", "zcr_m", "zcr_s"])
o = np.argsort(-imp)[:12]
print("特征重要性 top12: " + "  ".join("%s=%.3f" % (names[i], imp[i]) for i in o))
print("（均分 = %.4f；上一轮原始特征 top12 全在 0.013~0.016 = 平的）" % (1.0 / len(imp)))
pickle.dump({"rf": best, "km": km, "gidx": gidx, "clu": clu, "params": gs.best_params_},
            open(os.path.join(OUT, "rf_cluster512_inv.pkl"), "wb"))
