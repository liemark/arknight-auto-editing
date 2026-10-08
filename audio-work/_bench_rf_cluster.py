"""谱特征 + 随机森林，在【512 簇】粒度上正经训一次。

为什么是这个粒度：RF 每个叶节点要存长度=类别数 的分布向量, 内存 ≈ 树数×叶数×类数×8B。
22k 类时 6GB 预算只剩十几片叶子/树 -> 每片覆盖上千类, 结构性做不了实例级;
而 512 簇时每簇有几十个成员, 样本充足, 正是 RF 的甜区（旧 lab 的 label_mode=cluster 就是为此）。

协议（和前面所有基线同口径）：dense 档合成混音、oracle onset、0.24s 窗。
训练样本来自【干净模板 + 增广】，其中增广包含"叠另一条模板"（模拟混音干扰），否则域不匹配。
评测：512 簇上的 top1/top5，并与"PCEN 余弦在同一 512 簇上"对比；另附 11 类粗分类。
产物：out/rf_cluster512.pkl
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

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
OUT = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\out"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402
import synth                                                      # noqa: E402

dev = torch.device("cuda")
SR, WIN, NCLUST, NWIN = core.SR, 24, 512, 60
NFR = WIN * core.HOP                              # 240 ms 采样数
MAXPER, NAUG = 40, 4                              # 每簇最多取 40 个成员, 每个 4 种增广
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
fe_p = core.FrontEnd().to(dev).eval()
lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
bi = json.load(open("bank_index.json", encoding="utf-8"))
K = len(lens)

CAT = [("p_atk", "普攻"), ("p_skill", "技能"), ("p_field", "技能"), ("p_imp", "命中"),
       ("p_aoe", "AOE"), ("e_", "敌人"), ("enmy", "敌人"), ("g_ui", "UI"), ("g_", "UI"),
       ("general", "UI"), ("b_ui", "战斗"), ("btl_snd", "战斗"), ("d_avg", "剧情"),
       ("avg_se", "剧情"), ("a_bat", "环境"), ("v_", "人声"), ("dialog", "对话")]


def cat(n):
    for k, v in CAT:
        if str(n).startswith(k):
            return v
    return "其他"


def feats(S):
    """S: (128, W) log-mel -> 125 维特征。"""
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


def chunk(it, n=1024):
    for i in range(0, len(it), n):
        yield it[i:i + n]


# ------------------------------------------------------- 画廊特征 / 512 簇
gidx = np.array([k for k in range(K) if lens[k] > int(0.15 * SR)])
print("画廊 %d 条" % len(gidx), flush=True)
t0 = time.time()
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
print("画廊特征 %.0fs" % (time.time() - t0), flush=True)

km = MiniBatchKMeans(n_clusters=NCLUST, batch_size=4096, n_init=5, random_state=0)
clu = km.fit_predict(GB)
print("512 簇完成, 簇大小 中位 %d  最小 %d" % (np.median(np.bincount(clu)), np.bincount(clu).min()))
g2c = {int(k): int(c) for k, c in zip(gidx, clu)}          # bank 下标 -> 簇

# ------------------------------------------------------- RF 训练集
rng = np.random.default_rng(0)
Xtr, ytr = [], []
t0 = time.time()
members = [np.where(clu == c)[0] for c in range(NCLUST)]
allc = np.arange(len(gidx))
for c in range(NCLUST):
    m = members[c]
    if len(m) == 0:
        continue
    take = rng.choice(m, min(MAXPER, len(m)), replace=False)
    for p in take:
        base = np.asarray(bank[int(offs[gidx[p]]):int(offs[gidx[p]]) + int(min(lens[gidx[p]], NFR))], np.float32)
        if len(base) < NFR:
            base = np.pad(base, (0, NFR - len(base)))
        for a in range(NAUG):
            z = base.copy()
            if a == 1:                                     # 增益 + 噪声
                z = z * 10 ** (rng.uniform(-8, 8) / 20)
                z = z + rng.standard_normal(len(z)).astype(np.float32) * np.sqrt((z ** 2).mean() + 1e-12) * rng.uniform(0.05, 0.4)
            elif a == 2:                                   # 叠另一条模板（混音干扰）
                q = int(rng.integers(0, len(gidx)))
                w = np.asarray(bank[int(offs[gidx[q]]):int(offs[gidx[q]]) + int(min(lens[gidx[q]], NFR))], np.float32)
                L = min(len(z), len(w))
                z[:L] = z[:L] + w[:L] * 10 ** (rng.uniform(-12, 0) / 20)
            else:                                          # 时移
                z = np.roll(z, int(rng.integers(-40, 41)))
            Xtr.append(z); ytr.append(c)
    if (c + 1) % 128 == 0:
        print("  训练样本 %5d 簇  %.0fs  样本 %d" % (c + 1, time.time() - t0, len(Xtr)), flush=True)
Xtr = np.stack(Xtr); ytr = np.array(ytr)
F = np.zeros((len(Xtr), 125), np.float32)
for i, S in enumerate(chunk(Xtr, 1024)):
    S2 = mel_of(S)
    for j, s in enumerate(S2):
        F[i * 1024 + j] = feats(s)
print("训练特征 %s %.0fs" % (F.shape, time.time() - t0), flush=True)

nleaf_est = max(1, len(F) // 25)
mem_est = 2 * nleaf_est * NCLUST * 8 * 100 / 1e6
print("预计树内存约 %.0f MB（100 棵树, 叶节点 >=25 样本）" % mem_est)
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, n_jobs=-1, random_state=0)
rf.fit(F, ytr)
tfit = time.time() - t0
mem = sum(t.tree_.value.nbytes for t in rf.estimators_) / 1e6
os.makedirs(OUT, exist_ok=True)
pickle.dump({"rf": rf, "km": km, "gidx": gidx, "clu": clu}, open(os.path.join(OUT, "rf_cluster512.pkl"), "wb"))
print("RF 训练完成: %.0fs  树内存 %.0f MB  -> out/rf_cluster512.pkl" % (tfit, mem), flush=True)

# ------------------------------------------------------- 评测：合成混音
ds = synth.SynthDS(T=20.0, length=NWIN, seed=4242 + 404, dense_frac=0.6,
                   min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0,
                   empty_frac=0.0, silence_frac=0.2, min_src=1, max_src=5,
                   ivl_med=1.0, ivl_sig=1.2, once_len=1.0, label_mode="template",
                   onset_shape="gauss", onset_len=5, onset_sigma=2.5, bg_mode="mixed")
XF, XP, yclu, ycat, yrow = [], [], [], [], []
pos = {int(k): i for i, k in enumerate(gidx)}
for i in range(NWIN):
    mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
    v = lb >= 0
    if not v.any():
        continue
    for j in range(int(v.sum())):
        k = int(lb[v][j]); a = int(ev[v][j, 0])
        if a < 0 or k not in pos:
            continue
        s = a * core.HOP
        if s + NFR > len(mix):
            continue
        XF.append(mix[s:s + NFR]); yclu.append(g2c[k]); ycat.append(cat(bi[k].get("name", ""))); yrow.append(pos[k])
yclu = np.array(yclu); ycat = np.array(ycat); yrow = np.array(yrow)
QF = np.zeros((len(XF), 125), np.float32)
QP = np.zeros((len(XF), 128), np.float32)
for i, S in enumerate(chunk(XF, 512)):
    S2 = mel_of(S)
    for j, s in enumerate(S2):
        QF[i * 512 + j] = feats(s)
    QP[i * 512:i * 512 + len(S)] = pcen_of(S)
QP /= (np.linalg.norm(QP, axis=1, keepdims=True) + 1e-12)
print("查询事件 %d 个" % len(yclu))

p = rf.predict_proba(QF)
top1 = float((rf.classes_[np.argmax(p, 1)] == yclu).mean())
top5 = float(np.mean([yclu[i] in rf.classes_[np.argsort(-p[i])[:5]] for i in range(len(p))]))
# 同粒度的 PCEN 余弦：最近邻落在同一簇
cos = QP @ GP.T
nn = np.argmax(cos, axis=1)
nn_top1 = float((clu[nn] == yclu).mean())
catacc = float((rf.classes_[np.argmax(p, 1)][:] == yclu).mean())
print()
print("=" * 78)
print("谱特征 + 随机森林（512 簇粒度, %d 样本训练, %.0fs, %.0f MB）" % (len(F), tfit, mem))
print("=" * 78)
print("  RF      : 簇 top-1 %.1f%%   top-5 %.1f%%" % (100 * top1, 100 * top5))
print("  PCEN余弦: 簇 top-1 %.1f%%（同粒度对照）" % (100 * nn_top1))
print("  随机基线: 1/512 = %.2f%%" % (100 / 512))
print()
print("对照：训练好的 v3d 在 22k 类上的【实例级】oracle top-1 = 91.9%%")
