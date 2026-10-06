"""谱特征 + 随机森林：到底能到什么粒度？

口径与前面几个基线完全一致（dense 档、oracle onset、0.6s 窗）：
  * 特征：40 个 log-mel 带均值 + 20 维 MFCC 均值 + 20 维 MFCC 一阶差分 + 谱质心/带宽/滚降/平坦度/ZCR 的均值与标准差
  * 训练：从库里抽 N 类, 每类 3 个增广(增益/噪声/时移)
  * 测试：合成混音里的真实事件
产出两个数：
  (1) RF 当【10 类粗分类器】的准确率（普攻/技能/命中/敌人/UI/剧情/环境...）
  (2) 用 RF 预测的类别去【限制候选范围】后, 再用 PCEN 余弦做实例检索的 R@1/R@20
      —— 对比不限制时的 11.4% / 22.9%
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
import synth                                                      # noqa: E402

dev = torch.device("cuda")
SR = core.SR
NCAP, NAUG, NWIN = 8000, 3, 60
WIN = 24                                   # 240ms 谱帧
fe_l = core.FrontEnd(pcen=False).to(dev).eval()
fe_p = core.FrontEnd().to(dev).eval()

CAT = [("p_atk", "普攻"), ("p_skill", "技能"), ("p_field", "技能"), ("p_imp", "命中"),
       ("p_aoe", "AOE"), ("e_", "敌人"), ("enmy", "敌人"), ("g_ui", "UI"), ("g_", "UI"),
       ("general", "UI"), ("b_ui", "战斗"), ("btl_snd", "战斗"), ("d_avg", "剧情"),
       ("avg_se", "剧情"), ("a_bat", "环境"), ("v_", "人声"), ("dialog", "对话")]


def cat(name):
    for k, v in CAT:
        if str(name).startswith(k):
            return v
    return "其他"


def feats(S):
    """S: (128, W) log-mel -> 特征向量。"""
    band = S.mean(1)                                   # 40 维: 取 128 带里等距 40 个
    idx = np.linspace(0, S.shape[0] - 1, 40).astype(int)
    band = S[idx].mean(1)
    M = dct(S, type=2, axis=0, norm="ortho")[:20]
    mf = M.mean(1)
    dmf = np.diff(M, axis=1).mean(1)
    e = S.mean(0); e = e - e.min() + 1e-6
    cent = (np.arange(len(e)) * e).sum() / e.sum()
    bw = np.sqrt((((np.arange(len(e)) - cent) ** 2) * e).sum() / e.sum())
    cs = np.cumsum(e) / e.sum()
    roll = float(np.searchsorted(cs, 0.85))
    flat = float(np.exp(np.log(e + 1e-9).mean()) / (e.mean() + 1e-9))
    zcr = float(np.abs(np.diff(S.mean(0))).mean())
    stat = np.array([cent, bw, roll, flat, zcr])
    return np.concatenate([band, mf, dmf, stat, np.abs(np.diff(S, axis=1)).mean(1)[idx]])


lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
import json                                                        # noqa: E402
bi = json.load(open("bank_index.json", encoding="utf-8"))
K = len(lens)

# ---------------------------------------------------------------- 画廊特征(PCEN 余弦用)
gidx = [k for k in range(K) if lens[k] > int(0.15 * SR)]
print("画廊 %d 条 " % len(gidx), flush=True)
t0 = time.time()
GP = np.zeros((len(gidx), 128), np.float32)
GC = np.array([cat(bi[k].get("name", "")) for k in gidx])
GB = np.zeros((len(gidx), 128), np.float32)
for i0 in range(0, len(gidx), 1024):
    xs = []
    for k in gidx[i0:i0 + 1024]:
        L = int(min(int(lens[k]), WIN * core.HOP))
        x = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
        if len(x) < WIN * core.HOP:
            x = np.pad(x, (0, WIN * core.HOP - len(x)))
        xs.append(x)
    X = torch.from_numpy(np.stack(xs)).to(dev)
    with torch.no_grad():
        GP[i0:i0 + len(xs)] = fe_p(X)[:, :, :WIN].mean(-1).cpu().numpy()
        GB[i0:i0 + len(xs)] = fe_l(X)[:, :, :WIN].mean(-1).cpu().numpy()
GP /= (np.linalg.norm(GP, axis=1, keepdims=True) + 1e-12)
print("  画廊 PCEN/logmel 完成 %.0fs" % (time.time() - t0), flush=True)

# ---------------------------------------------------------------- RF 训练集
rng = np.random.default_rng(0)
pick = rng.choice(len(gidx), NCAP, replace=False)
Xr, yc, Xq = [], [], []
t0 = time.time()
for n, p in enumerate(pick):
    k = gidx[p]
    L = int(min(int(lens[k]), WIN * core.HOP))
    w = np.asarray(bank[int(offs[k]):int(offs[k]) + L], np.float32)
    if len(w) < WIN * core.HOP:
        w = np.pad(w, (0, WIN * core.HOP - len(w)))
    for a in range(NAUG):
        y = w.copy()
        if a == 1:
            y = y * 10 ** (rng.uniform(-8, 8) / 20)
            y = y + rng.standard_normal(len(y)).astype(np.float32) * np.sqrt((y ** 2).mean() + 1e-12) * rng.uniform(0.05, 0.5)
        if a == 2:
            s = int(rng.integers(-40, 41)); y = np.roll(y, s)
        with torch.no_grad():
            S = fe_l(torch.from_numpy(y)[None].to(dev))[0, :, :WIN].cpu().numpy()
        Xr.append(feats(S)); yc.append(cat(bi[k].get("name", "")))
    if (n + 1) % 2000 == 0:
        print("  训练特征 %5d/%d  %.0fs" % (n + 1, NCAP, time.time() - t0), flush=True)
Xr = np.stack(Xr); yc = np.array(yc)
print("RF 训练集 %s  类别分布 %s" % (Xr.shape, dict(zip(*np.unique(yc, return_counts=True)))))

rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=4, n_jobs=-1, random_state=0)
rf.fit(Xr, yc)
print("RF 训练完成")

# ---------------------------------------------------------------- 查询：合成混音
ds = synth.SynthDS(T=20.0, length=NWIN, seed=4242 + 404, dense_frac=0.6,
                   min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0,
                   empty_frac=0.0, silence_frac=0.2, min_src=1, max_src=5,
                   ivl_med=1.0, ivl_sig=1.2, once_len=1.0, label_mode="template",
                   onset_shape="gauss", onset_len=5, onset_sigma=2.5, bg_mode="mixed")
pos = {int(k): i for i, k in enumerate(gidx)}          # bank 下标 -> 画廊行号
Qf, Qp, lab, labc = [], [], [], []
for i in range(NWIN):
    mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
    v = lb >= 0
    if not v.any():
        continue
    f0 = ev[v][:, 0]; cl = lb[v]
    for j in range(len(cl)):
        a = int(f0[j])
        if a < 0:
            continue
        s = a * core.HOP
        if s + WIN * core.HOP > len(mix) or int(cl[j]) not in pos:
            continue
        seg = mix[s:s + WIN * core.HOP]
        with torch.no_grad():
            Sl = fe_l(torch.from_numpy(seg)[None].to(dev))[0, :, :WIN].cpu().numpy()
            Sp = fe_p(torch.from_numpy(seg)[None].to(dev))[0, :, :WIN].mean(-1).cpu().numpy()
        Qf.append(feats(Sl)); Qp.append(Sp)
        lab.append(pos[int(cl[j])]); labc.append(cat(bi[int(cl[j])].get("name", "")))
Qf = np.stack(Qf); Qp = np.stack(Qp); lab = np.array(lab); labc = np.array(labc)
print("查询事件 %d 个" % len(lab))

# ---------------------------------------------------------------- 结果
pred = rf.predict(Qf)
acc = float((pred == labc).mean())
print()
print("=" * 80)
print("(1) RF 当【粗分类器】（%d 类）: 准确率 %.1f%%" % (len(set(yc)), 100 * acc))
print("=" * 80)
# 每个事件取 RF 的 top3 类别
proba = rf.predict_proba(Qf)
classes = rf.classes_
top3 = classes[np.argsort(-proba, axis=1)[:, :3]]
in_top3 = np.array([labc[i] in top3[i] for i in range(len(labc))])
print("  真值类别落在 RF top-3 里的比例: %.1f%%" % (100 * in_top3.mean()))

cos = Qp @ GP.T
KS = [1, 5, 20]
print()
print("(2) 实例检索（PCEN 余弦）在不同候选范围下：")
print("%-38s %8s %8s %8s" % ("候选范围", "R@1", "R@5", "R@20"))
order = np.argsort(-cos, axis=1)[:, :max(KS)]
row = [float((order[:, :k] == lab[:, None]).any(1).mean()) for k in KS]
print("%-38s %7.1f%% %7.1f%% %7.1f%%" % ("全库 %d 条（不路由）" % len(gidx), *[100 * r for r in row]))
for name, maskf in (("只用 RF 的 top-1 类别", lambda i: top3[i, :1]),
                    ("只用 RF 的 top-3 类别", lambda i: top3[i, :3]),
                    ("用真值类别（上限）", lambda i: np.array([labc[i]]))):
    for k in KS:
        hit = 0; tot = 0
        for i in range(len(lab)):
            m = np.isin(GC, maskf(i))
            if not m.any():
                continue
            sub = np.where(m)[0]
            r = sub[np.argsort(-cos[i, sub])[:k]]
            tot += 1
            hit += int(lab[i] in r)
        if k == KS[0]:
            r1, r5, r20 = hit / max(tot, 1), 0, 0
        elif k == 5:
            r5 = hit / max(tot, 1)
        else:
            r20 = hit / max(tot, 1)
    print("%-38s %7.1f%% %7.1f%% %7.1f%%" % (name + " (%d 条)" % int(np.isin(GC, top3[0, :3]).sum()), 100 * r1, 100 * r5, 100 * r20))
print()
print("对照：训练好的 v3d 在同类口径下 oracle R@1 91.9% / R@50 98.1%；CED 冻结特征 R@1 4.1%")
