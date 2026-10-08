"""事件级定位评测 —— 你最终要的是"每个音效都在哪里", 而帧级 F1 测不出这件事。

合成数据里 onset 真值精确到 10ms。对 (阈值, 不应期, 容差) 扫描, 算事件级 precision/recall/F1,
并给出【结构性上限】: 若某个 onset 的近邻比不应期还近, 那么无论模型多好都不可能把它们分开
(alab.find_peaks 会抑制 refr 帧以内的第二个峰)。

于是能直接回答: 现在漏掉的那些, 是模型不行, 还是流水线的时间分辨率不允许?
"""
import os, sys, json, argparse
import numpy as np, torch
ML = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(ML)                      # 工作根目录（sfx/ long/ 与 alab.py 都在这层）
sys.path.insert(0, ML); sys.path.insert(0, D)
from core import FrontEnd, Detector, build_detector
from synth import SynthDS
import alab

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--windows", type=int, default=500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
ap.add_argument("--refrs", type=int, nargs="+", default=[1, 2, 3, 4, 6])
ap.add_argument("--T", type=float, default=0.0, help="窗口秒数. 0=自动用 checkpoint 里记的训练值")
ap.add_argument("--tol", type=float, default=0.05, help="主表用的匹配容差(秒)")
ap.add_argument("--tols", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15],
                help="容差扫描(秒)")
ap.add_argument("--proms", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6],
                help="峰显著度(prominence)扫描; 0=不启用")
ap.add_argument("--bg-mode", default="mixed",
                choices=["mixed", "noise", "silence", "recording", "long"],
                help="long = 用解包的真实长音(BGM+环境+战斗语音)当背景, 测录音里的虚警")
ap.add_argument("--bg-db", type=float, default=None,
                help="long 档背景电平 (RMS dBFS); 不给就用 SynthDS 默认")
ap.add_argument("--out", default="")
a = ap.parse_args()

dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ar = ck.get("args", {}); K = int(ck["K"]); LM = str(ar.get("label_mode", "cluster"))
if a.T <= 0:                       # 自动跟随 checkpoint 的窗口长度, 避免"训练 10s 却在 4s 上评测"
    a.T = float(ar.get("T", 4.0))
model = Detector(K, d=ar.get("d", 192), emb=ar.get("emb", 128), nl=ar.get("layers", 3), nh=int(ar.get("nh", 4)), rope=bool(ar.get("rope", str(ar.get("arch", "v1")) == "v2"))).to(dev)
model.load_state_dict(ck["model"], strict=False); model.eval()
fe = FrontEnd().to(dev).eval()
print("ckpt step=%s  K=%d  d=%d nl=%d  label_mode=%s  T=%.1fs" % (
    ck.get("step"), K, ar.get("d"), ar.get("layers"), LM, a.T), flush=True)

# 数据配置一律和 train.py 对齐, 否则测的就不是训练分布. 窗口长度用 --T.
ds = SynthDS(T=a.T, length=1200, seed=5, max_ev=128, dense_frac=0.6, silence_frac=0.2,
             empty_frac=0.2, min_src=1, max_src=5, ivl_med=1.0, ivl_sig=1.2, label_mode=LM,
             bg_mode=a.bg_mode,
             **({"bg_lo": a.bg_db, "bg_hi": a.bg_db} if a.bg_db is not None else {}))

GT = []          # 每个窗口的 onset 时刻(秒)
SPAN = []        # 每个窗口的事件区间 [start,end](秒)
PR = []          # 每个窗口的 sigmoid 曲线
batch = []
with torch.no_grad():
    for i in range(a.windows):
        batch.append(ds[i])
        if len(batch) < a.bs and i < a.windows - 1:
            continue
        mix = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
        lg, _ = model(fe(mix))
        p = torch.sigmoid(lg.float()).cpu().numpy()
        for bi, b in enumerate(batch):
            ev, lb = b[2], b[3]
            e = ev[lb >= 0].astype(np.float64) * 0.01
            if e.size:                                   # span 起点为负 = onset 在窗口外, 夹到 0
                np.clip(e[:, 0], 0.0, None, out=e[:, 0])
            SPAN.append(e)
            # 真值 onset 只保留落在窗口内的: 被左边界切掉的事件根本没有 onset 可匹配
            GT.append(np.sort((ev[lb >= 0, 0] * 0.01)[ev[lb >= 0, 0] >= 0]).astype(np.float64))
            PR.append(p[bi].astype(np.float64))
        batch = []
print("windows %d   真值 onset 共 %d" % (len(GT), sum(len(g) for g in GT)), flush=True)

FR = 0.02


def match(gt, pr, tol):
    used = np.zeros(len(pr), dtype=bool); hit = 0
    for t in gt:
        if len(pr) == 0:
            break
        d = np.abs(pr - t).copy(); d[used] = 1e9
        j = int(d.argmin())
        if d[j] <= tol:
            used[j] = True; hit += 1
    return hit, len(pr) - hit, len(gt) - hit


# ---- 结构性上限: 近邻距离 >= (refr+1) 帧的 onset 才可能被单独检出 ----
nn = []
for g in GT:
    if len(g) > 1:
        d = np.diff(g)
        nn.append(np.minimum(np.r_[d, d[-1]], np.r_[d[0], d]))
nn = np.concatenate(nn) if nn else np.zeros(0)
print("\n每个 onset 到最近邻的距离: p10 %.3f p25 %.3f p50 %.3f" % (
    np.percentile(nn, 10), np.percentile(nn, 25), np.median(nn)))


def max_sep(g, gap):
    """贪心选出最多的、两两间隔 >= gap 的 onset 个数 = 该不应期下召回率的硬上限"""
    if len(g) == 0:
        return 0
    c = 1; last = g[0]
    for t in g[1:]:
        if t - last >= gap:
            c += 1; last = t
    return c


tot_gt = sum(len(g) for g in GT)
print("%-8s %-14s %s" % ("不应期", "最小可分辨间隔", "召回率硬上限(同时可匹配的 onset 占比)"))
ceil = {}
for r in a.refrs:
    c = sum(max_sep(g, (r + 1) * FR) for g in GT) / max(tot_gt, 1)
    ceil[r] = c
    print("%-8s %-14s %.1f%%" % ("%d帧" % r, "%.0f ms" % ((r + 1) * FR * 1000), 100 * c))

print("\n=== 事件级 (容差 %.0f ms) ===" % (a.tol * 1000))
print("%-6s %-8s %-10s %-10s %-10s %-12s" % ("refr", "thr", "P", "R", "F1", "整窗全中"))
best = None
res = {}
for r in a.refrs:
    for thr in a.thrs:
        tp = fp = fn = 0; allw = 0; nw = 0
        for g, p in zip(GT, PR):
            if len(g) == 0:
                continue
            pk = alab.find_peaks(p, thr, r)[0].astype(np.float64) * FR
            h, f, m = match(g, pk, a.tol)
            tp += h; fp += f; fn += m
            allw += int(h == len(g)); nw += 1
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        res["%d_%.1f" % (r, thr)] = {"P": P, "R": R, "F1": F, "all_win": allw / max(nw, 1),
                                     "ceiling": ceil[r]}
        print("%-6d %-8.1f %-10s %-10s %-10s %-12s" % (
            r, thr, "%.1f%%" % (100 * P), "%.1f%%" % (100 * R), "%.1f%%" % (100 * F),
            "%.1f%%" % (100 * allw / max(nw, 1))))
        if best is None or F > best[0]:
            best = (F, r, thr, P, R, allw / max(nw, 1))
print("\n本表最优: refr=%d thr=%.1f  P %.1f%%  R %.1f%%  F1 %.1f%%  整窗全中 %.1f%%" % (
    best[1], best[2], 100 * best[3], 100 * best[4], 100 * best[0], 100 * best[5]))
print("同一 refr 的结构性上限 %.1f%%  ->  距上限还差 %.1f 个点 (召回)" % (
    100 * ceil[best[1]], 100 * (ceil[best[1]] - best[4])))

# ---- 第二把刀: 峰显著度过滤. find_peaks 只看"值够不够高", 不看这个峰比周围高多少;
#      概率曲线上一个鼓包里的次级起伏会被当成两个事件 -> 假阳性. 这里用 scipy 的 prominence. ----
try:
    from scipy.signal import find_peaks as spfind
    print("\n=== 加峰显著度过滤 (scipy.signal.find_peaks, distance=refr+1) ===")
    print("%-6s %-6s %-6s %-10s %-10s %-10s %-12s" % ("refr", "thr", "prom", "P", "R", "F1", "整窗全中"))
    for r in [1, 2, 3]:
        for thr in [0.5, 0.7, 0.9]:
            for pr in a.proms:
                tp = fp = fn = 0; allw = 0; nw = 0
                for g, p in zip(GT, PR):
                    if len(g) == 0:
                        continue
                    pk, _ = spfind(p, height=thr, distance=r + 1,
                                   prominence=(pr if pr > 0 else None))
                    h, f, m = match(g, pk.astype(np.float64) * FR, a.tol)
                    tp += h; fp += f; fn += m
                    allw += int(h == len(g)); nw += 1
                P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
                F = 2 * P * R / max(P + R, 1e-9)
                mk = "%d_%.1f_%.2f" % (r, thr, pr)
                res[mk] = {"P": P, "R": R, "F1": F, "all_win": allw / max(nw, 1), "ceiling": ceil[r]}
                print("%-6d %-6.1f %-6.2f %-10s %-10s %-10s %-12s" % (
                    r, thr, pr, "%.1f%%" % (100 * P), "%.1f%%" % (100 * R),
                    "%.1f%%" % (100 * F), "%.1f%%" % (100 * allw / max(nw, 1))))
    b2 = max([(v["F1"], k) for k, v in res.items() if k.count("_") == 2])
    print("显著度过滤后全局最优: %s  F1 %.1f%%  P %.1f%%  R %.1f%%" % (
        b2[1], 100 * res[b2[1]]["F1"], 100 * res[b2[1]]["P"], 100 * res[b2[1]]["R"]))
except ImportError:
    print("\n(没有 scipy, 跳过显著度扫描)")

# ---- 假峰是"无中生有"还是"真事件但时间放歪了"? ----
print("\n=== 假峰的来源 (refr=2 thr=0.9): 到最近真值 onset 的距离 ===")
d_all = []
for g, p in zip(GT, PR):
    if len(g) == 0:
        continue
    pk = alab.find_peaks(p, 0.9, 2)[0].astype(np.float64) * FR
    if len(pk) == 0:
        continue
    used = np.zeros(len(pk), dtype=bool)
    for t in g:
        d = np.abs(pk - t).copy(); d[used] = 1e9
        j = int(d.argmin())
        if d[j] <= a.tol:
            used[j] = True
    for q in pk[~used]:
        d_all.append(float(np.abs(g - q).min()))
d_all = np.array(d_all) if d_all else np.zeros(0)
print("假峰共 %d 个" % len(d_all))
for x in (0.06, 0.10, 0.15, 0.20, 0.30):
    print("   距最近真值 onset < %.2fs : %5.1f%%" % (x, 100 * (d_all < x).mean()))
if len(d_all):
    print("   中位距离 %.3fs   p25 %.3f   p75 %.3f" % (
        np.median(d_all), np.percentile(d_all, 25), np.percentile(d_all, 75)))
    fpdist = {"n": int(len(d_all)), "p50": float(np.median(d_all)),
              "within": {str(x): float((d_all < x).mean()) for x in (0.06, 0.10, 0.15, 0.20, 0.30)}}
# ---- 容差扫描: 那些"假峰"是不是只是时间放歪了? ----
print("\n=== 容差扫描 (refr=2): P / R / F1 ===")
print("%-6s %s" % ("thr", "  ".join("%-16s" % ("tol=%.0fms" % (t * 1000)) for t in a.tols)))
for thr in [0.3, 0.5, 0.7, 0.9]:
    row = []
    for tl in a.tols:
        tp = fp = fn = 0
        for g, p in zip(GT, PR):
            if len(g) == 0:
                continue
            pk = alab.find_peaks(p, thr, 2)[0].astype(np.float64) * FR
            h, f, m = match(g, pk, tl)
            tp += h; fp += f; fn += m
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        row.append("%.0f/%.0f/%.0f" % (100 * P, 100 * R, 100 * 2 * P * R / max(P + R, 1e-9)))
    print("%-6.1f %s" % (thr, "  ".join("%-16s" % x for x in row)))

# ---- 假峰分类: 重复峰 / 错位峰(附近有漏检) / 无中生有 / 该处的真值密度 ----
print("\n=== 假峰分类 (refr=2, 容差 %.0fms): 所谓「多峰」到底是哪种 ===" % (a.tol * 1000))
print("%-6s %-8s %-12s %-16s %-14s %-12s %s" % (
    "thr", "假峰数", "重复(<=tol)", "错位(附近有漏检)", "无中生有", "FP处真值密度", "TP处真值密度"))
cls_rep = {}
for thr in [0.3, 0.5, 0.7, 0.9]:
    dup = mis = hal = nfp = 0
    dens_fp = []; dens_tp = []
    for g, p in zip(GT, PR):
        if len(g) == 0:
            continue
        pk = alab.find_peaks(p, thr, 2)[0].astype(np.float64) * FR
        if len(pk) == 0:
            continue
        used = np.zeros(len(pk), dtype=bool); gtused = np.zeros(len(g), dtype=bool)
        for gi, t in enumerate(g):
            d = np.abs(pk - t).copy(); d[used] = 1e9
            j = int(d.argmin())
            if d[j] <= a.tol:
                used[j] = True; gtused[gi] = True
        for j in np.flatnonzero(used):
            dens_tp.append((np.abs(g - pk[j]) <= 0.15).sum())
        for j in np.flatnonzero(~used):
            q = pk[j]; dd = np.abs(g - q); nfp += 1
            dens_fp.append((dd <= 0.15).sum())
            if dd.min() <= a.tol:
                dup += 1
            elif len(dd) and ((dd <= 0.10) & (~gtused)).any():
                mis += 1
            else:
                hal += 1
    f = lambda v: ("%.1f" % np.mean(v)) if len(v) else "-"
    print("%-6.1f %-8d %-12s %-16s %-14s %-12s %s" % (
        thr, nfp, "%.1f%%" % (100 * dup / max(nfp, 1)), "%.1f%%" % (100 * mis / max(nfp, 1)),
        "%.1f%%" % (100 * hal / max(nfp, 1)), f(dens_fp), f(dens_tp)))
    cls_rep[str(thr)] = {"n_fp": nfp, "dup": dup / max(nfp, 1), "misplaced": mis / max(nfp, 1),
                         "hallucinated": hal / max(nfp, 1),
                         "dens_fp": float(np.mean(dens_fp)) if dens_fp else 0.0,
                         "dens_tp": float(np.mean(dens_tp)) if dens_tp else 0.0}

# ---- 假峰是不是落在【正在响的那个事件】里面? (mid-event 误触发) ----
print("\n=== 假峰的时间位置 (refr=2, 容差 %.0fms) ===" % (a.tol * 1000))
print("%-6s %-8s %-14s %-16s %-18s %s" % (
    "thr", "假峰数", "落在事件区间内", "不在内,起点后<0.2s", "不在内,起点后0.2-0.5s", "更远/前面没事件"))
mid_rep = {}
for thr in [0.3, 0.5, 0.7, 0.9]:
    n = n_in = n_a = n_b = n_c = 0
    for g, sp, p in zip(GT, SPAN, PR):
        if len(g) == 0:
            continue
        pk = alab.find_peaks(p, thr, 2)[0].astype(np.float64) * FR
        if len(pk) == 0:
            continue
        used = np.zeros(len(pk), dtype=bool)
        for t in g:
            d = np.abs(pk - t).copy(); d[used] = 1e9
            j = int(d.argmin())
            if d[j] <= a.tol:
                used[j] = True
        for q in pk[~used]:
            n += 1
            if sp.size and ((q >= sp[:, 0]) & (q < sp[:, 1])).any():
                n_in += 1
            else:
                prev = g[g <= q]
                dt = (q - prev.max()) if len(prev) else 9.0
                if dt < 0.2:
                    n_a += 1
                elif dt < 0.5:
                    n_b += 1
                else:
                    n_c += 1
    pc = lambda v: "%.1f%%" % (100 * v / max(n, 1))
    print("%-6.1f %-8d %-14s %-16s %-18s %s" % (thr, n, pc(n_in), pc(n_a), pc(n_b), pc(n_c)))
    mid_rep[str(thr)] = {"n_fp": n, "inside_event": n_in / max(n, 1), "after<0.2": n_a / max(n, 1),
                         "after0.2-0.5": n_b / max(n, 1), "far": n_c / max(n, 1)}

# ---- 对照: "落在事件区间内"这个判据本身有多廉价? 加上 相对位置 才说明问题 ----
print("\n=== 对照与相对位置 (refr=2, 容差 %.0fms) ===" % (a.tol * 1000))
print("  「区间内」占比: 随机时刻 / TP峰 / FP峰;  以及 FP 落在所属事件内的相对位置")
rngc = np.random.default_rng(3)
for thr in [0.3, 0.7, 0.9]:
    n = n_in = 0; nt = nt_in = 0
    rnd_in = []; rel = []
    for g, sp, p in zip(GT, SPAN, PR):
        if len(g) == 0 or sp.size == 0:
            continue
        ts = rngc.uniform(0, a.T, 200)
        rnd_in.append(float(((ts[:, None] >= sp[None, :, 0]) & (ts[:, None] < sp[None, :, 1])).any(1).mean()))
        pk = alab.find_peaks(p, thr, 2)[0].astype(np.float64) * FR
        if len(pk) == 0:
            continue
        used = np.zeros(len(pk), dtype=bool)
        for t in g:
            d = np.abs(pk - t).copy(); d[used] = 1e9
            j = int(d.argmin())
            if d[j] <= a.tol:
                used[j] = True
        for j, q in enumerate(pk):
            hit = (q >= sp[:, 0]) & (q < sp[:, 1])
            if used[j]:
                nt += 1; nt_in += int(hit.any())
                continue
            n += 1
            if hit.any():
                n_in += 1
                k = int(np.flatnonzero(hit)[0])
                L = max(sp[k, 1] - sp[k, 0], 1e-6)
                rel.append((q - sp[k, 0]) / L)
    rel = np.array(rel) if rel else np.zeros(0)
    print("thr %.1f:  随机 %.1f%%   TP %.1f%%   FP %.1f%%   |  FP相对位置 p25 %.2f p50 %.2f p75 %.2f   相对<0.15 占 %.1f%%"
          % (thr, 100 * np.mean(rnd_in), 100 * nt_in / max(nt, 1), 100 * n_in / max(n, 1),
             np.percentile(rel, 25) if len(rel) else -1, np.median(rel) if len(rel) else -1,
             np.percentile(rel, 75) if len(rel) else -1, 100 * (rel < 0.15).mean() if len(rel) else -1))

if a.out:
    json.dump({"ckpt": a.ckpt, "step": ck.get("step"), "windows": len(GT),
               "n_gt": int(sum(len(g) for g in GT)), "ceiling": {str(k): v for k, v in ceil.items()},
               "tol": a.tol, "grid": res, "fp_dist": locals().get("fpdist", {}),
               "fp_class": cls_rep, "fp_mid": mid_rep},
              open(a.out, "w", encoding="utf-8"), indent=1)
    print("-> %s" % a.out)
