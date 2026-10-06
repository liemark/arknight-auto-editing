
"""At what top-k does the coarse identification reach 99%?

Ranks the true cluster against all 512 prototypes, at (a) the ground-truth onset with the
training-style whole-span pooling, and (b) the actual detected onset with the 30-frame
inference pooling.  Stratified by how many events share the window (density)."""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
ML = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(ML), "data", "atoms")
sys.path.insert(0, ML); sys.path.insert(0, D)
from core import FrontEnd, Detector, build_detector
from synth import SynthDS
import alab

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ML, "ckpt", "_recall_probe.pt"))
ap.add_argument("--windows", type=int, default=900, help="评测多少个窗口")
ap.add_argument("--out", default="", help="把曲线存成 json")
ap.add_argument("--tag", default="")
ap.add_argument("--T", type=float, default=0.0, help="窗口秒数. 0=自动用 checkpoint 里记的训练值")
ap.add_argument("--pool-frames", type=int, default=0,
                help="detected 路径的识别池化窗(输出帧/20ms). 0=跟随 checkpoint 的训练值 (推荐); "
                     "这里原来是硬编码 30 帧(600ms), 与 train/infer 的 12 帧(240ms) 不一致")
ap.add_argument("--label-mode", default="", choices=["", "cluster", "template"],
                help="留空 = 自动用 checkpoint 里记的训练值")
a = ap.parse_args()

dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ar = ck.get("args", {}); K = int(ck["K"])
POOL_FRAMES = max(1, int(a.pool_frames) if a.pool_frames > 0
                  else int(ar.get("pool_frames", 12) or 12))
print("识别池化 = %d 帧 (%.0f ms)%s" % (
    POOL_FRAMES, POOL_FRAMES * 20,
    "  <- 跟随 checkpoint" if a.pool_frames <= 0 else "  <- 命令行覆盖"), flush=True)
model = Detector(K, d=ar.get("d", 192), emb=ar.get("emb", 128), nl=ar.get("layers", 3), nh=int(ar.get("nh", 4)), rope=bool(ar.get("rope", str(ar.get("arch", "v1")) == "v2")), enc=str(ar.get("enc", "attn"))).to(dev)
model.load_state_dict(ck["model"], strict=False); model.eval()
fe = FrontEnd().to(dev).eval()
proto = F.normalize(model.proto, dim=-1).detach()
# 这两个以前是写死的: T 硬编码 4.0, label_mode 默认 "cluster".
# 对一个 T=10 / template / K=6576 的模型来说, 那等于【用 4 秒窗口去测 10 秒训练出来的模型】,
# 还拿 cluster 标签去和 6576 个 template 原型比对 —— 算出来的数字完全没有意义.
# loc_eval.py 早就做对了 (--T 0 自动 + 从 checkpoint 读 label_mode), 这里对齐它的口径.
T_eval = float(a.T) if a.T > 0 else float(ar.get("T", 4.0))
LM_eval = a.label_mode or str(ar.get("label_mode", "cluster"))
print("ckpt step=%s  K=%d  d=%d emb=%d nl=%d  T=%.1fs  label_mode=%s"
      % (ck.get("step"), K, ar.get("d"), ar.get("emb"), ar.get("layers"), T_eval, LM_eval), flush=True)

ds = SynthDS(T=T_eval, length=1200, seed=5, max_ev=24, dense_frac=0.3, silence_frac=0.3,
             empty_frac=0.2, min_src=1, max_src=4, ivl_med=1.0, ivl_sig=0.5,
             label_mode=LM_eval)

def tier(n):
    return "easy(1-3)" if n <= 3 else "mid(4-7)" if n <= 7 else "hard(8-12)" if n <= 12 else "dense(13+)"

E_or, L_or, E_dt, L_dt, T_or, T_dt = [], [], [], [], [], []
t0 = time.time()
batch = []
for i in range(a.windows):
    batch.append(ds[i])
    if len(batch) < 8 and i < a.windows - 1:
        continue
    mix = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
    with torch.no_grad():
        lg, e = model(fe(mix))
        p = torch.sigmoid(lg).cpu().numpy(); E = e.cpu().numpy()
    for bi, b in enumerate(batch):
        _, po, ev, lb = b
        nev = int((lb >= 0).sum())
        if nev == 0:
            continue
        # detected onsets
        pi = p[bi]
        med = float(np.median(pi)); mad = float(np.median(np.abs(pi - med))) * 1.4826
        thr = float(min(max(med + 6.0 * (mad if mad > 1e-6 else 0.05), 0.30), 0.90))
        pk, _ = alab.find_peaks(pi, thr, 3)
        Tm2 = E[bi].shape[0]
        for j in range(ev.shape[0]):
            if lb[j] < 0:
                continue
            st = int(max(0, int(ev[j, 0]) // 2)); en = int((int(ev[j, 1]) + 1) // 2)
            en = min(max(en, st + 1), Tm2)
            if en <= st: continue
            E_or.append(E[bi, st:en].mean(0)); L_or.append((int(lb[j]), tier(nev)))
            ot = int(ev[j, 0]) * 0.01
            hit = [q for q in pk if abs(q * 0.02 - ot) <= 0.06]
            if hit:
                q = min(hit, key=lambda q: abs(q * 0.02 - ot))
                z = min(q + POOL_FRAMES, Tm2)
                E_dt.append(E[bi, q:z].mean(0)); L_dt.append((int(lb[j]), tier(nev)))
    batch = []
E_or = np.stack(E_or); E_dt = np.stack(E_dt)
print("events: oracle-onset %d   detected-onset %d   (%.0fs)" % (len(E_or), len(E_dt), time.time() - t0), flush=True)

KS = [k for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 64, 80, 100, 128, 160, 192,
                  224, 256, 320, 384, 448, 512, 640, 768, 1024, 2048, 4096, 6576] if k <= K]
if KS[-1] != K:
    KS.append(K)
def curve(E, L, tag):
    er = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    cos = er @ proto.cpu().numpy().T
    order = np.argsort(-cos, axis=1)
    lab = np.array([x[0] for x in L]); tr = np.array([x[1] for x in L])
    rank = np.empty(len(lab), dtype=np.int64)
    for i in range(len(lab)):
        rank[i] = int(np.where(order[i] == lab[i])[0][0]) + 1
    print("\n=== %s  (n=%d) ===" % (tag, len(lab)))
    print("%-6s %-8s %s" % ("k", "overall", "  ".join("%-11s" % t for t in ["easy(1-3)", "mid(4-7)", "hard(8-12)", "dense(13+)"])))
    res = {}
    for k in KS:
        ok = rank <= k
        res[k] = ok.mean()
        cells = []
        for t in ["easy(1-3)", "mid(4-7)", "hard(8-12)", "dense(13+)"]:
            m = tr == t
            cells.append("%-11s" % ("%.1f%%" % (100 * ok[m].mean()) if m.sum() >= 20 else "n/a"))
        print("%-6d %-8s %s" % (k, "%.2f%%" % (100 * ok.mean()), "  ".join(cells)))
    hit = [k for k in KS if res[k] >= 0.99]
    print("--> 99%% reached at top-%s" % (hit[0] if hit else "never (max %.2f%% at k=%d)" % (100 * res[KS[-1]], KS[-1])))
    for t in ["easy(1-3)", "mid(4-7)", "hard(8-12)", "dense(13+)"]:
        m = tr == t
        if m.sum() < 20: continue
        h = [k for k in KS if (rank[m] <= k).mean() >= 0.99]
        print("    %-11s 99%% at top-%s   (n=%d)" % (t, h[0] if h else "never, max %.1f%%" % (100 * (rank[m] <= KS[-1]).mean()), m.sum()))
    return res

R1 = curve(E_or, L_or, "ground-truth onset + whole-span pooling (ceiling)")
R2 = curve(E_dt, L_dt, "detected onset + %d-frame pooling (what infer.py actually does)" % POOL_FRAMES)
if a.out:
    json.dump({"ckpt": a.ckpt, "tag": a.tag or os.path.basename(a.ckpt), "step": ck.get("step"), "K": K,
               "T": T_eval, "label_mode": LM_eval, "windows": int(a.windows),
               "n_oracle": int(len(L_or)), "n_detected": int(len(L_dt)),
               "tiers": {t: int(sum(1 for x in L_dt if x[1] == t)) for t in ["easy(1-3)", "mid(4-7)", "hard(8-12)", "dense(13+)"]},
               "oracle": {str(k): float(v) for k, v in R1.items()},
               "detected": {str(k): float(v) for k, v in R2.items()}},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n曲线已存 -> %s" % a.out)
