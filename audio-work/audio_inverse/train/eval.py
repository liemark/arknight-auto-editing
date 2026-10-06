"""Evaluate the trained detector on fixed synthetic tiers (easy / mid / hard) + anchors."""
import os, sys, json, argparse, time, math
import numpy as np
import torch
import torch.nn.functional as F
ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
from core import FrontEnd, Detector, build_detector
from synth import SynthDS

_MID = dict(min_ev=2, max_ev=3, lo=-35.0, hi=-20.0, bg_lo=-50.0, bg_hi=-50.0)
TIERS = {
    "easy":  dict(min_ev=1, max_ev=1, lo=-25.0, hi=-10.0, bg_lo=-55.0, bg_hi=-55.0),
    "mid":   dict(_MID),
    "hard":  dict(min_ev=4, max_ev=5, lo=-38.0, hi=-25.0, bg_lo=-45.0, bg_hi=-45.0),
    "dense": dict(min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0),
    # 事件密度/电平与 mid 完全相同, 只把背景从 -50 dBFS 噪声换成【游戏解包的真实长音】
    # (BGM + 环境 + 干员战斗语音). long 与 mid 的差 = "背景换成录音里的长音" 的代价.
    # 注意 long 档的 bg_lo/bg_hi 是 RMS dBFS 而噪声档是峰值 dBFS, 见 synth.py 的说明.
    "long":  dict(_MID, bg_mode="long", bg_lo=-50.0, bg_hi=-45.0),
}
SEED_OFF = {"easy": 101, "mid": 202, "hard": 303, "dense": 404, "long": 909}
THR_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TOL_FR = 2          # +-2 model frames (=40 ms) for frame level
EV_TOL = 3          # +-3 model frames (=60 ms) for event matching
NMS = 7             # 140 ms
# 识别池化窗(输出帧). 必须跟【训练和推理】用同一个值, 否则量的是模型没训过的窗口:
# train.py --pool-frames 12 (240ms), infer.py 也默认 12; 这里原来硬编码 30 帧(600ms),
# 而历史扫描的结论正是"240ms 在四档全部优于 600ms(30帧), dense +9.2 点".
# 现在默认跟随 checkpoint 里记的训练值, 命令行可覆盖.
POOL_FRAMES = 12


def decode_onsets(p, thr, refr=3):
    """the head is trained on 50 ms ONSET pulses -> local maxima with a 60 ms refractory."""
    p = np.asarray(p)
    idx = np.where(p >= thr)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(-p[idx])]
    taken = np.zeros(len(p), dtype=bool); out = []
    for i in order:
        a = max(0, i - refr); b = min(len(p), i + refr + 1)
        if taken[a:b].any():
            continue
        taken[i] = True; out.append(int(i))
    return sorted(out)


def wilson(k, n, z=1.96):
    """95% Wilson 置信区间半宽; 用来判断测试集够不够大."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return h / d


def dilate(b, r):
    out = b.copy()
    for k in range(1, r + 1):
        out[k:] |= b[:-k]
        out[:-k] |= b[k:]
    return out


def run_tier(model, fe, ds, n, dev, dl_model):
    feats = []
    for i in range(n):
        mix, po, ev, lb = ds[i]
        with torch.no_grad():
            mel = fe(torch.from_numpy(mix)[None].to(dev))
            logit, e = model(mel)
            p = torch.sigmoid(logit)[0].cpu().numpy()
            emb = e[0].cpu().numpy()
        Tm = len(p)
        po2 = F.max_pool1d(torch.from_numpy(po)[None, None], 2, 2).squeeze().numpy()[:Tm]
        gts = []
        for j in range(ev.shape[0]):
            if lb[j] >= 0:
                a = int(ev[j, 0]) // 2; b = min(int(ev[j, 1]) // 2, Tm)
                gts.append((a, b, int(lb[j])))
        feats.append((p, po2, emb, gts, Tm))
    return feats


def score(feats, thr):
    """presence-frame F1 (model is a presence detector) + rising-edge onset events + ID accuracy."""
    tp = fp = fn = 0
    etp = efp = efn = 0
    correct1 = correct5 = ndet = 0
    for p, po2, emb, gts, Tm in feats:
        pr = p[:len(po2)] >= thr
        tgt = dilate(po2 > 0.5, TOL_FR)
        tp += int((pr & tgt).sum()); fp += int((pr & ~tgt).sum()); fn += int((tgt & ~pr).sum())
        det = decode_onsets(p, thr)
        used = set()
        for d in det:
            hit = -1
            for gi, (a, b, _) in enumerate(gts):
                if gi not in used and abs(d - a) <= EV_TOL:
                    hit = gi; break
            if hit < 0:
                for gi, (a, b, _) in enumerate(gts):
                    if gi not in used and a <= d <= b:
                        hit = gi; break
            if hit < 0:
                efp += 1
            else:
                used.add(hit); etp += 1; ndet += 1
                v = emb[max(0, d):min(Tm, d + POOL_FRAMES)].mean(0)
                v = v / (np.linalg.norm(v) + 1e-9)
                order = np.argsort(-(PROTO @ v))
                if order[0] == gts[hit][2]:
                    correct1 += 1
                if gts[hit][2] in order[:5]:
                    correct5 += 1
        efn += len(gts) - len(used)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    ep = etp / max(etp + efp, 1); er = etp / max(etp + efn, 1)
    ef1 = 2 * ep * er / max(ep + er, 1e-9)
    return dict(frame_f1=f1, frame_p=tp / max(tp + fp, 1), frame_r=tp / max(tp + fn, 1),
                ev_p=ep, ev_r=er, ev_f1=ef1, n_events=etp + efn, n_det=etp + efp,
                top1=correct1 / max(ndet, 1), top5=correct5 / max(ndet, 1),
                k1=correct1, k5=correct5, nd=ndet,
                ci_top1=wilson(correct1, ndet), ci_top5=wilson(correct5, ndet),
                ci_evp=wilson(etp, etp + efp), ci_evr=wilson(etp, etp + efn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="latest", help="checkpoint 路径, 或 'latest'")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--T", type=float, default=0.0,
                    help="窗口秒数. 0=自动用 checkpoint 里记的训练值 (推荐, 不会和训练对不上)")
    ap.add_argument("--pool-frames", type=int, default=0,
                    help="识别池化窗(输出帧/20ms). 0=自动跟随 checkpoint 记的训练值 (推荐); "
                         "12=240ms, 30=600ms(历史旧值)")
    ap.add_argument("--out", default="", help="默认 eval_report_<tag>_s<step>.md")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.ckpt == "latest":
        d = os.path.join(ML, "ckpt")
        c = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("ckpt_") and f.endswith(".pt")]
        if not c:
            raise SystemExit("ckpt/ 下没有 ckpt_*.pt")
        a.ckpt = max(c, key=os.path.getmtime)
    ck = torch.load(a.ckpt, map_location=dev)
    K = int(ck["K"]); args = ck.get("args", {})
    tag = str(ck.get("tag", "") or "notag"); cstep = ck.get("step", "?")
    # label_mode must follow the checkpoint, otherwise a template-mode model is scored against
    # cluster labels and every accuracy reads ~0.
    LM = str(args.get("label_mode", "cluster"))
    if a.T <= 0:                       # 自动跟随 checkpoint 的窗口长度
        a.T = float(args.get("T", 4.0))
    print("label_mode = %s   window T = %.1fs" % (LM, a.T), flush=True)
    if not a.out:
        a.out = os.path.join(ML, "eval_report_%s_s%s.md" % (tag, cstep))
    model = Detector(K, d=args.get("d", 192), emb=args.get("emb", 128), nl=args.get("layers", 3), nh=int(args.get("nh", 4)), rope=bool(args.get("rope", str(args.get("arch", "v1")) == "v2")), enc=str(args.get("enc", "attn"))).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    print("loaded %s (step %s, C=%d)" % (a.ckpt, ck.get("step", "?"), K), flush=True)
    global PROTO, POOL_FRAMES
    POOL_FRAMES = max(1, int(a.pool_frames) if a.pool_frames > 0
                      else int(args.get("pool_frames", 12) or 12))
    print("识别池化 = %d 帧 (%.0f ms)%s" % (
        POOL_FRAMES, POOL_FRAMES * 20,
        "  <- 跟随 checkpoint" if a.pool_frames <= 0 else "  <- 命令行覆盖"), flush=True)
    PROTO = F.normalize(model.proto, dim=-1).detach().cpu().numpy()
    fe = FrontEnd().to(dev).eval()
    fe_nopcen = FrontEnd(pcen=False).to(dev).eval()

    lines = ["# SFX detector - synthetic evaluation", "",
             "checkpoint: %s (tag %s, step %s), classes C=%d, window T=%.1fs, pool %.0fms, n=%d per tier, seed=%d" % (
                 a.ckpt, tag, cstep, K, a.T, POOL_FRAMES * 20, a.n, a.seed), "",
             "| tier | 事件数 | 检出 | frame F1 | event P | event R | event F1 | top-1 (95%CI) | top-5 (95%CI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    results = {}
    for name, cfg in TIERS.items():
        cfg = dict(cfg); cfg["label_mode"] = LM
        ds = SynthDS(T=a.T, length=10 ** 4, seed=a.seed + SEED_OFF[name], **cfg)
        t0 = time.time()
        feats = run_tier(model, fe, ds, a.n, dev, None)
        best = None
        for thr in THR_GRID:
            s = score(feats, thr)
            if best is None or s["ev_f1"] > best[1]["ev_f1"]:      # 按事件 F1 选阈值(我们真正关心的)
                best = (thr, s)
        thr, s = best
        results[name] = dict(thr=thr, **s)
        print("%-5s thr %.1f | 事件 %4d 个(检出 %4d) | 帧F1 %.3f | 事件P %.3f±%.3f  R %.3f±%.3f  F1 %.3f | "
              "top1 %.1f%%±%.1f  top5 %.1f%%±%.1f | %.0fs" % (
            name, thr, s["n_events"], s["n_det"], s["frame_f1"],
            s["ev_p"], 100 * s["ci_evp"], s["ev_r"], 100 * s["ci_evr"], s["ev_f1"],
            100 * s["top1"], 100 * s["ci_top1"], 100 * s["top5"], 100 * s["ci_top5"],
            time.time() - t0), flush=True)
        lines.append("| %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.1f%% ±%.1f | %.1f%% ±%.1f |" % (
            name, s["n_events"], s["n_det"], s["frame_f1"], s["ev_p"], s["ev_r"], s["ev_f1"],
            100 * s["top1"], 100 * s["ci_top1"], 100 * s["top5"], 100 * s["ci_top5"]))

    # anchor: clean condition (no distortion, no codec, no truncation)
    ds = SynthDS(T=a.T, length=10 ** 4, seed=a.seed + 7, no_fx=True,
                 label_mode=LM, **TIERS["easy"])
    feats = run_tier(model, fe, ds, a.n, dev, None)
    best = None
    for thr in THR_GRID:
        s = score(feats, thr)
        if best is None or s["ev_f1"] > best[1]["ev_f1"]:
            best = (thr, s)
    results["clean"] = dict(thr=best[0], **best[1])
    print("clean(thr %.1f) frameF1 %.3f top1 %.3f top5 %.3f" % (best[0], best[1]["frame_f1"], best[1]["top1"], best[1]["top5"]))
    lines.append("| clean (anchor) | %d | %d | %.3f | %.3f | %.3f | %.3f | %.1f%% ±%.1f | %.1f%% ±%.1f |" % (
        best[1]["n_events"], best[1]["n_det"], best[1]["frame_f1"], best[1]["ev_p"], best[1]["ev_r"],
        best[1]["ev_f1"], 100 * best[1]["top1"], 100 * best[1]["ci_top1"],
        100 * best[1]["top5"], 100 * best[1]["ci_top5"]))

    # anchor: hard tier without PCEN
    ds = SynthDS(T=a.T, length=10 ** 4, seed=a.seed + SEED_OFF["hard"],
                 label_mode=LM, **TIERS["hard"])
    feats = run_tier(model, fe_nopcen, ds, a.n, dev, None)
    s = score(feats, results["hard"]["thr"])
    print("hard w/o PCEN frameF1 %.3f top1 %.3f" % (s["frame_f1"], s["top1"]))
    lines.append("| hard, no PCEN (ablation) | %d | %d | %.3f | %.3f | %.3f | %.3f | %.1f%% ±%.1f | %.1f%% ±%.1f |" % (
        s["n_events"], s["n_det"], s["frame_f1"], s["ev_p"], s["ev_r"], s["ev_f1"],
        100 * s["top1"], 100 * s["ci_top1"], 100 * s["top5"], 100 * s["ci_top5"]))
    thr = results["mid"]["thr"]
    secs = min(a.n, 400) * a.T
    # anchors: pure background, no events -> false-alarm rate per second.
    #   噪声      : 峰值 -50 dBFS    (long_probe.py 实测背景 RMS p50 = -62.8)
    #   真实长音  : RMS -50..-45     —— 录音里 BGM/语音的真实量级
    #   长音等响度: RMS -67.5        —— 实测背景 RMS 与噪声那档持平, 单独隔离
    #                                   "背景【结构】变了" 造成的虚警, 不带响度这个变量
    anchors = [("噪声 (峰值 -50 dBFS)", dict(bg_lo=-50.0, bg_hi=-50.0), 505),
               ("真实长音 BGM+语音 (RMS -50..-45)", dict(bg_mode="long", bg_lo=-50.0, bg_hi=-45.0), 606),
               ("真实长音 BGM+语音 (等响度 RMS -67.5)", dict(bg_mode="long", bg_lo=-67.5, bg_hi=-67.5), 707)]
    print("")
    lines += ["", "- 阈值 %.1f 下的**纯背景虚警率** (%.0f 秒背景音, 零事件):" % (thr, secs)]
    for aname, akw, aseed in anchors:
        dsb = SynthDS(T=a.T, length=10 ** 4, seed=a.seed + aseed, min_ev=0, max_ev=0,
                      label_mode=LM, **akw)
        ft = run_tier(model, fe, dsb, min(a.n, 400), dev, None)
        n_fa = sum(len(decode_onsets(p, thr)) for p, _, _, _, _ in ft)
        print("纯背景虚警 %-34s %.2f 次/秒  (%.0f 个 / %.0f 秒)" % (aname, n_fa / secs, n_fa, secs))
        lines.append("  - %s: **%.2f 次/秒**" % (aname, n_fa / secs))
    lines += ["- random-guess floor for top-1 = 1/%d = %.5f" % (K, 1.0 / K), ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(results, open(os.path.join(ML, "eval_results_%s_s%s.json" % (tag, cstep)), "w"), indent=1)
    print("report ->", a.out)


if __name__ == "__main__":
    main()
