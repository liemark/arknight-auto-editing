"""Run the trained coarse-class detector over a long recording and emit an event timeline.

Pipeline per event: onset peak -> pooled embedding -> prototype top-k -> non-negative least
squares refit against the real template waveforms (joint solve; resolves overlapping sounds).
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
ML = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ML), "data", "atoms")
sys.path.insert(0, ML); sys.path.insert(0, D)
from core import FrontEnd, Detector, build_detector, PoolAttn
import alab

SR = 44100
CAT = [("p_atk", "\u666e\u653b"), ("p_skill", "\u6280\u80fd"), ("p_field", "\u6280\u80fd"),
       ("p_imp", "\u547d\u4e2d/\u5f39\u7740"), ("p_aoe", "AOE"), ("e_", "\u654c\u4eba"),
       ("enmy", "\u654c\u4eba"), ("g_ui", "\u901a\u7528/UI"), ("g_", "\u901a\u7528/UI"),
       ("general", "\u901a\u7528/UI"), ("b_ui", "\u6218\u6597"), ("btl_snd", "\u6218\u6597"),
       ("d_avg", "\u5267\u60c5"), ("avg_se", "\u5267\u60c5"), ("a_bat", "\u73af\u5883"),
       ("v_", "\u4eba\u58f0"), ("dialog", "\u5bf9\u8bdd")]


def cat(name):
    for k, v in CAT:
        if name.startswith(k):
            return v
    return "\u5176\u4ed6"


def load_audio(path):
    x, sr = alab.wav_read(path, mono=True)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = alab.bandpass(x, sr, 30.0, min(sr, SR) * 0.475)
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x)
    return np.asarray(x, dtype=np.float32)


def peaks(p, thr, refr):
    idx = np.where(p >= thr)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    order = idx[np.argsort(-p[idx])]
    taken = np.zeros(len(p), dtype=bool); out = []
    for i in order:
        a = max(0, i - refr); b = min(len(p), i + refr + 1)
        if taken[a:b].any():
            continue
        taken[i] = True; out.append(int(i))
    return np.array(sorted(out), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="latest", help="checkpoint 路径, 或 'latest'")
    ap.add_argument("--wav", default=os.path.join(D, "nl_mono.wav"))
    ap.add_argument("--out", default="", help="默认 timeline_<wav>_<tag>_s<step>.json")
    ap.add_argument("--txt", default="", help="默认 out/timeline_<wav>_<tag>_s<step>.txt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mad", type=float, default=6.0)
    ap.add_argument("--thr-lo", type=float, default=0.30)
    ap.add_argument("--thr-hi", type=float, default=0.90)
    ap.add_argument("--chunk", type=float, default=20.0)
    ap.add_argument("--gain-thr", type=float, default=0.05)
    ap.add_argument("--max-refit-s", type=float, default=2.0,
                    help="NNLS 重拟合的窗长上限(秒). 必须 <= 训练时模板被截断的长度 "
                         "(synth.py 的 lmax_fx = 2.0s): 模板库现在是【变长流】, 最长可达 83s, "
                         "而 alab.nnls 要枚举 2^n-1 个子集, 每次 lstsq 的矩阵高度 = 最长候选模板 -> "
                         "不截断会慢几个数量级(旧库封顶 2.5s 所以没暴露). 0=不截断")
    ap.add_argument("--top-per-onset", type=int, default=1, help="每个 onset 最多保留几条")
    ap.add_argument("--top", type=int, default=0, help="整体只保留概率最高的前 N 个事件 (0=全部)")
    ap.add_argument("--pool-mode", default="mean", choices=["mean", "attn"],
                    help="attn = 用 checkpoint 里学到的池化权重 (训练时 --pool-mode attn 才有)")
    ap.add_argument("--refr", type=int, default=1,
                    help="峰不应期(输出帧/20ms). 实测 loc_eval 的 P-R 平面上 refr=1 全面压过 3: "
                         "同样的召回下精确率更高 (85%% 召回档: 48.8%% vs 42.6%%). 3=旧行为")
    ap.add_argument("--pool-frames", type=int, default=12,
                    help="事件识别时往后池化多少输出帧(20ms/帧). 实测扫描: 240ms(12帧) 在四档全部优于"
                         "原来的 600ms(30帧), dense 档 +9.2 点 (68.4->77.6). 30=旧行为")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.ckpt == "latest":
        _d = os.path.join(ML, "ckpt")
        _c = [os.path.join(_d, f) for f in os.listdir(_d) if f.startswith("ckpt_") and f.endswith(".pt")]
        if not _c:
            raise SystemExit("ckpt/ 下没有 ckpt_*.pt")
        a.ckpt = max(_c, key=os.path.getmtime)
    ck = torch.load(a.ckpt, map_location=dev)
    K = int(ck["K"]); ar = ck.get("args", {})
    _tag = str(ck.get("tag", "") or "notag"); _step = ck.get("step", "?")
    _base = "%s_%s_s%s" % (os.path.splitext(os.path.basename(a.wav))[0], _tag, _step)
    if not a.out:
        a.out = os.path.join(D, "timeline_%s.json" % _base)
    if not a.txt:
        a.txt = os.path.join(D, "out", "timeline_%s.txt" % _base)
    os.makedirs(os.path.dirname(a.txt), exist_ok=True)
    model = Detector(K, d=ar.get("d", 192), emb=ar.get("emb", 128), nl=ar.get("layers", 3), nh=int(ar.get("nh", 4)), rope=bool(ar.get("rope", str(ar.get("arch", "v1")) == "v2")), enc=str(ar.get("enc", "attn"))).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    pooler = None
    if a.pool_mode == "attn" and ck.get("pooler") is not None:
        pooler = PoolAttn(ar.get("emb", 128)).to(dev)
        pooler.load_state_dict(ck["pooler"]); pooler.eval()
        print("池化: attn  (模型自己学的每帧权重)", flush=True)
    else:
        if a.pool_mode == "attn":
            print("池化: 这个 checkpoint 没有 pooler, 退回 mean", flush=True)
        print("池化: mean  %d 帧 (%.0f ms)" % (a.pool_frames, a.pool_frames * 20), flush=True)
    fe = FrontEnd().to(dev).eval()
    proto = F.normalize(model.proto, dim=-1).detach()
    print("model: %s step=%s C=%d" % (a.ckpt, ck.get("step", "?"), K), flush=True)

    # class -> representative template waveform.  K == number of templates means the model was
    # trained with label_mode="template" (full 6576 library, no clustering), so the class id IS
    # the template id.  Otherwise K is the cluster count and the class maps to its first member.
    idx = json.load(open(os.path.join(ML, "bank_index.json"), encoding="utf-8"))
    lens = np.load(os.path.join(ML, "bank_lens.npy"))
    offs = np.load(os.path.join(ML, "bank_offs.npy"))
    bank = np.load(os.path.join(ML, "bank_pool.npy"), mmap_mode="r")
    if K == len(lens):
        print("mode: FULL LIBRARY (K=%d = every template, no clustering)" % K, flush=True)
        rep = {k: k for k in range(K)}
    else:
        clusters = np.load(os.path.join(ML, "clusters_%d.npy" % K))
        rep = {}
        for k, c in enumerate(clusters):
            rep.setdefault(int(c), k)
    # 不把整库预读进内存: K=22326 条、总时长 1085 分钟, 展开成 float32 约 11.5 GB,
    # 而每个事件只需要 top-k 那几个候选的波形 -> 按需从 mmap 切片.
    def tmpl(c, cap=None):
        k = rep[int(c)]; L = int(lens[k])
        if cap is not None:
            L = min(L, int(cap))
        return np.asarray(bank[int(offs[k]):int(offs[k]) + L], dtype=np.float32)

    x = load_audio(a.wav)
    print("audio %.1fs" % (len(x) / SR), flush=True)
    CH = int(a.chunk * SR)
    probs = []; embs = []
    t0 = time.time()
    for s in range(0, len(x), CH):
        seg = x[s:s + CH]
        if len(seg) < SR:
            break
        with torch.no_grad():
            mel = fe(torch.from_numpy(seg)[None].to(dev))
            logit, e = model(mel)
            probs.append(torch.sigmoid(logit)[0].cpu().numpy())
            embs.append(e[0].cpu().numpy())
    p = np.concatenate(probs); E = np.concatenate(embs, axis=0)
    print("inference %.0fs, %d frames (%.2f s/frame)" % (time.time() - t0, len(p), len(p) * 0.02), flush=True)

    med = float(np.median(p)); mad = float(np.median(np.abs(p - med))) * 1.4826
    thr = med + a.mad * (mad if mad > 1e-6 else 0.05)
    thr = float(min(max(thr, a.thr_lo), a.thr_hi))      # 自适应阈值必须落回概率区间内
    pk = peaks(p, thr, a.refr)
    print("onset 概率: p50 %.3f  p90 %.3f  p99 %.3f  max %.3f  -> 阈值 %.3f  (%d 个候选)" % (
        np.percentile(p, 50), np.percentile(p, 90), np.percentile(p, 99), p.max(), thr, len(pk)), flush=True)

    events = []
    # 注意: infer 时不知道事件的结束位置, 所以这里 pool_frames 必须为正; 0 会被当成旧行为 30 帧
    _pf = a.pool_frames if a.pool_frames > 0 else 30
    for i in pk:
        f0 = int(i); f1 = min(len(E), f0 + _pf)
        if pooler is not None:
            with torch.no_grad():
                em = torch.from_numpy(E[f0:f1]).to(dev)[None]
                mk = torch.ones(1, 1, f1 - f0, dtype=torch.bool, device=dev)
                v = pooler(em, mk)[0][0, 0].cpu().numpy()
        else:
            v = E[f0:f1].mean(0)
        v = v / (np.linalg.norm(v) + 1e-9)
        cos = (proto @ torch.from_numpy(v).to(dev)).cpu().numpy()
        top = np.argsort(-cos)[:a.k]
        cand = [int(c) for c in top if int(c) in rep]
        if not cand:
            continue
        t_start = f0 * 0.02
        w0 = int(t_start * SR); wmax = int(max(int(lens[rep[c]]) for c in cand))
        if a.max_refit_s > 0:                                   # 与训练的 lmax_fx 对齐, 见参数说明
            wmax = min(wmax, int(a.max_refit_s * SR))
        w0 = min(w0, max(0, len(x) - wmax - 1))
        seg = x[w0:w0 + wmax].astype(np.float64)
        if len(seg) < 256:
            continue
        A = np.zeros((len(seg), len(cand)))
        for j, c in enumerate(cand):
            w = tmpl(c, wmax)                                   # 同样截断, 否则广播不过去
            A[:len(w), j] = w - w.mean()
        g = alab.nnls(A, seg - seg.mean())
        keep = [(c, float(g[j]), float(cos[c])) for j, c in enumerate(cand) if g[j] > a.gain_thr]
        keep.sort(key=lambda z: -z[1])
        if not keep and cand:                       # 一个都没过阈值时, 保留增益最高的那一个 (top-1)
            j = int(np.argmax(g))
            keep = [(cand[j], float(g[j]), float(cos[cand[j]]))]
        keep = keep[:max(1, a.top_per_onset)]
        if not keep:
            continue
        events.append({"t": round(t_start, 3), "onset_prob": float(p[f0]),
                       "cands": [{"class": c, "template": idx[rep[c]]["name"],
                                  "bundle": idx[rep[c]]["bundle"], "cat": cat(idx[rep[c]]["name"]),
                                  "gain": gv, "cos": cs} for c, gv, cs in keep[:4]]})

    if a.top > 0:
        events.sort(key=lambda e: -e["onset_prob"])
        events = sorted(events[:a.top], key=lambda e: e["t"])
    json.dump(events, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(a.txt, "w", encoding="utf-8") as f:
        f.write("=== timeline: %s ===\n" % os.path.basename(a.wav))
        f.write("model %s (step %s)  onsets>=%.3f  %d events\n\n" % (os.path.basename(a.ckpt), ck.get("step", "?"), thr, len(events)))
        for e in events:
            top = e["cands"][0]
            f.write("%8.2fs  %-9s  %s\n" % (e["t"], top["cat"], "  ".join(
                "%s(g=%.3f,cos=%.2f)" % (c["template"], c["gain"], c["cos"]) for c in e["cands"])))
    print("timeline -> %s  (%d events)" % (a.out, len(events)), flush=True)
    for e in events[:40]:
        t = e["cands"][0]
        print("  %7.2fs %-9s %s" % (e["t"], t["cat"], "  ".join("%s(%.2f)" % (c["template"], c["gain"]) for c in e["cands"][:3])))
    print("txt ->", a.txt)


if __name__ == "__main__":
    main()
