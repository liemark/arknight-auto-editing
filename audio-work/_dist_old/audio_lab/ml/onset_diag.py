"""诊断: onset 概率分布 —— 合成 vs 真实录音 vs 孤立模板。

回答的问题: infer.py 在真实录音上算出 p50≈0.3~0.5(一半的帧都在"响"),
而同一个模型在合成数据上 oracle top-1 有 58%。到底是
  (a) domain gap —— 模型在合成上好好的, 一碰真实录音就饱和; 还是
  (b) 模型本身坏了 —— 连合成窗口也饱和, 那 58% 是别的口径量出来的。

四组对照, 同一个模型同一个 FrontEnd:
  [A] SynthDS 合成窗口   —— 训练/评测同分布
  [B] 真实录音切片       —— 目标域
  [C] 孤立模板(单音效重复, 无背景) —— 最干净的"应该有峰"的输入
  [D] 数字静音 / 纯噪声底 —— 最干净的"不该有峰"的输入

C 和 D 是关键: 如果连 C 都 p50=0.5、D 也 p50=0.5, 检测头根本没在区分;
如果 C 干净、D 干净、只有 B 饱和, 那是 domain gap。
"""
import os, sys, argparse
import numpy as np, torch
ML = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(ML)                      # 工作根目录（sfx/ long/ 与 alab.py 都在这层）
sys.path.insert(0, ML); sys.path.insert(0, D)
from core import FrontEnd, Detector
from synth import SynthDS
import alab

SR = 16000
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--wav", default=os.path.join(D, "nl_mono.wav"))
ap.add_argument("--windows", type=int, default=16, help="每组测几个窗口")
ap.add_argument("--out", default="")
a = ap.parse_args()

dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ar = ck.get("args", {}); K = int(ck["K"])
model = Detector(K, d=ar.get("d", 192), emb=ar.get("emb", 128), nl=ar.get("layers", 3),
                 nh=int(ar.get("nh", 4)), rope=bool(ar.get("rope", str(ar.get("arch", "v1")) == "v2"))).to(dev)
miss = model.load_state_dict(ck["model"], strict=False)
model.eval()
fe = FrontEnd().to(dev).eval()
T = float(ar.get("T", 10.0)); LM = str(ar.get("label_mode", "template"))
print("ckpt=%s  step=%s  T=%.1fs  label_mode=%s  K=%d  nl=%d" % (
    os.path.basename(a.ckpt), ck.get("step"), T, LM, K, ar.get("layers")), flush=True)
print("加载: 缺失 %d 个, 多余 %d 个" % (len(miss.missing_keys), len(miss.unexpected_keys)), flush=True)
print("%-30s %-8s %s" % ("组", "rms", "onset 概率分位"), flush=True)

rows = []


def probe(tag, xs):
    """xs: list of 1-D float32 windows, 每个长 T 秒."""
    acc = []
    for x in xs:
        with torch.no_grad():
            mel = fe(torch.from_numpy(np.asarray(x, dtype=np.float32))[None].to(dev))
            lg, _ = model(mel)
            acc.append(torch.sigmoid(lg)[0].cpu().numpy())
    p = np.concatenate(acc)
    rms = float(np.mean([np.sqrt(np.mean(np.asarray(x) ** 2)) for x in xs]))
    q = np.percentile(p, [50, 75, 90, 99])
    row = dict(tag=tag, rms=rms, p50=float(q[0]), p75=float(q[1]), p90=float(q[2]), p99=float(q[3]),
               pmax=float(p.max()), frac50=float((p > 0.5).mean()), frac90=float((p > 0.9).mean()))
    rows.append(row)
    print("%-30s %-8.4f p50=%.3f p75=%.3f p90=%.3f p99=%.3f max=%.3f   >0.5:%5.1f%%  >0.9:%5.1f%%" % (
        tag, rms, q[0], q[1], q[2], q[3], p.max(), 100 * row["frac50"], 100 * row["frac90"]), flush=True)
    return p


# ---------- [A] 合成窗口 (与 recall_at_k.py 完全同参) ----------
ds = SynthDS(T=T, length=1200, seed=5, max_ev=24, dense_frac=0.3, silence_frac=0.3,
             empty_frac=0.2, min_src=1, max_src=4, ivl_med=1.0, ivl_sig=0.5, label_mode=LM)
probe("A 合成窗口 SynthDS", [ds[i][0] for i in range(a.windows)])

# ---------- [B] 真实录音 ----------
x, sr = alab.wav_read(a.wav, mono=True)
if x.ndim > 1:
    x = x.mean(axis=1)
if sr != SR:
    x = alab.bandpass(x, sr, 30.0, 7500.0)
    x = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x)
x = np.asarray(x, dtype=np.float32)
n = int(T * SR)
chunks = [x[s:s + n] for s in range(0, len(x) - n, n)]
print("  (真实录音 %.1fs -> %d 个 %0.0fs 窗口)" % (len(x) / SR, len(chunks), T), flush=True)
probe("B 真实录音 %s" % os.path.basename(a.wav), chunks[:a.windows])

# 电平剖面: 训练里 SFX 增益档是 10**(uniform(-32,0)/20), 背景是 -70..-40 dBFS RMS.
# 真实录音的电平到底落在哪? 用逐帧 RMS 的分位数估一下"背景底"和"事件峰"差多少 dB.
def rms_db(y):
    r = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    return 20 * np.log10(r + 1e-12)

fr, hop = alab.frame_rms(x, SR, frame=0.010, hop=0.010)
fr_db = 20 * np.log10(fr + 1e-12)
q = np.percentile(fr_db, [5, 25, 50, 75, 90, 99])
print("  真实录音逐帧 RMS 电平(dBFS): p5 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f  p99 %.1f"
      % tuple(q), flush=True)
print("    -> 背景底(p25) 与 事件峰(p99) 相差 %.1f dB" % (q[5] - q[1]), flush=True)
print("    训练背景档 -70..-40 dBFS 对 p25=%.1f 是【偏低 %.1f dB】"
      % (q[1], q[1] - (-55.0)), flush=True)

# ---------- [C] 孤立模板: 单个音效按 1s 间隔重复, 无背景 ----------
lens = np.load(os.path.join(ML, "bank_lens.npy"))
bank = np.load(os.path.join(ML, "bank_clean.npy"), mmap_mode="r")
rng = np.random.default_rng(7)
cs = []
for _ in range(a.windows):
    k = int(rng.integers(0, len(lens)))
    L = int(lens[k]); w = np.asarray(bank[k, :L], dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    g = 10 ** (rng.uniform(-20.0, -10.0) / 20.0)      # 训练里的典型增益档 (lo..hi 附近)
    t = 0.5
    while t * SR + L < n:
        o = int(t * SR)
        y[o:o + L] += w * g
        t += 1.0
    cs.append(y)
probe("C 孤立模板(重复,无背景)", cs)

# ---------- [D] 数字静音 / 噪声底 ----------
probe("D 数字静音", [np.zeros(n, dtype=np.float32) for _ in range(a.windows)])
noise = np.random.default_rng(11).normal(0, 0.003, (a.windows, n)).astype(np.float32)
probe("D 噪声底(-50dB)", [noise[i] for i in range(a.windows)])

# ---------- [E] 合成 SFX + 真实录音背景 (bg16.npy) ----------
# 这一组才是"如果拿真实背景重训, 模型会看到什么"。逐档抬背景电平, 看 p50 什么时候饱和:
# 如果一加真实背景 p50 就冲到 0.3+, 就坐实了"背景内容"是根因;
# 如果只有把背景抬到接近真实录音的电平才饱和, 那"背景电平"是第二个必须一起修的变量。
for db in [None, -45.0, -38.0, -32.0, -26.0]:
    kw = {} if db is None else {"bg_lo": float(db), "bg_hi": float(db)}
    dsr = SynthDS(T=T, length=1200, seed=5, max_ev=24, dense_frac=0.3, silence_frac=0.0,
                  empty_frac=0.0, min_src=1, max_src=4, ivl_med=1.0, ivl_sig=0.5,
                  label_mode=LM, bg_mode="recording", **kw)
    wins = [dsr[i][0] for i in range(a.windows)]
    probe("E SFX+真实背景 %s" % ("默认(-70..-40)" if db is None else "%+.0f dBFS" % db), wins)

if a.out:
    import json
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out, flush=True)
