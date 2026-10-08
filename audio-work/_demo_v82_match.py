"""v82：把【匹配出来的事件】单独放出来听 —— 只铺识别到的原子(拟合增益+延迟), 不含原音。

用途：直接判断"匹配出来的到底是什么"。若重建出来的声音与录音不像, 那 PSR≈0 就不是渲染问题,
而是"录音里的音效根本不在库里"。

产物 (out/demo_v82/):
  matched_only.wav   全片重建(只含识别到的原子, 正相) —— 与 v82_mono.wav 对比听
  ab_orig.wav        A/B 窗口(事件最密的一段) 的原音
  ab_matched.wav     同一窗口的重建
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav, rms_normalize_db, resample        # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 48000
W0, W1 = 110.0, 140.0                    # A/B 窗口(秒)
OUT = os.path.join(AI, "out", "demo_v82")
os.makedirs(OUT, exist_ok=True)

cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", "v82_mono.wav"), mono=True),
                 np.float32)
tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_v82_v3d.json"), encoding="utf-8"))

# 重建：按 timeline 的候选(top-per-event 个) 逐一拟合增益后叠加（正相）
rec = np.zeros_like(mix, np.float64)
n_used = n_skip = 0
for e in tl:
    t = int(round(e["t"] * SR))
    best = None
    for c in e["cands"][:4]:
        aid = mp.get(int(c["class"]))
        if aid is None:
            continue
        x, asr = lib.raw(int(aid))
        x = np.asarray(rms_normalize_db(x, -34.0), np.float64)
        if asr != SR:
            x = np.asarray(resample(x.astype(np.float32), asr, SR), np.float64)
        if len(x) < 512 or t + len(x) > len(mix):
            continue
        w = mix[t:t + len(x)].astype(np.float64)
        xm = x - x.mean(); wm = w - w.mean()
        nx = np.linalg.norm(xm)
        if nx < 1e-9:
            continue
        g = float(xm @ wm) / (nx * nx)
        ncc = float(xm @ wm) / (nx * np.linalg.norm(wm) + 1e-12)
        if best is None or abs(ncc) > abs(best[0]):
            best = (ncc, g, x, c["template"])
    if best is None or abs(best[1]) < 1e-4:
        n_skip += 1
        continue
    ncc, g, x, nm = best
    rec[t:t + len(x)] += g * (x - x.mean())
    n_used += 1

print("重建: 用上 %d 个事件, 跳过 %d 个（%d 个事件里）" % (n_used, n_skip, len(tl)))
print("重建能量 / 原音能量 = %.4f  (%.1f dB)"
      % ((rec ** 2).sum() / (mix ** 2).sum(),
         10 * np.log10((rec ** 2).sum() / (mix ** 2).sum() + 1e-12)))
r = (rec / max(np.abs(rec).max(), 1e-9) * 0.9).astype(np.float32)
sf.write(os.path.join(OUT, "matched_only.wav"), r, SR, subtype="PCM_16")
a, b = int(W0 * SR), int(W1 * SR)
sf.write(os.path.join(OUT, "ab_orig.wav"), mix[a:b], SR, subtype="PCM_16")
sf.write(os.path.join(OUT, "ab_matched.wav"), r[a:b], SR, subtype="PCM_16")
print("窗口 %.0f-%.0fs 内事件 %d 个" % (W0, W1, sum(1 for e in tl if W0 <= e["t"] < W1)))
print("已导出:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print("  %-22s %7.1f s  %5.1f MB" % (f, sf.info(p).duration, os.path.getsize(p) / 1e6))
