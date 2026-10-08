"""正面示例：录音里【能对上】的那部分（语音）真做一次采样级对齐 + 增益拟合 + 反相。

背景：全库搜索证明录音里的 SFX 与库中任何变体都不波形相关(|NCC| 0.23-0.34),
但语音 CN_022 在 nl_mono 里 NCC 0.98（延迟约 -10ms，正好是 20ms 帧格误差）。
这里对语音事件做:
  延迟搜索(±30ms, 1 采样步) x 最小二乘增益 -> 铺反相轨 -> 报该段 PSR -> 导出试听片段。
产物: out/demo/voice_orig.wav / voice_resid.wav（同一段 8s）
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav, rms_normalize_db                   # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 48000
NCC_MIN = 0.55
SEARCH = int(SR * 0.03)


def fit(x, mix, t0):
    """在 t0 附近搜 (delay, gain) 使残差最小 -> (delay采样, gain, ncc)。"""
    nx = np.linalg.norm(x - x.mean())
    best = (0.0, 0, 0.0)
    for lag in range(-SEARCH, SEARCH + 1):
        a = t0 + lag
        if a < 0 or a + len(x) > len(mix):
            continue
        w = mix[a:a + len(x)]
        wc = w - w.mean()
        nw = np.linalg.norm(wc)
        if nw < 1e-9:
            continue
        g = float((x - x.mean()) @ wc) / (nx * nx)
        r = float((x - x.mean()) @ wc) / (nx * nw)
        res = wc - g * (x - x.mean())
        c = float(res @ res)
        if best[2] == 0.0 or c < best[0]:
            best = (c, lag, g)
            best_ncc = r
    return best[1], best[2], best_ncc


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", "nl_mono.wav"), mono=True),
                 np.float32).astype(np.float64)
tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_nl_mono_v3d.json"),
                    encoding="utf-8"))

cancel = np.zeros_like(mix)
kept = []
for e in tl:
    c0 = e["cands"][0]
    if c0["cat"] != "其他":            # 只做语音（"其他" = CN_xxx 语音）
        continue
    aid = mp.get(int(c0["class"]))
    if aid is None:
        continue
    x, asr = lib.raw(int(aid))
    if asr != SR:
        from audio_inverse.audio import resample
        x = resample(x, asr, SR)
    x = np.asarray(rms_normalize_db(x, -34.0), np.float64)
    if len(x) > 4 * SR or len(x) < 1024:
        continue
    t0 = int(round(e["t"] * SR))
    if t0 + len(x) > len(mix):
        continue
    # 先在全段残差上拟合（避免与已铺的反相轨冲突）
    resid = mix + cancel
    lag, g, r = fit(x, resid, t0)
    if abs(r) < NCC_MIN or abs(g) < 1e-4:
        continue
    a = t0 + lag
    cancel[a:a + len(x)] += -g * (x - x.mean())
    kept.append((e["t"], c0["template"], r, g, lag / SR * 1000, len(x) / SR))

print("语音事件（|NCC| >= %.2f）：%d 个" % (NCC_MIN, len(kept)))
for t, nm, r, g, lag_ms, dur in kept:
    print("  t=%7.2fs  %-12s  NCC %+.3f  增益 %.3f  延迟 %+.1fms  时长 %.2fs"
          % (t, nm, r, g, lag_ms, dur))

res = mix + cancel
print("\n全片 PSR: %.2f dB" % (10 * np.log10((mix ** 2).sum() / ((res ** 2).sum() + 1e-12))))
os.makedirs(os.path.join(AI, "out", "demo"), exist_ok=True)
# 导出每个语音事件附近 8s 的试听片段
for i, (t, nm, r, g, lag_ms, dur) in enumerate(kept[:4]):
    a = max(0, int((t - 2.0) * SR)); b = min(len(mix), a + 8 * SR)
    seg_o, seg_r = mix[a:b], res[a:b]
    psr = 10 * np.log10((seg_o ** 2).sum() / ((seg_r ** 2).sum() + 1e-12))
    sf.write(os.path.join(AI, "out", "demo", "voice%d_%s_orig.wav" % (i, nm)), seg_o, SR, subtype="PCM_16")
    sf.write(os.path.join(AI, "out", "demo", "voice%d_%s_resid.wav" % (i, nm)), seg_r, SR, subtype="PCM_16")
    print("  试听 %d: t=%.2fs %s  该 8s 段 PSR %.2f dB  (orig/resid 已导出)" % (i, t, nm, psr))
