"""诊断：录音里的音效是不是【变了速率/音高】的？—— NCC 低到底是"没对齐"还是"被改过"。

上一测：SFX 原子在 ±30ms 内穷举对齐后 NCC 中位只有 0.16（语音却到 0.98）。
如果游戏对每次播放做了随机速率/音高变化(常见做法, 防"机关枪"感), 那么固定模板永远对不上,
但对一个【速率搜索】应该能救回来: NCC(最佳速率) 会跳回 0.7+。
纯 CPU。
"""
import json
import os
import sys

import numpy as np

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav                                    # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 48000
RATES = np.arange(0.90, 1.101, 0.01)
SEARCH_MS = 30.0
NEV = 6


def ncc(x, seg, c):
    """零均值 NCC（x 与 seg[c:c+len(x)]）。"""
    w = seg[c:c + len(x)]
    xm = x - x.mean(); wm = w - w.mean()
    d = np.linalg.norm(xm) * np.linalg.norm(wm)
    return float(xm @ wm) / d if d > 1e-12 else 0.0


def best_over(x, seg, c, srch):
    best = (0.0, 0)
    for lag in range(-srch, srch + 1, 4):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        r = ncc(x, seg, a)
        if abs(r) > abs(best[0]):
            best = (r, lag)
    f = best
    for lag in range(best[1] - 4, best[1] + 5):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        r = ncc(x, seg, a)
        if abs(r) > abs(f[0]):
            f = (r, lag)
    return f


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
srch = int(SR * SEARCH_MS / 1000)

for tag, wavname in (("v82", "v82_mono"), ("nl_mono", "nl_mono")):
    tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_%s_v3d.json" % tag),
                        encoding="utf-8"))
    mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", wavname + ".wav"), mono=True),
                     np.float32)
    # 只取音效（排除语音"其他"），按增益排序
    evs = [e for e in tl if e["cands"][0]["cat"] != "其他"]
    evs = sorted(evs, key=lambda e: -e["cands"][0]["gain"])[:NEV]
    print("=" * 104)
    print("%s：前 %d 个音效事件的【速率搜索】(0.90~1.10, 步长 0.01) + 延迟 ±30ms" % (wavname, len(evs)))
    print("=" * 104)
    print("%8s %-24s %8s %10s %10s %10s %8s" % (
        "t(s)", "模板", "增益", "NCC(原速)", "NCC(最佳)", "最佳速率", "最佳延迟"))
    best_rates, gain_best = [], []
    for e in evs:
        t = e["t"]; c0 = e["cands"][0]
        aid = mp.get(int(c0["class"]))
        if aid is None:
            continue
        x0 = lib.get48k(int(aid))[:int(2.0 * SR)]
        if len(x0) < 2048:
            continue
        pad = int(0.1 * SR)
        a = max(0, int(t * SR) - pad)
        seg = mix[a: a + len(x0) + 2 * pad].astype(np.float64)
        cc = int(t * SR) - a
        if len(seg) < len(x0) + 2 * pad - 8 or cc < 0:
            continue
        r_fix, _ = best_over(x0, seg, cc, srch)
        top = (abs(r_fix), 1.0, 0, r_fix)
        for rate in RATES:
            if abs(rate - 1.0) < 1e-9:
                continue
            n = max(256, int(len(x0) / rate))
            xr = np.interp(np.arange(n) * rate, np.arange(len(x0)), x0).astype(np.float64)
            if len(xr) > len(x0):
                continue
            r, lag = best_over(xr, seg, cc, srch)
            if abs(r) > top[0]:
                top = (abs(r), float(rate), lag, r)
        best_rates.append(top[1]); gain_best.append(top[0])
        print("%8.2f %-24s %8.4f %10.3f %10.3f %10.2f %7.1fms" % (
            t, c0["template"][:24], c0["gain"], abs(r_fix), top[0], top[1], top[2] / SR * 1000))
    if best_rates:
        br = np.array(best_rates); gb = np.array(gain_best)
        print("-" * 104)
        print("  原速 NCC 中位 %.3f  ->  速率搜索后 NCC 中位 %.3f   (>=0.7 的 %d/%d)"
              % (0.0, np.median(gb), int((gb >= 0.7).sum()), len(gb)))
        off = 100 * (br - 1.0)
        print("  最佳速率: %s   非 1.0 的比例 %.0f%%"
              % (np.round(np.unique(np.round(br, 2)), 2)[:12], 100 * (np.abs(br - 1.0) > 1e-9).mean()))
        print("  速率偏离中位 %.2f%%（绝对值 %.2f%%）" % (np.median(off), np.median(np.abs(off))))
