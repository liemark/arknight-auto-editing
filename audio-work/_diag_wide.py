"""宽窗延迟搜索：把 ±30ms 放宽到 ±600ms、逐采样、配 LS 增益, 再看 NCC。

上一轮的错: 搜索窗只有 ±30ms, 而输出时间栅格是 20ms/帧、onset 头学的是 50ms 高斯脉冲,
真实对齐偏 40~80ms 完全可能 -> 搜不到就误判成"素材不在库里"。
这里用 FFT 一次算出所有 lag 的归一化互相关(逐 lag 做零均值/能量归一), 取最大值。
纯 CPU。
"""
import json
import os
import sys

import numpy as np

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav, resample, rms_normalize_db        # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 48000
WIDE = 0.6          # 搜索半径(秒), 相当于 ±600ms
NEV = 20


def ncc_wide(x, w):
    """x 在 w 内滑动, 返回逐 lag 的零均值 NCC 数组（长度 = len(w)-len(x)+1）。"""
    xc = x - x.mean()
    nx = np.linalg.norm(xc)
    L, N = len(x), len(w)
    if nx < 1e-9 or N < L:
        return np.zeros(1)
    nf = 1 << int(np.ceil(np.log2(N + L)))
    C = np.fft.irfft(np.fft.rfft(w, nf) * np.conj(np.fft.rfft(xc, nf)), nf)[:N - L + 1]
    cs = np.concatenate([[0.0], np.cumsum(w)])
    cs2 = np.concatenate([[0.0], np.cumsum(w * w)])
    a = np.arange(N - L + 1)
    s1 = cs[a + L] - cs[a]
    s2 = cs2[a + L] - cs2[a]
    var = np.maximum(s2 - s1 * s1 / L, 0.0)
    return C / (np.sqrt(var) * nx + 1e-12)


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)

for tag, wavname in (("v82", "v82_mono"), ("nl_mono", "nl_mono")):
    mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", wavname + ".wav"), mono=True),
                     np.float32)
    tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_%s_v3d.json" % tag),
                        encoding="utf-8"))
    sfx = sorted([e for e in tl if e["cands"][0]["cat"] != "其他"],
                 key=lambda e: -e["cands"][0]["gain"])[:NEV]
    vox = sorted([e for e in tl if e["cands"][0]["cat"] == "其他"],
                 key=lambda e: -e["cands"][0]["gain"])[:NEV]
    print("=" * 100)
    print("%s：宽窗(±600ms, 逐采样) 延迟搜索" % wavname)
    print("=" * 100)
    print("%-14s %8s %9s %10s %10s %9s" % ("类别", "t(s)", "cos", "NCC(±30ms)", "NCC(±600ms)", "最佳偏移"))
    for label, group in (("音效", sfx), ("语音", vox)):
        n30, n600, offs = [], [], []
        for e in group:
            c0 = e["cands"][0]
            aid = mp.get(int(c0["class"]))
            if aid is None:
                continue
            x, asr = lib.raw(int(aid))
            x = np.asarray(rms_normalize_db(x, -34.0), np.float64)
            if asr != SR:
                x = np.asarray(resample(x.astype(np.float32), asr, SR), np.float64)
            x = x[:int(2.0 * SR)]
            if len(x) < 2048:
                continue
            t0 = int(round(e["t"] * SR))
            a = max(0, t0 - int(WIDE * SR))
            w = mix[a: min(len(mix), t0 + len(x) + int(WIDE * SR))].astype(np.float64)
            c0i = t0 - a
            if len(w) < len(x) + 8 or c0i < 0:
                continue
            r = ncc_wide(x, w)
            k = int(np.argmax(np.abs(r)))
            p30 = int(round(30 / 1000 * SR))
            lo = max(0, c0i - p30 // 4); hi = min(len(r), c0i + p30 // 4 + 1)
            n30.append(float(np.abs(r[lo:hi]).max()) if hi > lo else 0.0)
            n600.append(float(abs(r[k])))
            offs.append((k - c0i) / SR * 1000)
            print("%-14s %8.2f %9.2f %10.3f %10.3f %8.1fms"
                  % (label, e["t"], c0["cos"], n30[-1], n600[-1], offs[-1]))
        if n600:
            print("  %s 汇总: NCC ±30ms 中位 %.3f  ->  ±600ms 中位 %.3f   (>=0.7: %d/%d)  偏移|中位| %.0fms"
                  % (label, np.median(n30), np.median(n600),
                     int((np.array(n600) >= 0.7).sum()), len(n600),
                     np.median(np.abs(offs))))
    print()
