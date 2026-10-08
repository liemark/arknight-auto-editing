"""诊断：录音里到底有没有【能对齐的】原子？—— 决定"能不能抵消"的根本问题。

对每个事件, 取 render 用的 atomlib 原子(48k, 与渲染链同一份波形), 在录音里搜索
±30 ms 的最佳对齐, 算零均值归一化互相关(NCC)与最小二乘增益:
  * NCC(报告延迟处) 低 而 NCC(最佳对齐) 高  -> 原子在, 只是延迟没对齐到采样精度 -> 可修
  * 两者都低                              -> 原子不在录音里(有损/重采样/处理过) -> 这套原子抵消不了
纯 CPU, 不碰 GPU。
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
NEV = 10
SEARCH_MS = 30.0


def ncc_best(x, seg, c, search=SR * SEARCH_MS / 1000.0, coarse=8):
    """seg 里对 x 做 ±search 采样对齐搜索. c = seg 中对应"零延迟"的下标。
    返回 (ncc@0, 最佳 ncc, 最佳 lag 采样, 该处 LS 增益)。"""
    x = np.asarray(x, np.float64); x = x - x.mean(); nx = np.linalg.norm(x)
    if nx < 1e-9:
        return 0.0, 0.0, 0, 0.0
    best = (0.0, 0, 0.0)
    for lag in list(range(-int(search), int(search) + 1, coarse)):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        w = seg[a:a + len(x)]
        w = w - w.mean(); nw = np.linalg.norm(w)
        if nw < 1e-9:
            continue
        r = float(x @ w) / (nx * nw)
        if abs(r) > abs(best[0]):
            best = (r, lag, float(x @ w) / (nx * nx))
    fine = best
    for lag in range(best[1] - coarse, best[1] + coarse + 1):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        w = seg[a:a + len(x)]
        w = w - w.mean(); nw = np.linalg.norm(w)
        if nw < 1e-9:
            continue
        r = float(x @ w) / (nx * nw)
        if abs(r) > abs(fine[0]):
            fine = (r, lag, float(x @ w) / (nx * nx))
    w0 = seg[c:c + len(x)]
    w0 = w0 - w0.mean()
    r0 = float(x @ w0) / (nx * np.linalg.norm(w0) + 1e-12)
    return r0, fine[0], fine[1], fine[2]


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
for name in ("v82_mono", "nl_mono"):
    tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_%s_v3d.json" %
                                     ("v82" if name.startswith("v82") else "nl_mono")),
                        encoding="utf-8"))
    mix = read_wav(os.path.join(AI, "data", "atoms", name + ".wav"), mono=True)
    mix = np.asarray(mix, np.float32)
    print("=" * 96)
    print("%s   %.1f s @%d Hz   事件 %d 个（按 top-1 增益取前 %d）" % (
        name, len(mix) / SR, SR, len(tl), NEV))
    print("=" * 96)
    order = sorted(tl, key=lambda e: -e["cands"][0]["gain"])[:NEV]
    print("%8s %-10s %-26s %8s %8s %9s %8s %8s" % (
        "t(s)", "类别", "模板", "cos", "增益", "NCC@报告", "NCC最佳", "最佳偏移"))
    rows = []
    for e in order:
        t = e["t"]; c = e["cands"][0]
        aid = mp.get(int(c["class"]))
        if aid is None:
            continue
        x = lib.get48k(int(aid))
        x = x[:int(2.0 * SR)]                                # 与 --max-refit-s 2.0 一致
        if len(x) < 1024:
            continue
        pad = int(0.1 * SR)
        a = max(0, int(t * SR) - pad)
        seg = mix[a: a + len(x) + 2 * pad].astype(np.float64)
        c0 = int(t * SR) - a                              # seg 中对应报告延迟的下标
        if len(seg) < len(x) + 2 * pad - 8 or c0 < 0:
            continue
        r0, rb, lag, g = ncc_best(x, seg, c0)
        w = seg[c0:c0 + len(x)]
        res = w - g * (x - x.mean())
        red = 10 * np.log10((w ** 2).sum() / max((res ** 2).sum(), 1e-12))
        rows.append((r0, rb, lag, g, red))
        print("%8.2f %-10s %-26s %8.2f %8.4f %9.3f %8.3f %7.1fms" % (
            t, c["cat"], c["template"][:26], c["cos"], c["gain"], r0, rb, lag / SR * 1000))
    if rows:
        r = np.array(rows)
        print("-" * 96)
        print("  NCC@报告延迟 中位 %.3f   |   最佳对齐 NCC 中位 %.3f  (>=0.7 的事件 %d/%d)"
              % (np.median(np.abs(r[:, 0])), np.median(np.abs(r[:, 1])),
                 int((np.abs(r[:, 1]) >= 0.7).sum()), len(r)))
        print("  最佳偏移 中位 %.1f ms, 绝对值中位 %.1f ms   |   单原子最小二乘可降 %.1f dB (中位)"
              % (np.median(r[:, 2]) / SR * 1000, np.median(np.abs(r[:, 2])) / SR * 1000,
                 np.median(r[:, 4])))
