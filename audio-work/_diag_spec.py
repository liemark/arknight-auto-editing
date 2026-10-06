"""诊断：录音里的音效是"同一个声音但波形被毁"还是"根本不是同一素材"？

波形 NCC 低有两种可能, 处理方式完全不同:
  A) 同一个声音, 但被有损编码/重采样/限幅毁掉波形 -> 包络相关与频谱余弦仍然高
     => 检测是对的, 但这套录音做不了深抵消(需要更干净的录音源)
  B) 根本不是同一素材(版本不同/别的声音) -> 包络与频谱也低
     => 模板库版本对不上, 换素材才能解决
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
NEV = 8


def spec_cos(a, b):
    """对数幅度谱(1024/240)余弦。"""
    def _s(x):
        n = 1024; h = 240
        m = 1 + (len(x) - n) // h
        if m < 4:
            return None
        idx = np.arange(n)[None, :] + h * np.arange(m)[:, None]
        X = np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1)) + 1e-6
        L = np.log(X)
        v = L.ravel() - L.mean()
        return v / (np.linalg.norm(v) + 1e-12)
    u, v = _s(a), _s(b)
    if u is None or v is None or len(u) != len(v):
        return float("nan")
    return float(u @ v)


def env(x, hop=480):
    """10ms RMS 包络。"""
    m = len(x) // hop
    if m < 4:
        return None
    e = np.sqrt((x[:m * hop].reshape(m, hop) ** 2).mean(1)) + 1e-9
    return e


def env_corr(a, b):
    u, v = env(a), env(b)
    if u is None or v is None or len(u) != len(v):
        return float("nan")
    u = u - u.mean(); v = v - v.mean()
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v) / d if d > 1e-12 else float("nan")


def wave_ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b) / d if d > 1e-12 else 0.0


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
srch = int(SR * 0.03)

for tag, wavname in (("v82", "v82_mono"), ("nl_mono", "nl_mono")):
    tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_%s_v3d.json" % tag),
                        encoding="utf-8"))
    mix = np.asarray(read_wav(os.path.join(AI, "data", "atoms", wavname + ".wav"), mono=True),
                     np.float32)
    evs = sorted([e for e in tl if e["cands"][0]["cat"] != "其他"],
                 key=lambda e: -e["cands"][0]["gain"])[:NEV]
    print("=" * 100)
    print("%s：前 %d 个音效事件 —— 波形 / 包络 / 频谱 三种一致性" % (wavname, len(evs)))
    print("=" * 100)
    print("%8s %-22s %10s %10s %10s %8s" % ("t(s)", "模板", "波形NCC", "包络相关", "频谱余弦", "延迟"))
    rows = []
    for e in evs:
        t = e["t"]; c0 = e["cands"][0]
        aid = mp.get(int(c0["class"]))
        if aid is None:
            continue
        x = lib.get48k(int(aid))[:int(2.0 * SR)].astype(np.float64)
        if len(x) < 4096:
            continue
        pad = int(0.1 * SR)
        a = max(0, int(t * SR) - pad)
        seg = mix[a: a + len(x) + 2 * pad].astype(np.float64)
        cc = int(t * SR) - a
        if len(seg) < len(x) + 2 * pad - 8 or cc < 0:
            continue
        # 用【包络相关】对齐（波形会因编码损伤而找不到峰，包络更稳）
        best = (-9, 0)
        for lag in range(-srch, srch + 1, 8):
            p = cc + lag
            if p < 0 or p + len(x) > len(seg):
                continue
            r = env_corr(x, seg[p:p + len(x)])
            if not np.isnan(r) and r > best[0]:
                best = (r, lag)
        lag = best[1]; w = seg[cc + lag: cc + lag + len(x)]
        wn = wave_ncc(x, w); ec = env_corr(x, w); sc = spec_cos(x, w)
        rows.append((abs(wn), ec, sc))
        print("%8.2f %-22s %10.3f %10.3f %10.3f %7.1fms" % (
            t, c0["template"][:22], wn, ec, sc, lag / SR * 1000))
    if rows:
        r = np.array(rows)
        print("-" * 100)
        print("  中位:  波形NCC %.3f   包络相关 %.3f   频谱余弦 %.3f"
              % (np.median(r[:, 0]), np.nanmedian(r[:, 1]), np.nanmedian(r[:, 2])))
        print("  >=0.7 的比例:  波形 %.0f%%   包络 %.0f%%   频谱 %.0f%%" % (
            100 * (r[:, 0] >= 0.7).mean(), 100 * np.nanmean(r[:, 1] >= 0.7),
            100 * np.nanmean(r[:, 2] >= 0.7)))
