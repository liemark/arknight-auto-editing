"""后处理：识别结果（timeline）→ 反相轨 + 报告。

数据流：

    train/infer.py  ──>  timeline_*.json          （识别 + NNLS 增益）
                              │
                              │  class = 模板下标（0..6575, 16 kHz 的 bank_clean.npy）
                              ▼
    本模块 postproc.render
        1. class → 本包 atom id     用 (name, bundle) 映射；实测 **6576/6576 全部命中、无歧义**
        2. 增益                    默认用本包的渲染链**重新最小二乘拟合**（timeline 里的 gain 口径
                                   不同：bank_clean 的模板没做 RMS 归一化，而 render_events
                                   会把原子归一化到 -34 dB）
        3. 渲染反相轨              复用 models/synthesize.py::render_events(invert=True)
        4. 报告                    复用 detector/cancel.py::psr_db

**为什么渲染必须走本包**：识别的模型与模板库是 16 kHz 的（`core.py: SR=16000`，
`bank_clean.npy` 也是 16 kHz 重采样）。48 kHz 原始录音里 8–24 kHz 那一段，用 16 kHz
模板永远抵消不掉。本包的 `atomlib` 是**原始采样率**（16k/44.1k/48k 原样混存）的池子，
渲染链全程 48 kHz，所以只有走这边才可能真正做到"近乎静音"。

    python -m audio_inverse.postproc.render ^
        --timeline data/atoms/timeline_ui_demo_mono48k_labA_s60000.json ^
        --wav data/atoms/real/ui_demo_mono48k.wav ^
        --out-dir out/lab_post
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

from ..atomlib import AtomLib
from ..audio import SR, read_wav, resample
from ..config import PKG_ROOT, load_cfg
from ..detector.cancel import psr_db
from ..manifest import Manifest
from ..models.synthesize import Event, render_events, band_psr


def _map_templates(cfg) -> dict[int, int]:
    """模板下标 -> 本包 atom id。键 = (name, bundle, lang) 小写，回退 (name, bundle)。

    语音必须带 lang：日配和中配的同一条语音 name/bundle 完全相同（如 CN_017 +
    char_002_amiya 在 jp/cn 各有一条），只用 (name,bundle) 会让 6853 个键重复、
    13706 个语音模板全部判为"有歧义"而无法渲染（SFX 不受影响，它们没有重复键）。
    """
    bi = json.load(open(os.path.join(PKG_ROOT, "train", "bank_index.json"),
                        encoding="utf-8"))
    man = Manifest.load(os.path.join(cfg.abspath("data_root"), "manifest.json"))
    proj3: dict[tuple[str, str, str], int] = {}
    dup3: set[tuple[str, str, str]] = set()
    proj2: dict[tuple[str, str], int] = {}
    dup2: set[tuple[str, str]] = set()
    for a in man.atoms:
        n, b = str(a.name).lower(), str(a.bundle).lower()
        lg = str(a.lang or "").lower()
        proj3.setdefault((n, b, lg), a.id)
        if (n, b, lg) in proj3 and proj3[(n, b, lg)] != a.id:
            dup3.add((n, b, lg))
        if (n, b) in proj2:
            dup2.add((n, b))
        proj2.setdefault((n, b), a.id)
    out: dict[int, int] = {}
    miss: list[int] = []
    for i, v in enumerate(bi):
        n = str(v.get("name", "")).lower()
        b = str(v.get("bundle", "")).lower()
        g = str(v.get("group", "")).lower()
        lg = str(v.get("lang") or (g if g in ("jp", "cn") else "")).lower()
        if (n, b, lg) in proj3 and (n, b, lg) not in dup3:
            out[i] = proj3[(n, b, lg)]
        elif (n, b) in proj2 and (n, b) not in dup2:
            out[i] = proj2[(n, b)]
        else:
            miss.append(i)
    print("模板映射: %d/%d 命中" % (len(out), len(bi)), end="")
    if miss:
        print("，未命中 %d 个（前 5: %s）" % (len(miss), miss[:5]))
    else:
        print("（全部命中，无歧义）")
    return out


def _load_wav48k(path: str) -> np.ndarray:
    x = read_wav(path, mono=True)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sr = int(sf.info(path).samplerate)
    if sr != SR:
        x = resample(x, sr, SR).astype(np.float32)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("audio_inverse.postproc.render")
    ap.add_argument("--timeline", required=True, help="train/infer.py 产出的 timeline_*.json")
    ap.add_argument("--wav", required=True, help="同一条录音（会被升采样到 48k 对齐）")
    ap.add_argument("--out-dir", default="out/lab_post")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--gain-mode", default="refit", choices=["refit", "timeline"],
                    help="refit = 用本项目渲染链重新最小二乘拟合（默认，推荐）；"
                         "timeline = 直接用 timeline 里 NNLS 的 gain（两侧电平口径不同，仅作对照）")
    ap.add_argument("--top-per-event", type=int, default=4,
                    help="每个 onset 最多用几个候选（timeline 里每条最多 4 个）")
    ap.add_argument("--gain-thr", type=float, default=0.05, help="线性增益低于它的事件丢掉")
    ap.add_argument("--max-events", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--fit-win", type=float, default=0.5, help="最小二乘拟合窗（秒）")
    ap.add_argument("--overrides", nargs="*", default=[])
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_cfg(a.config, list(a.overrides))
    data_root = cfg.abspath("data_root")
    atom_root = cfg.abspath("atom_root")
    lib = AtomLib(os.path.join(data_root, "atomlib"))

    tl = json.load(open(a.timeline, encoding="utf-8"))
    print("timeline: %s  %d 个事件" % (os.path.basename(a.timeline), len(tl)))
    tmap = _map_templates(cfg)

    x = _load_wav48k(a.wav)
    n_total = x.size
    print("录音: %s  %.1fs -> 48k / %d 采样" % (os.path.basename(a.wav), n_total / SR, n_total))

    tl = sorted(tl, key=lambda e: float(e["t"]))
    if a.max_events:
        tl = tl[:a.max_events]

    resid = x.copy()
    evs: list[Event] = []
    n_nomap = 0
    t0 = time.time()
    for e in tl:
        t = float(e["t"])
        cands = []
        for c in e.get("cands", [])[:max(1, a.top_per_event)]:
            aid = tmap.get(int(c["class"]))
            if aid is None:
                n_nomap += 1
                continue
            cands.append((aid, float(c.get("gain", 0.0)), float(c.get("cos", 0.0)),
                          str(c.get("template", ""))))
        if not cands:
            continue
        j = int(round(t * SR))
        if j >= n_total:
            continue
        best = None
        for aid, g_old, cos_v, tname in cands:
            one = Event(atom_id=int(aid), delay=t, gain_db=0.0, score=cos_v, source="lab")
            tmpl = render_events(lib, [one], n_total, sr=SR, invert=False)
            b = min(n_total, j + int(a.fit_win * SR))
            if b <= j:
                continue
            A = tmpl[j:b].astype(np.float64)
            S = resid[j:b].astype(np.float64)
            den = float(A @ A)
            if den <= 1e-12:
                continue
            if a.gain_mode == "timeline":
                # timeline 的 gain 是"对未归一化模板"的线性系数；本项目渲染已把原子归一化到 -34 dB，
                # 所以这里换算一次电平差再比能量。换算只是近似，要精确请用 refit。
                gain = max(0.0, g_old)
            else:
                gain = max(0.0, float(S @ A) / den)
            score = gain * gain * den
            if best is None or score > best[0]:
                best = (score, aid, gain, cos_v, tname)
        if best is None or best[2] < a.gain_thr:
            continue
        gain_db = 20.0 * np.log10(best[2] + 1e-12)
        evs.append(Event(atom_id=int(best[1]), delay=t, gain_db=float(gain_db),
                         score=float(best[3]), source="lab"))
        one = render_events(lib, [Event(atom_id=int(best[1]), delay=t, gain_db=float(gain_db))],
                            n_total, sr=SR, invert=False)
        resid = resid - one

    os.makedirs(a.out_dir, exist_ok=True)
    cancel = render_events(lib, evs, n_total, sr=SR, invert=True) if evs \
        else np.zeros(n_total, dtype=np.float32)
    residual = x + cancel
    sf.write(os.path.join(a.out_dir, "cancel.wav"), np.clip(cancel, -1, 1), SR, subtype="PCM_16")
    sf.write(os.path.join(a.out_dir, "residual.wav"), np.clip(residual, -1, 1), SR, subtype="PCM_16")

    gains = [e.gain_db for e in evs]
    dur = n_total / SR
    rep = {
        "timeline": os.path.abspath(a.timeline), "wav": os.path.abspath(a.wav),
        "gain_mode": a.gain_mode, "seconds": round(dur, 2),
        "events_in_timeline": len(tl), "events_rendered": len(evs),
        "events_per_sec": round(len(evs) / max(dur, 1e-9), 3),
        "no_mapping": n_nomap,
        "gain_db_median": (round(float(np.median(gains)), 2) if gains else None),
        "gain_db_p10": (round(float(np.percentile(gains, 10)), 2) if gains else None),
        "psr_db": round(float(psr_db(x, cancel)), 2),
        "band_psr_db": {k: round(float(v), 2) for k, v in
                        band_psr(x, cancel, sr=SR).items()} if evs else {},
    }
    with open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"report": rep,
                   "events": [{"t": e.delay, "atom_id": e.atom_id, "gain_db": e.gain_db,
                               "score": e.score} for e in evs]},
                  f, ensure_ascii=False, indent=1)

    print("\n===== 后处理报告（渲染链 = 本项目 48 kHz）=====")
    for k, v in rep.items():
        print("  %-20s %s" % (k, v))
    print("\n  用时 %.0fs -> %s" % (time.time() - t0, os.path.abspath(a.out_dir)))
    print("    cancel.wav    反相轨（可直接与原音相加）")
    print("    residual.wav  残差 = 原音 + 反相轨 —— **直接听这个**")
    print("    report.json   本表 + 逐事件增益")
    if rep["events_per_sec"] > 5:
        print("\n  ⚠️ 事件密度 %.1f 个/秒，偏高。域不匹配时最典型的症状是 onset 概率饱和"
              "（阈值贴到 0.90 上限、上千个事件、NNLS 增益 0.00~0.11）——"
              "先跑 ml/onset_diag.py 确认是不是训练背景与真实录音的域差。"
              % rep["events_per_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
