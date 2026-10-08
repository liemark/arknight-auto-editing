"""导出【试听用】的单轨：把 timeline 的匹配结果铺成能直接听的东西。

给"匹配对不对"这个问题一个耳朵可判的答案：把识别到的原子单独铺一遍，
与原音逐段 A/B。若 `matched` 听起来和原音不是同一个声音，那么 PSR≈0 就不是渲染的锅，
而是"录音里根本没有这些素材"。

产物（默认都写 `--out-dir`）：

| 文件 | 内容 | 听什么 |
|---|---|---|
| `matched.wav` | 只含匹配到的原子（拟合增益/延迟，正相） | 匹配出来的到底是什么 |
| `cuts.wav` | **原声直出**：把每个匹配事件所在的录音片段剪出来按时间拼接 | 检测器"以为"的事件在原音里长什么样 |
| `pairs.wav` | 最自信的前 N 个事件，逐条「原声片段 → 重建片段」交替 | 一条一条对比，最容易判对错 |
| `resid.wav` | 原音 + 反相轨 = 抵消后剩下的 | 抵消了多少（PSR 见 report.json） |
| `orig.wav` | 原音（只在给了 `--window` 时写） | A/B 的另一半 |
| `ab.wav` | orig / matched 交替拼接 | 整段 A/B，最省事 |
| `cuts/cut_0001_...wav` | `--split-cuts` 时逐事件一个文件（配 `--pair` 同写重建） | 挑单条复盘 |

    # 全片：只铺匹配到的原子（正相）
    python -m audio_inverse.postproc.export --timeline data/atoms/timeline_x.json ^
        --wav data/atoms/x.wav --out-dir out/export_x --mode matched

    # 原声直出：所有匹配事件的原声片段拼一条，外加前 12 条的 A/B 对照
    python -m audio_inverse.postproc.export --timeline ... --wav ... --out-dir ... ^
        --mode all --cut-pad 0.05 --pair-top 12

    # 指定窗口做 A/B 交替（原音 5s / 重建 5s）
    python -m audio_inverse.postproc.export --timeline ... --wav ... --out-dir ... ^
        --window 110 140 --interleave --excerpt 5

    # 只保留波形真的对得上的事件（配合 --search-ms 做采样级延迟搜索）
    python -m audio_inverse.postproc.export ... --search-ms 30 --min-ncc 0.5 --limit-events 200

注意 gain 口径：timeline 里的 gain 是"对未归一化模板"的线性系数，而本包渲染会把原子
RMS 归一化到 -34 dBFS，所以默认用 `--gain-mode refit`（在录音上重新最小二乘拟合），
与 `postproc.render` 的默认口径一致；只有要和历史 timeline 数字对比时才用 `timeline`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

from ..atomlib import AtomLib
from ..audio import read_wav, resample, rms_normalize_db
from ..config import PKG_ROOT, load_cfg
from ..models.synthesize import Event, render_events
from .render import _map_templates

SR = 48000


def _load_timeline(path: str) -> list:
    tl = json.load(open(path, encoding="utf-8"))
    return sorted(tl, key=lambda e: float(e["t"]))


def _best_lag(x: np.ndarray, w: np.ndarray, srch: int):
    """在 w 内对零均值 x 做 ±srch 采样的对齐搜索 -> (lag, ncc, gain)。

    w 的中心对应"timeline 报告的延迟"。用 FFT 一次算出所有 lag 的互相关，
    再用前缀和得到每个 lag 处窗口的均值/能量，从而得到逐 lag 的归一化相关。
    """
    L = len(x)
    xc = x - x.mean()
    nx = float(np.linalg.norm(xc))
    if nx < 1e-9 or len(w) < L + 2 * srch:
        return 0, 0.0, 0.0
    n = len(w)
    nf = 1 << int(np.ceil(np.log2(n + L)))
    C = np.fft.irfft(np.fft.rfft(w, nf) * np.conj(np.fft.rfft(xc, nf)), nf)[:2 * srch + 1]
    cs = np.concatenate([[0.0], np.cumsum(w)])
    cs2 = np.concatenate([[0.0], np.cumsum(w * w)])
    a = np.arange(2 * srch + 1)
    s1 = cs[a + L] - cs[a]
    s2 = cs2[a + L] - cs2[a]
    var = np.maximum(s2 - s1 * s1 / L, 0.0)
    r = C / (np.sqrt(var) * nx + 1e-12)
    k = int(np.argmax(np.abs(r)))
    return k - srch, float(r[k]), float(C[k]) / (nx * nx)


def build_events(lib: AtomLib, tmap: dict, tl: list, mix: np.ndarray, *,
                 top_per_event: int = 4, fit_win: float = 2.0, gain_mode: str = "refit",
                 gain_thr: float = 0.0, search_ms: float = 0.0, min_ncc: float = 0.0,
                 limit_events: int = 0):
    """逐个事件拟合 (延迟, 增益) -> (事件表, 逐事件诊断, 正相重建, 反相轨)。"""
    n_total = mix.size
    srch = int(SR * search_ms / 1000.0)
    rec = np.zeros(n_total, dtype=np.float64)
    cancel = np.zeros(n_total, dtype=np.float64)
    evs, rows = [], []
    for e in tl:
        t = float(e["t"])
        j0 = int(round(t * SR))
        if j0 >= n_total:
            continue
        best = None
        for c in (e.get("cands") or [])[:max(1, top_per_event)]:
            aid = tmap.get(int(c.get("class", -1)))
            if aid is None:
                continue
            x, asr = lib.raw(int(aid))
            x = np.asarray(rms_normalize_db(x, -34.0), dtype=np.float64)
            if asr != SR:
                x = np.asarray(resample(x.astype(np.float32), asr, SR), np.float64)
            if x.size < 512:
                continue
            if x.size > int(fit_win * SR):      # 长模板截到拟合窗长: NCC 才不会被远处背景稀释
                x = x[:int(fit_win * SR)]
            if gain_mode == "timeline":
                lag, ncc, gain = 0, float("nan"), max(0.0, float(c.get("gain", 0.0)))
            elif srch > 0:
                a = max(0, j0 - srch)
                w = mix[a:min(n_total, j0 + x.size + srch)].astype(np.float64)
                if w.size < x.size + 2 * (j0 - a):
                    continue
                lag, ncc, gain = _best_lag(x, w, srch)
                lag = lag - (j0 - a)
            else:
                b = min(n_total, j0 + int(fit_win * SR), j0 + x.size)
                w = mix[j0:b].astype(np.float64)
                xw = x[:b - j0]
                xm = xw - xw.mean()
                wm = w - w.mean()
                nx = float(np.linalg.norm(xm))
                if nx < 1e-9:
                    continue
                gain = float(xm @ wm) / (nx * nx)
                ncc = float(xm @ wm) / (nx * float(np.linalg.norm(wm)) + 1e-12)
                lag = 0
            if not np.isfinite(ncc):
                ncc = 0.0
            if best is None or abs(ncc) > abs(best[0]):
                best = (ncc, gain, lag, x, aid, str(c.get("template", "")))
        if best is None:
            continue
        ncc, gain, lag, x, aid, tname = best
        if gain < gain_thr or abs(ncc) < min_ncc:
            continue
        j = j0 + lag
        if j < 0 or j >= n_total:
            continue
        b = min(n_total, j + x.size)
        xm = x[:b - j] - x[:b - j].mean()
        rec[j:b] += gain * xm
        cancel[j:b] -= gain * xm
        evs.append(Event(atom_id=int(aid), delay=j / SR, gain_db=float(20 * np.log10(gain + 1e-12)),
                         score=float(ncc), source="export"))
        rows.append({"t": round(t, 3), "t_fit": round(j / SR, 4), "template": tname,
                     "atom_id": int(aid), "dur": round(x.size / SR, 3),
                     "ncc": round(float(ncc), 3), "gain_db": round(float(20 * np.log10(gain + 1e-12)), 2),
                     "lag_ms": round(lag / SR * 1000, 2)})
        if limit_events and len(evs) >= limit_events:
            break
    return evs, rows, rec, cancel


def _psr(mix: np.ndarray, resid: np.ndarray) -> float:
    return float(10 * np.log10((mix ** 2).sum() / ((resid ** 2).sum() + 1e-12)))


def _write(path: str, x: np.ndarray) -> None:
    peak = float(np.abs(x).max())
    y = (x / peak * 0.9) if peak > 1e-9 else x
    sf.write(path, np.asarray(y, dtype=np.float32), SR, subtype="PCM_16")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("audio_inverse.postproc.export")
    ap.add_argument("--timeline", required=True, help="infer.py 产出的 timeline JSON")
    ap.add_argument("--wav", required=True, help="同一份真实录音")
    ap.add_argument("--out-dir", default="out/export", help="产物目录")
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--mode", default="all",
                    choices=["all", "matched", "resid", "ab"],
                    help="all=matched+resid(+窗口时加 orig/ab)")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="只导出这个时间窗(秒); 给了就额外写 orig.wav")
    ap.add_argument("--excerpt", type=float, default=5.0, help="A/B 交替时每段秒数")
    ap.add_argument("--interleave", action="store_true",
                    help="写 ab.wav：原音/重建交替拼接（最省事的 A/B）")
    ap.add_argument("--top-per-event", type=int, default=4, help="每个事件最多试几个候选原子")
    ap.add_argument("--gain-mode", default="refit", choices=["refit", "timeline"],
                    help="refit=在录音上重新最小二乘(与 render.py 默认一致); timeline=直接用 timeline 的 gain")
    ap.add_argument("--gain-thr", type=float, default=0.0, help="拟合增益低于此值的事件丢弃")
    ap.add_argument("--fit-win", type=float, default=2.0, help="refit 的拟合窗长(秒)")
    ap.add_argument("--search-ms", type=float, default=0.0,
                    help="采样级延迟搜索半径(ms). 0=直接用 timeline 的延迟(20ms 帧格)")
    ap.add_argument("--min-ncc", type=float, default=0.0,
                    help="只保留波形 |NCC| >= 该值的事件（配合 --search-ms 才公平）")
    ap.add_argument("--limit-events", type=int, default=0, help="最多铺多少个事件(0=全部)")
    ap.add_argument("--cut-pad", type=float, default=0.05,
                    help="原声直出时每段前后各留多少秒（onset 在 20ms 帧格上，留一点余量）")
    ap.add_argument("--pair-top", type=int, default=0,
                    help="写 pairs.wav：|NCC| 最高的前 N 个事件，逐条「原声片段 -> 重建片段」交替（0=不写）")
    ap.add_argument("--gap", type=float, default=0.15, help="拼接各段之间的静音间隔（秒）")
    ap.add_argument("--split-cuts", action="store_true",
                    help="把原声片段逐事件写成 cuts/cut_0001_....wav（配 --pair 时同时写重建）")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_cfg(a.config, list(a.overrides))
    lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
    tmap = _map_templates(cfg)
    mix = np.asarray(read_wav(a.wav, mono=True), dtype=np.float32)
    if mix.ndim > 1:
        mix = mix.mean(axis=1)
    tl = _load_timeline(a.timeline)
    print("timeline %d 个事件  录音 %.1fs @%dHz" % (len(tl), mix.size / SR, SR))

    evs, rows, rec, cancel = build_events(
        lib, tmap, tl, mix, top_per_event=a.top_per_event, fit_win=a.fit_win,
        gain_mode=a.gain_mode, gain_thr=a.gain_thr, search_ms=a.search_ms,
        min_ncc=a.min_ncc, limit_events=a.limit_events)
    resid = mix.astype(np.float64) + cancel
    psr = _psr(mix.astype(np.float64), resid)
    ncc = np.array([r["ncc"] for r in rows]) if rows else np.zeros(1)
    print("铺了 %d 个事件（候选未命中/被阈值筛掉的未计）  波形 NCC 中位 %.3f  (>=0.5 的 %d 个)"
          % (len(evs), float(np.median(np.abs(ncc))), int((np.abs(ncc) >= 0.5).sum())))
    print("全片 PSR %.2f dB   重建能量/原音能量 %.1f dB"
          % (psr, 10 * np.log10((rec ** 2).sum() / ((mix ** 2).sum() + 1e-12) + 1e-12)))

    os.makedirs(a.out_dir, exist_ok=True)
    lo, hi = (0, mix.size) if a.window is None else (int(a.window[0] * SR), min(mix.size, int(a.window[1] * SR)))
    written = []
    if a.mode in ("all", "matched"):
        _write(os.path.join(a.out_dir, "matched.wav"), rec[lo:hi])
        written.append("matched.wav")
    if a.mode in ("all", "resid"):
        _write(os.path.join(a.out_dir, "resid.wav"), resid[lo:hi])
        written.append("resid.wav")
    if a.window is not None:
        _write(os.path.join(a.out_dir, "orig.wav"), mix[lo:hi].astype(np.float64))
        written.append("orig.wav")
    if a.interleave and a.window is not None:
        n = int(a.excerpt * SR)
        parts = []
        for k in range(lo, hi, n):
            parts.append(mix[k:k + n].astype(np.float64))
            parts.append(rec[k:k + n])
        _write(os.path.join(a.out_dir, "ab.wav"), np.concatenate(parts) if parts else rec[:0])
        written.append("ab.wav（原音/重建交替）")
    # ---- 原声直出：把每个匹配事件所在的录音片段剪出来 ----
    if rows:
        gap = np.zeros(max(1, int(a.gap * SR)), dtype=np.float64)
        segs, recs, meta = [], [], []
        for i, r in enumerate(rows, 1):
            if a.window is not None and not (a.window[0] - 1.0 <= r["t"] <= a.window[1] + 1.0):
                continue
            L = int(r["dur"] * SR)
            p = int(a.cut_pad * SR)
            s = max(0, int(r["t_fit"] * SR) - p)
            b = min(mix.size, s + L + 2 * p)
            if b - s < 256:
                continue
            segs.append(mix[s:b].astype(np.float64))
            recs.append(rec[s:b])
            meta.append((i, r))

        def _cat(xs):
            out = []
            for k, x in enumerate(xs):
                if k:
                    out.append(gap)
                out.append(x)
            return np.concatenate(out) if out else rec[:0]

        if segs:
            _write(os.path.join(a.out_dir, "cuts.wav"), _cat(segs))
            written.append("cuts.wav（原声直出 %d 段，按时间顺序）" % len(segs))
            if a.pair_top > 0:
                order = sorted(range(len(segs)), key=lambda k: -abs(meta[k][1]["ncc"]))[:a.pair_top]
                pair = []
                for k in order:
                    pair += [segs[k], gap, recs[k], gap]
                _write(os.path.join(a.out_dir, "pairs.wav"), np.concatenate(pair))
                written.append("pairs.wav（|NCC| 最高的 %d 条：原声 -> 重建 交替）" % len(order))
            if a.split_cuts:
                d = os.path.join(a.out_dir, "cuts")
                os.makedirs(d, exist_ok=True)
                for k, (i, r) in enumerate(meta):
                    nm = "t%08.2f_%s" % (r["t"], r["template"][:28].replace("/", "_"))
                    _write(os.path.join(d, "cut_%04d_%s.wav" % (i, nm)), segs[k])
                    _write(os.path.join(d, "rec_%04d_%s.wav" % (i, nm)), recs[k])
                written.append("cuts/（%d 对 cut_/rec_ 单条文件）" % len(meta))
    json.dump({"timeline": a.timeline, "wav": a.wav, "gain_mode": a.gain_mode,
               "search_ms": a.search_ms, "min_ncc": a.min_ncc,
               "n_events_in_timeline": len(tl), "n_events_used": len(evs),
               "psr_db": round(psr, 3), "events": rows},
              open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    written.append("report.json（逐事件 gain/NCC/延迟）")
    for f in written:
        print("  -> %s" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
