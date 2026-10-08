"""构建模板库（**变长流**，统一重采样到 SR）。

为什么是变长流而不是定长矩阵 [K, LMAX]：
  * 定长矩阵的存储按【最长素材】计费：LMAX=25 s 时约 5.3 GB，而 90% 的素材远短于它。
  * 变长流按【实际总时长】计费，且**没有任何长度截断** —— 长语音、长音效都能作为模板。
  * 取模板时按 offs/lens 切片，与长音池（long_pool）同一套做法。

输出（都写在 ML 下）：
  bank_pool.npy   float16，所有模板首尾相接
  bank_offs.npy   int64，每条在池中的起始采样
  bank_lens.npy   int32，每条长度（采样）
  bank_index.json list，顺序 = 池顺序 = class id（name/bundle/group/kind/dur/sr）
  bank_meta.json  {sr, n, samples, total_min, dur_min/dur_p50/dur_p90/dur_max,
                   groups, filter, src_sr_hist, n_err}

用法：
    python build_bank.py                       # SFX + 战斗语音
    python build_bank.py --no-voice            # 只要 SFX
    python build_bank.py --min-dur 0.05 --max-dur 0
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

ML = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ML), "data", "atoms")   # 数据根: sfx/ long44/ 录音
sys.path.insert(0, D)
import alab                                                     # noqa: E402

SR = 44100
SFX_GROUPS = ["player", "root", "custom_se", "enemy"]
VOICE_TAGS = ["voice_battle_jp", "voice_battle_cn"]


def probe(path: str):
    """只读 wav 头 -> (采样数, 采样率)。"""
    with wave.open(path, "rb") as w:
        return int(w.getnframes()), int(w.getframerate())


def resample_to(x: np.ndarray, sr_in: int, sr_out: int, lp: float | None = None):
    """FFT 低通 + 线性插值重采样（不依赖额外库）。"""
    if sr_in == sr_out:
        return x
    if lp is None:
        lp = min(sr_in, sr_out) * 0.475
    n = len(x)
    nf = 1 << int(np.ceil(np.log2(max(n, 64))))
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1.0 / sr_in)
    X[f > lp] = 0.0
    y = np.fft.irfft(X, nf)[:n]
    return np.interp(np.arange(int(len(y) * sr_out / sr_in)) / sr_out,
                     np.arange(len(y)) / sr_in, y).astype(np.float32)


def collect(use_voice: bool, min_dur: float, max_dur: float):
    items = []
    for g in SFX_GROUPS:
        p = os.path.join(D, "sfx", "index_%s.json" % g)
        if not os.path.exists(p):
            continue
        for v in json.load(open(p, encoding="utf-8")).values():
            if not v.get("file") or not os.path.exists(v["file"]):
                continue
            items.append(dict(v, group=g, kind="sfx"))
    if use_voice:
        for tag in VOICE_TAGS:
            p = os.path.join(D, "long44", "index_%s.json" % tag)
            if not os.path.exists(p):
                print("  [跳过] 没有 %s（先跑 x_extract_long44_mp.py）" % p, flush=True)
                continue
            for v in json.load(open(p, encoding="utf-8")).values():
                if not v.get("file") or not os.path.exists(v["file"]):
                    continue
                items.append(dict(v, group=v.get("lang", "voice"), kind="voice_battle"))
    out, skipped = [], 0
    for v in items:
        d = float(v.get("dur") or 0.0)
        if d < min_dur or (max_dur > 0 and d > max_dur):
            skipped += 1
            continue
        out.append(v)
    return out, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("build_bank")
    ap.add_argument("--no-voice", action="store_true", help="只收 SFX，不收战斗语音")
    ap.add_argument("--min-dur", type=float, default=0.05, help="短于该时长的素材丢弃（秒）")
    ap.add_argument("--max-dur", type=float, default=0.0, help="长于该时长的丢弃（0 = 不限）")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, skipped = collect(not a.no_voice, a.min_dur, a.max_dur)
    if not items:
        raise SystemExit("没有可用素材：检查 sfx/index_*.json 与 long44/index_voice_battle_*.json")
    print("模板候选 %d 条（按时长过滤掉 %d 条）" % (len(items), skipped), flush=True)

    # 预扫描：算出重采样后的总采样数，一次性开 memmap
    t0 = time.time()
    keep, total = [], 0
    for i, v in enumerate(items):
        try:
            nfr, sr_in = probe(v["file"])
        except Exception:
            continue
        n_out = int(round(nfr * SR / float(sr_in)))
        if n_out <= 0:
            continue
        keep.append((v, sr_in, n_out))
        total += n_out
        if (i + 1) % 2000 == 0:
            print("  扫描 %d/%d  %.0fs" % (i + 1, len(items), time.time() - t0), flush=True)
    K = len(keep)
    print("预扫描完成：%d 条，合计 %.1f 分钟 -> 池 %.0f MB (fp16)"
          % (K, total / SR / 60.0, total * 2 / 1e6), flush=True)

    pool = np.lib.format.open_memmap(os.path.join(ML, "bank_pool.npy"), mode="w+",
                                     dtype=np.float16, shape=(total,))
    offs = np.zeros(K, dtype=np.int64)
    lens = np.zeros(K, dtype=np.int32)
    index = []
    pos, n_ok, n_err = 0, 0, 0
    srs = {}
    for i, (v, sr_in, n_out) in enumerate(keep):
        try:
            x, sr = alab.wav_read(v["file"], mono=True)
        except Exception:
            n_err += 1
            offs[i], lens[i] = pos, 0
            index.append({"name": v.get("name", ""), "bundle": v.get("bundle", ""),
                          "group": v.get("group", ""), "kind": v.get("kind", ""),
                          "dur": 0.0, "sr": int(sr_in)})
            continue
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        y = resample_to(x, int(sr), SR)
        if len(y) > n_out:
            y = y[:n_out]
        pool[pos:pos + len(y)] = y.astype(np.float16)
        offs[i], lens[i] = pos, len(y)
        pos += len(y)
        srs[int(sr)] = srs.get(int(sr), 0) + 1
        n_ok += 1
        index.append({"name": v.get("name", ""), "bundle": v.get("bundle", ""),
                      "group": v.get("group", ""), "kind": v.get("kind", ""),
                      "dur": round(len(y) / SR, 4), "sr": int(sr)})
        if (i + 1) % 2000 == 0:
            print("  写入 %d/%d  %.0fs" % (i + 1, K, time.time() - t0), flush=True)
    pool.flush()
    del pool
    np.save(os.path.join(ML, "bank_offs.npy"), offs)
    np.save(os.path.join(ML, "bank_lens.npy"), lens)
    json.dump(index, open(os.path.join(ML, "bank_index.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    # 时长统计按【实际写进池子】的 lens 算，不是命令行过滤值：过滤值只说明门槛，
    # 读者真正要知道的是库里最长/中位多少秒（变长流下这个数可以很大）。
    lv = lens[lens > 0].astype(np.float64) / SR
    q = np.percentile(lv, [50, 90]) if lv.size else np.zeros(2)
    groups: dict = {}
    for it in index:
        g = str(it.get("group") or it.get("kind") or "?")
        groups[g] = groups.get(g, 0) + 1
    json.dump({"sr": SR, "n": int(K), "samples": int(pos),
               "total_min": round(pos / SR / 60.0, 2),
               "dur_min": round(float(lv.min()), 4) if lv.size else 0.0,
               "dur_max": round(float(lv.max()), 4) if lv.size else 0.0,
               "dur_p50": round(float(q[0]), 4), "dur_p90": round(float(q[1]), 4),
               "groups": groups,
               "filter": {"min_dur": a.min_dur, "max_dur": a.max_dur},
               "src_sr_hist": srs, "n_err": int(n_err)},
              open(os.path.join(ML, "bank_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("DONE K=%d  ok=%d  err=%d  源采样率分布=%s  %.0fs" % (K, n_ok, n_err, srs, time.time() - t0),
          flush=True)
    print("  池 %.0f MB，最长 %.2f s，合计 %.1f 分钟"
          % (pos * 2 / 1e6, float(lens.max()) / SR if K else 0.0, pos / SR / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
