"""构建长音池（BGM / 环境 / 剧情 / 长音效 + 干员战斗语音），全部重采样到 SR。

与 build_bank.py 的分工：模板库放【一个事件一条】的素材（供识别，长度不限），
长音池放【成段连续】的素材（供 --bg-mode long 当背景床），两者的池格式相同：

    long_pool.npy     float16，所有条目首尾相接
    long_offs.npy     int64，每条在池中的起始采样
    long_lens.npy     int32，每条长度（采样）
    long_meta.json    list，与池顺序对齐的元数据（kind / lang / loop / dur / voiceTitle）
    long_summary.json {sr, n, samples, total_min, kinds, cap, bad}（人读的概况）

输入：<数据根>/long44/index_long.json（44.1 kHz 版；缺失时回退 long/index_long.json）。
构建顺序：解包 refs -> build_long.py -> synth.py bg_mode="long"
"""
import os, sys, json, time
import numpy as np
ML = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(ML), "data", "atoms")
sys.path.insert(0, D)
import alab

SR = 44100
CAP = 120.0

ip = os.path.join(D, "long44", "index_long.json")      # 44.1 kHz 版
if not os.path.exists(ip):
    ip = os.path.join(D, "long", "index_long.json")      # 回退：旧的 16 kHz 版
assert os.path.exists(ip), "no long index - run the unpacker (refs) or `assets long-index` first"
idx = json.load(open(ip, encoding="utf-8"))
idx = [it for it in idx if it.get("file") and os.path.exists(it["file"])]
n_samp = [int(min(float(it.get("dur") or 0.0), CAP) * SR) for it in idx]
idx = [it for it, L in zip(idx, n_samp) if L > 0]
n_samp = [L for L in n_samp if L > 0]
total = int(sum(n_samp))
print("items %d  %.1f M samples  %.0f MB fp16  %.1f min of audio"
      % (len(idx), total / 1e6, total * 2 / 1e6, total / SR / 60), flush=True)
assert total > 0

pool = np.lib.format.open_memmap(os.path.join(ML, "long_pool.npy"), mode="w+",
                                 dtype=np.float16, shape=(total,))
offs = np.zeros(len(idx), dtype=np.int64)
lens = np.zeros(len(idx), dtype=np.int32)
pos = 0; t0 = time.time(); bad = 0
for i, it in enumerate(idx):
    L = n_samp[i]
    offs[i] = pos
    try:
        x, sr = alab.wav_read(it["file"], mono=True)
    except Exception:
        bad += 1; pos += L; continue
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR and len(x) > 8:
        x = alab.bandpass(x, sr, 30.0, min(sr, SR) * 0.475)
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR,
                      np.arange(len(x)) / sr, x).astype(np.float32)
    x = x[:L]
    pool[pos:pos + len(x)] = x.astype(np.float16)
    lens[i] = len(x)
    pos += L
    if (i + 1) % 1000 == 0:
        print("  %5d/%d  %.0fs" % (i + 1, len(idx), time.time() - t0), flush=True)
pool.flush()
np.save(os.path.join(ML, "long_offs.npy"), offs)
np.save(os.path.join(ML, "long_lens.npy"), lens)
json.dump([{k: it.get(k) for k in ("name", "bundle", "kind", "lang", "loop", "dur",
                                   "voiceIndex", "voiceTitle", "placeType")} for it in idx],
          open(os.path.join(ML, "long_meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
# 概况单独一个文件：long_meta.json 必须与池顺序严格对齐（list），不能塞头部字段。
kinds = {}
for m in json.load(open(os.path.join(ML, "long_meta.json"), encoding="utf-8")):
    k = str(m.get("kind") or "?")
    kinds[k] = kinds.get(k, 0) + 1
lv = np.asarray(lens, dtype=np.float64) / SR
json.dump({"sr": SR, "n": len(idx), "samples": int(pos),
           "total_min": round(pos / SR / 60.0, 2), "kinds": kinds,
           "dur_p50": round(float(np.percentile(lv, 50)), 3) if lv.size else 0.0,
           "dur_max": round(float(lv.max()), 3) if lv.size else 0.0,
           "cap": CAP, "bad": int(bad), "index": os.path.basename(os.path.dirname(ip))
           + "/" + os.path.basename(ip)},
          open(os.path.join(ML, "long_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE bad=%d  %.0fs" % (bad, time.time() - t0), flush=True)
