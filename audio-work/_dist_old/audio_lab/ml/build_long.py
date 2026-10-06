"""Build the 16 kHz long/looping background pool (BGM + ambience + operator battle voice).

Companion to build_bank.py. The template bank hard-caps clips at 2.5 s, so long material
cannot live in it; it is kept as a separate flat stream here:

    long_pool.npy   float16, all items concatenated
    long_offs.npy   int64  start sample of each item
    long_lens.npy   int32  length of each item
    long_meta.json  aligned metadata (kind / lang / loop / dur / voiceTitle)

Build order: x_extract_long.py all  ->  build_long.py  ->  synth.py bg_mode="long"
"""
import os, sys, json, time
import numpy as np
ML = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(ML)                      # 工作根目录（sfx/ long/ 与 alab.py 都在这层）
sys.path.insert(0, D)
import alab

SR = 16000
CAP = 120.0

ip = os.path.join(D, "long", "index_long.json")
assert os.path.exists(ip), "run x_extract_long.py refs first"
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
        x = alab.bandpass(x, sr, 30.0, 7500.0)
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
print("DONE bad=%d  %.0fs" % (bad, time.time() - t0), flush=True)
