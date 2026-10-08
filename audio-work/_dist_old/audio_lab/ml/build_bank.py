"""Build the 16 kHz SFX template bank (clean + AAC-degraded) for the learned detector."""
import os, sys, json, time, wave, io
import numpy as np
ML = os.path.dirname(os.path.abspath(__file__))          # 本目录 = 模板库/长音池的存放处
D = os.path.dirname(ML)                      # 工作根目录（sfx/ long/ 与 alab.py 都在这层）   # 数据根: sfx/ long/ 录音
sys.path.insert(0, D)
import alab

os.makedirs(ML, exist_ok=True)
SR = 16000
LMAX = 40000            # 2.5 s
GAP = 0.25

items = []
for g in ["player", "root", "custom_se", "enemy"]:
    p = os.path.join(D, "sfx", "index_%s.json" % g)
    if not os.path.exists(p):
        continue
    for v in json.load(open(p, encoding="utf-8")).values():
        if 0.05 <= v["dur"] <= LMAX / SR:
            v["group"] = g
            items.append(v)
K = len(items)
print("templates", K, flush=True)


def to16(path, tsr):
    t, sr = alab.wav_read(path, mono=True)
    if t.ndim > 1:
        t = t.mean(axis=1)
    if sr != SR:
        t = alab.bandpass(t, sr, 30.0, 7500.0)
        t = np.interp(np.arange(int(len(t) * SR / sr)) / SR, np.arange(len(t)) / sr, t)
    return np.asarray(t, dtype=np.float32)


def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    return np.interp(np.arange(int(len(x) * sr_out / sr_in)) / sr_out,
                     np.arange(len(x)) / sr_in, x).astype(np.float32)


t0 = time.time()
bank = np.zeros((K, LMAX), dtype=np.float16)
lens = np.zeros(K, dtype=np.int32)
stream = []
offs = np.zeros(K, dtype=np.int64)
pos = 0
for i, v in enumerate(items):
    t = to16(v["file"], v["sr"])[:LMAX]
    bank[i, :len(t)] = t.astype(np.float16)
    lens[i] = len(t)
    offs[i] = pos
    stream.append(t)
    stream.append(np.zeros(int(GAP * SR), dtype=np.float32))
    pos += len(t) + int(GAP * SR)
print("clean bank built in %.0fs  stream %.1f M samples" % (time.time() - t0, pos / 1e6), flush=True)
np.save(os.path.join(ML, "bank_clean.npy"), bank)
np.save(os.path.join(ML, "bank_lens.npy"), lens)
np.save(os.path.join(ML, "bank_offs.npy"), offs)
json.dump([{k: v[k] for k in ("name", "bundle", "group", "dur")} for v in items],
          open(os.path.join(ML, "bank_index.json"), "w", encoding="utf-8"), ensure_ascii=False)

cat = np.concatenate(stream)
alab.wav_write(os.path.join(ML, "_cat.wav"), cat, SR)
print("concat wav written", flush=True)

for br in ["96k", "48k"]:
    alab.ff(["-i", os.path.join(ML, "_cat.wav"), "-c:a", "aac", "-b:a", br,
             "-ar", str(SR), "-ac", "1", os.path.join(ML, "_cat_%s.m4a" % br)], "aac_" + br)
    dec = os.path.join(ML, "_cat_%s.wav" % br)
    alab.ff(["-i", os.path.join(ML, "_cat_%s.m4a" % br), "-ar", str(SR), "-ac", "1",
             "-c:a", "pcm_s16le", dec], "dec_" + br)
    d, dsr = alab.wav_read(dec, mono=True)
    print("  %s decoded %d samples (ratio %.5f)" % (br, len(d), len(d) / len(cat)), flush=True)
    out = np.zeros((K, LMAX), dtype=np.float16)
    shift_best = []
    for i in range(K):
        L = lens[i]
        s = offs[i]
        if L <= 0:
            continue
        a = max(0, s - 64); b = min(len(d), s + L + 64)
        seg = d[a:b]
        if len(seg) < 64:
            continue
        # align by NCC over a small lag range
        ref = cat[s:s + L]
        best = -2; bo = 0
        for sh in range(0, len(seg) - L + 1):
            c = float(np.dot(seg[sh:sh + L], ref))
            if c > best:
                best = c; bo = sh
        out[i, :L] = seg[bo:bo + L].astype(np.float16)
        shift_best.append(a + bo - s)
    np.save(os.path.join(ML, "bank_%s.npy" % br), out)
    print("  aligned, offsets mean %.1f  saved bank_%s.npy" % (float(np.mean(shift_best)), br), flush=True)

os.remove(os.path.join(ML, "_cat.wav"))
print("DONE", flush=True)
