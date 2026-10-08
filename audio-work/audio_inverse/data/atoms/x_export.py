"""Export the 10-20 s segment for listening: original, normalized, and a rough template reconstruction."""
import os, sys, json, math, subprocess
import numpy as np
D = r"F:\杂七杂八\arknight-auto-editing\_audio_lab"
sys.path.insert(0, D)
import alab

SR = 48000
T0, T1 = 10.0, 20.0
OUT = os.path.join(D, "out"); os.makedirs(OUT, exist_ok=True)

# ---------- 1. original segment (stereo, bit-exact from source wav) ----------
st, _ = alab.wav_read(os.path.join(D, "v82_stereo.wav"), mono=False)
seg = st[int(T0 * SR):int(T1 * SR)]
alab.wav_write(os.path.join(OUT, "seg_10-20_original.wav"), seg, SR)
pk = float(np.abs(seg).max())
print("original: peak %.5f (%.1f dBFS)  rms %.5f (%.1f dBFS)" % (
    pk, 20 * np.log10(pk + 1e-12), np.sqrt((seg.astype(np.float64) ** 2).mean()),
    20 * np.log10(np.sqrt((seg.astype(np.float64) ** 2).mean()) + 1e-12)))

# ---------- 2. peak-normalised ----------
g = 0.89 / max(pk, 1e-9)
norm = np.clip(seg * g, -1, 1)
alab.wav_write(os.path.join(OUT, "seg_10-20_normalized.wav"), norm, SR)
print("normalized: gain %+.1f dB" % (20 * np.log10(g)))

# ---------- 3. rough reconstruction from the corrected detections ----------
data = json.load(open(os.path.join(D, "gpu_scan.json"), encoding="utf-8"))
idx = {}
for grp in ["player", "root", "custom_se", "enemy"]:
    p = os.path.join(D, "sfx", "index_%s.json" % grp)
    if os.path.exists(p):
        for v in json.load(open(p, encoding="utf-8")).values():
            idx[(v["bundle"], v["name"])] = v

events = []
for r in data:
    f = 1.0 / math.sqrt(max(1, int(round(r["dur"] * SR))))
    for v, t in r["top"]:
        s = v * f
        if s >= 0.45 and T0 <= t < T1:
            events.append((s, t, r["bundle"], r["name"]))
events.sort(reverse=True)
picked = []
for s, t, b, n in events:
    if all(abs(t - t2) > 0.12 or n2 == n for _s, t2, _b, n2 in picked):
        picked.append((s, t, b, n))
    if len(picked) >= 40:
        break
print("reconstruction events:", len(picked))

N = int((T1 - T0) * SR)
recon = np.zeros(N, dtype=np.float64)
for s, t, b, n in picked:
    v = idx.get((b, n))
    if v is None:
        continue
    w, wsr = alab.wav_read(v["file"], mono=True)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if wsr != SR:
        w = alab.bandpass(w, wsr, 40.0, min(7500.0, wsr / 2 - 500))
        w = np.interp(np.arange(int(len(w) * SR / wsr)) / SR, np.arange(len(w)) / wsr, w)
    i = int((t - T0) * SR)
    L = min(len(w), N - i)
    if L > 0:
        recon[i:i + L] += w[:L] * s
alab.wav_write(os.path.join(OUT, "seg_10-20_recon_only.wav"), recon, SR)
print("recon raw peak %.4f rms %.4f" % (np.abs(recon).max(), np.sqrt((recon ** 2).mean())))

# mixed: normalized original + reconstruction
mix = norm.mean(axis=1) * 0.9 + recon * (0.35 / max(np.abs(recon).max(), 1e-9))
mx = float(np.abs(mix).max())
mix = np.clip(mix / mx * 0.89, -1, 1) if mx > 0 else mix
alab.wav_write(os.path.join(OUT, "seg_10-20_original_plus_recon.wav"), mix, SR)

# ---------- mp3 for easy playback + spectrogram ----------
for f in ["seg_10-20_original", "seg_10-20_normalized", "seg_10-20_recon_only", "seg_10-20_original_plus_recon"]:
    alab.ff(["-i", os.path.join(OUT, f + ".wav"), "-b:a", "192k", os.path.join(OUT, f + ".mp3")], "enc_" + f)
alab.ff(["-i", os.path.join(OUT, "seg_10-20_normalized.wav"),
         "-lavfi", "showspectrumpic=s=1400x500:legend=1", "-frames:v", "1", os.path.join(OUT, "seg_10-20_spectrogram.png")], "spec")

print("\n--- reconstruction event list ---")
for s, t, b, n in sorted(picked, key=lambda x: x[1])[:40]:
    print("  %6.2fs  %-30s  score %.2f" % (t, n, s))
print("\nfiles:")
for f in sorted(os.listdir(OUT)):
    print("  %-42s %8.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
