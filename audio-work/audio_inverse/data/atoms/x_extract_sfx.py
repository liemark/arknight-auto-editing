"""Extract Arknights SFX AudioClips from the installed PC client's AB bundles."""
import sys, os, json, time, wave, io
TOOLS = r"G:\AI使用\Arknights解包\tools"
sys.path.insert(0, TOOLS)
import arkpy
arkpy.patch_unitypy()
from UnityPy import load

BASE = r"D:\download\Hypergryph Launcher\games\Arknights Game\Arknights_Data\StreamingAssets\AB\Windows\audio\sound_beta_2"
OUT = r"F:\杂七杂八\arknight-auto-editing\_audio_lab\sfx"

def collect(group):
    if group == "player":
        d = os.path.join(BASE, "player")
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".ab"))
    if group == "root":
        return sorted(os.path.join(BASE, f) for f in os.listdir(BASE) if f.endswith(".ab"))
    if group == "enemy":
        out = []
        for root, _dirs, files in os.walk(os.path.join(BASE, "enemy")):
            out += [os.path.join(root, f) for f in files if f.endswith(".ab")]
        return sorted(out)
    if group == "custom_se":
        d = os.path.join(os.path.dirname(BASE), "custom_se")
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".ab"))
    raise SystemExit("unknown group " + group)

def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "player"
    outdir = os.path.join(OUT, group)
    os.makedirs(outdir, exist_ok=True)
    idxpath = os.path.join(OUT, "index_%s.json" % group)
    index = json.load(open(idxpath, "r", encoding="utf-8")) if os.path.exists(idxpath) else {}
    bundles = collect(group)
    t0 = time.time()
    n_new = n_skip = n_err = 0
    for bp in bundles:
        stem = os.path.splitext(os.path.basename(bp))[0]
        try:
            env = load(bp)
        except Exception as e:
            print("LOAD FAIL", stem, type(e).__name__, e); n_err += 1; continue
        for obj in env.objects:
            if obj.type.name != "AudioClip":
                continue
            try:
                d = obj.read()
                smp = d.samples
            except Exception as e:
                n_err += 1
                if n_err < 6: print("READ FAIL", stem, type(e).__name__, e)
                continue
            for fn, data in smp.items():
                key = "%s/%s" % (stem, fn)
                dst = os.path.join(outdir, "%s__%s" % (stem, fn))
                if key in index and os.path.exists(dst):
                    n_skip += 1
                    continue
                if not data:
                    continue
                with open(dst, "wb") as f:
                    f.write(data)
                try:
                    with wave.open(io.BytesIO(data), "rb") as w:
                        sr, ch, fr = w.getframerate(), w.getnchannels(), w.getnframes()
                except Exception:
                    sr = ch = fr = 0
                index[key] = {"bundle": stem, "name": fn[:-4] if fn.endswith(".wav") else fn,
                              "file": dst, "sr": sr, "ch": ch, "frames": fr,
                              "dur": round(fr / sr, 4) if sr else 0}
                n_new += 1
        print("  %-22s clips_so_far=%d  %.1fs" % (stem, n_new, time.time() - t0), flush=True)
    with open(idxpath, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print("DONE group=%s new=%d skip=%d err=%d total=%d in %.1fs" % (group, n_new, n_skip, n_err, len(index), time.time() - t0))

main()
