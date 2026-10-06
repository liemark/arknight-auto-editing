"""Extract LONG / LOOPING game audio (operator battle voice, BGM) into the test pool.

The template bank (build_bank.py) hard-caps clips at 2.5 s, so everything long --
voice lines, BGM, ambience -- was silently dropped. This script pulls those out
into _audio_lab/long/ (NOT _audio_lab/sfx/, so the template bank is unaffected).

  groups:
    voice-jp / voice-cn : operator BATTLE voice only (charword voiceIndex 17..32)
    refs                : index_long.json = voice + long clips already in sfx/

BGM (music/**) is deliberately NOT extracted: user dropped it.  If it is ever wanted back,
it is ~565 bundles of *_loop/_intro pairs under StreamingAssets; the naming trap is that one
bundle holds two AudioClips with the SAME bundle stem, so the output name must include the
clip name or the loop silently overwrites the intro.

Prints ASCII only (the Chinese titles go into the UTF-8 JSON index).
"""
import os, sys, io, json, wave, argparse, time
import numpy as np

TOOLS = r"G:\AI使用\Arknights解包\tools"
sys.path.insert(0, TOOLS)
import arkpy
arkpy.patch_unitypy()
from UnityPy import load

GAME = r"D:\download\Hypergryph Launcher\games\Arknights Game"
PD = os.path.join(GAME, r"Arknights_Data\PersistentData\Bundles\audio\sound_beta_2")
D = r"F:\杂七杂八\arknight-auto-editing\_audio_lab"
LONG = os.path.join(D, "long")
DATA = os.path.join(D, "_data")
SR16 = 16000
BATTLE_IDX = set(range(17, 33))          # 编入队伍 .. 行动失败
LANG_DIR = {"jp": "voice", "cn": "voice_cn", "en": "voice_en", "custom": "voice_custom"}
MAXSEC = 60.0                            # per track kept


def battle_titles():
    """voiceIndex -> voiceTitle / placeType, read from the community charword table."""
    p = os.path.join(DATA, "charword_table.json")
    m = {}
    if not os.path.exists(p):
        return m
    for e in json.load(open(p, encoding="utf-8"))["charWords"].values():
        try:
            i = int(e["voiceIndex"])
        except Exception:
            continue
        m.setdefault(i, (e.get("voiceTitle", ""), e.get("placeType", "")))
    return m


def bundles(root):
    out = []
    for dp, _dn, fn in os.walk(root):
        out += [os.path.join(dp, f) for f in fn if f.endswith(".ab")]
    return sorted(out)


def decode(data):
    """-> mono float32 in [-1,1], sr. Handles 8/16/24/32-bit PCM."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, fr, sw = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(fr)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("sampwidth %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def resample(x, sr_in, sr_out, lp=7400.0):
    if sr_in == sr_out:
        return x
    n = len(x)
    nf = 1 << int(np.ceil(np.log2(max(n, 64))))
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1.0 / sr_in)
    X[f > lp] = 0.0
    y = np.fft.irfft(X, nf)[:n]
    return np.interp(np.arange(int(len(y) * sr_out / sr_in)) / sr_out,
                     np.arange(len(y)) / sr_in, y).astype(np.float32)


def wav_write16(path, x, sr):
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


# ---------------------------------------------------------------- voice
def extract_voice(lang, limit=0, maxsec=MAXSEC):
    src = os.path.join(PD, LANG_DIR[lang])
    outdir = os.path.join(LONG, "voice_battle_%s" % lang)
    os.makedirs(outdir, exist_ok=True)
    ipath = os.path.join(LONG, "index_voice_battle_%s.json" % lang)
    index = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    titles = battle_titles()
    bs = bundles(src)
    if limit:
        bs = bs[:limit]
    t0 = time.time(); n_new = n_skip = n_err = n_clip = 0
    for bp in bs:
        stem = os.path.splitext(os.path.basename(bp))[0]
        try:
            env = load(bp)
        except Exception as e:
            print("LOAD FAIL", stem, type(e).__name__, e, flush=True); n_err += 1; continue
        for obj in env.objects:
            if obj.type.name != "AudioClip":
                continue
            try:
                d = obj.read()
                smp = d.samples
            except Exception as e:
                n_err += 1
                if n_err < 6:
                    print("READ FAIL", stem, type(e).__name__, e, flush=True)
                continue
            if not smp:
                continue
            for fn, data in smp.items():
                vid = fn[:-4] if fn.endswith(".wav") else fn
                try:
                    i = int(vid.split("_")[1])
                except Exception:
                    continue
                if i not in BATTLE_IDX:
                    continue
                n_clip += 1
                key = "%s/%s" % (stem, vid)
                dst = os.path.join(outdir, "%s__%s.wav" % (stem, vid))
                if key in index and os.path.exists(dst):
                    n_skip += 1; continue
                if not data:
                    continue
                try:
                    x, src_sr = decode(data)
                except Exception as e:
                    n_err += 1
                    if n_err < 8:
                        print("DEC FAIL", stem, vid, type(e).__name__, e, flush=True)
                    continue
                y = resample(x, src_sr, SR16)[:int(maxsec * SR16)]
                wav_write16(dst, y, SR16)
                ti, pl = titles.get(i, ("", ""))
                index[key] = {"bundle": stem, "name": vid, "file": dst, "sr": SR16,
                              "dur": round(len(y) / SR16, 4), "src_sr": src_sr,
                              "voiceIndex": i, "voiceTitle": ti, "placeType": pl,
                              "kind": "voice_battle", "lang": lang}
                n_new += 1
        print("  %-28s battle_clips=%d  %.0fs" % (stem, len(index), time.time() - t0), flush=True)
    json.dump(index, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    print("DONE voice-%s bundles=%d battle_clips=%d new=%d skip=%d err=%d in %.0fs"
          % (lang, len(bs), n_clip, n_new, n_skip, n_err, time.time() - t0), flush=True)


# ---------------------------------------------------------------- refs
def build_refs(min_sec=2.5):
    """index_long.json: everything long, wherever it lives."""
    items = []
    for lang in LANG_DIR:
        p = os.path.join(LONG, "index_voice_battle_%s.json" % lang)
        if os.path.exists(p):
            items += list(json.load(open(p, encoding="utf-8")).values())
    # long clips that x_extract_sfx.py already pulled into sfx/ (all groups, not just root):
    # 环境音和剧情音本来就在那儿, 只是 build_bank.py 的 2.5 s 上限把它们滤掉了.
    for g in ("root", "player", "enemy", "custom_se"):
        p = os.path.join(D, "sfx", "index_%s.json" % g)
        if not os.path.exists(p):
            continue
        for v in json.load(open(p, encoding="utf-8")).values():
            b = v["bundle"]
            if b in ("ambience", "dialog"):
                items.append(dict(v, kind="ambience" if b == "ambience" else "dialog"))
            elif v["dur"] > min_sec:
                items.append(dict(v, kind="long_se"))
    keep = ("name", "bundle", "file", "sr", "dur", "kind", "voiceIndex", "voiceTitle",
            "placeType", "lang", "loop", "src_dur", "src_sr")
    items = [{k: v[k] for k in keep if k in v} for v in items]
    json.dump(items, open(os.path.join(LONG, "index_long.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    import collections
    c = collections.Counter(v["kind"] for v in items)
    durs = np.array([v["dur"] for v in items])
    print("index_long.json: %d items %s" % (len(items), dict(c)), flush=True)
    print("  dur min=%.2f p50=%.2f max=%.2f  total=%.1f min"
          % (durs.min(), np.median(durs), durs.max(), durs.sum() / 60), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["voice-jp", "voice-cn", "voice-en", "voice-custom",
                                     "refs", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxsec", type=float, default=MAXSEC)
    a = ap.parse_args()
    os.makedirs(LONG, exist_ok=True)
    if a.what == "all":
        extract_voice("jp", a.limit, a.maxsec)
        extract_voice("cn", a.limit, a.maxsec)
        build_refs()
    elif a.what.startswith("voice"):
        extract_voice(a.what.split("-")[1], a.limit, a.maxsec)
    else:
        build_refs()
