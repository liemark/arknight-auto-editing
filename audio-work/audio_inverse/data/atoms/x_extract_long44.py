"""重新解包为**原始采样率**（语音源是 44.1 kHz），并补上之前没抽的 BGM。

与 x_extract_long.py 的差别：
  * 不做 16 kHz 降采样：`SR_OUT = 0` 表示保留 AudioClip 的原始采样率（语音为 44100）。
    16 kHz 会把 8 kHz 以上全部丢掉，那一段永远无法参与匹配/抵消。
  * 输出到 long44/，与旧的 16 kHz 素材分开，互不覆盖。
  * 增加 music（BGM）抽取。BGM 在真实录音里长期存在，是背景域差的主要来源。
  * 输出文件名始终带 clip 名（`<bundle>__<clip>.wav`）：一个 bundle 里可能有多个 AudioClip，
    只用 bundle 名会互相覆盖。

用法：
    python x_extract_long44.py voice-jp --limit 2      # 先小规模验证
    python x_extract_long44.py all
    python x_extract_long44.py music
"""
import argparse
import io
import json
import os
import sys
import time
import wave

import numpy as np

TOOLS = r"G:\AI使用\Arknights解包\tools"
sys.path.insert(0, TOOLS)
import arkpy                                                    # noqa: E402
arkpy.patch_unitypy()
from UnityPy import load                                        # noqa: E402

GAME = r"D:\download\Hypergryph Launcher\games\Arknights Game"
PD = os.path.join(GAME, r"Arknights_Data\PersistentData\Bundles\audio\sound_beta_2")
SA = os.path.join(GAME, r"Arknights_Data\StreamingAssets\AB\Windows\audio\sound_beta_2")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "long44")

SR_OUT = 0                        # 0 = 保留原始采样率；非 0 = 重采样到该值
BATTLE_IDX = set(range(17, 33))   # 编入队伍 .. 行动失败
LANG_DIR = {"jp": "voice", "cn": "voice_cn", "en": "voice_en", "custom": "voice_custom"}
MAXSEC_VOICE = 60.0
MAXSEC_MUSIC = 180.0


def battle_titles() -> dict:
    """voiceIndex -> (voiceTitle, placeType)，来自社区解包表。"""
    m = {}
    for p in (os.path.join(D, "_data", "charword_table.json"),
              os.path.join(D, "samples", "_data", "charword_table.json")):
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["charWords"].values():
            try:
                i = int(e["voiceIndex"])
            except Exception:
                continue
            m.setdefault(i, (e.get("voiceTitle", ""), e.get("placeType", "")))
        break
    return m


def bundles(root: str) -> list[str]:
    out = []
    for dp, _dn, fn in os.walk(root):
        out += [os.path.join(dp, f) for f in fn if f.endswith(".ab")]
    return sorted(out)


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """wav 字节 -> (mono float32, 采样率)。支持 8/16/24/32-bit PCM。"""
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


def wav_write(path: str, x: np.ndarray, sr: int) -> None:
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


def _clips(bp: str):
    """逐个 AudioClip 产出 (bundle_stem, clip_name, wav_bytes)。"""
    stem = os.path.splitext(os.path.basename(bp))[0]
    env = load(bp)
    for obj in env.objects:
        if obj.type.name != "AudioClip":
            continue
        d = obj.read()
        for fn, data in (d.samples or {}).items():
            vid = fn[:-4] if fn.endswith(".wav") else fn
            yield stem, vid, data


def extract_voice(lang: str, limit: int = 0, maxsec: float = MAXSEC_VOICE) -> None:
    src = os.path.join(PD, LANG_DIR[lang])
    if not os.path.isdir(src):
        print("跳过 %s: 目录不存在 %s" % (lang, src), flush=True)
        return
    outdir = os.path.join(OUT, "voice_battle_%s" % lang)
    os.makedirs(outdir, exist_ok=True)
    ipath = os.path.join(OUT, "index_voice_battle_%s.json" % lang)
    index = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    titles = battle_titles()
    bs = bundles(src)
    if limit:
        bs = bs[:limit]
    t0 = time.time()
    n_new = n_skip = n_err = 0
    srs: dict[int, int] = {}
    for bp in bs:
        try:
            clips = list(_clips(bp))
        except Exception as e:
            print("LOAD FAIL", os.path.basename(bp), type(e).__name__, e, flush=True)
            n_err += 1
            continue
        for stem, vid, data in clips:
            try:
                i = int(vid.split("_")[1])
            except Exception:
                continue
            if i not in BATTLE_IDX:
                continue
            key = "%s/%s" % (stem, vid)
            dst = os.path.join(outdir, "%s__%s.wav" % (stem, vid))
            if key in index and os.path.exists(dst):
                n_skip += 1
                continue
            if not data:
                continue
            try:
                x, src_sr = decode(data)
            except Exception as e:
                n_err += 1
                if n_err < 8:
                    print("DEC FAIL", stem, vid, type(e).__name__, e, flush=True)
                continue
            sr = int(SR_OUT) or int(src_sr)              # 保留原始采样率
            y = x[:int(maxsec * sr)]
            wav_write(dst, y, sr)
            srs[sr] = srs.get(sr, 0) + 1
            ti, pl = titles.get(i, ("", ""))
            index[key] = {"bundle": stem, "name": vid, "file": dst, "sr": sr,
                          "dur": round(len(y) / sr, 4), "src_sr": int(src_sr),
                          "voiceIndex": i, "voiceTitle": ti, "placeType": pl,
                          "kind": "voice_battle", "lang": lang}
            n_new += 1
        print("  %-28s clips=%d  %.0fs" % (os.path.basename(bp), len(index), time.time() - t0),
              flush=True)
    json.dump(index, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    print("DONE voice-%s bundles=%d new=%d skip=%d err=%d 采样率分布=%s  %.0fs"
          % (lang, len(bs), n_new, n_skip, n_err, srs, time.time() - t0), flush=True)


def extract_music(limit: int = 0, maxsec: float = MAXSEC_MUSIC) -> None:
    """BGM：StreamingAssets/.../music/** 下的 *_intro/_loop 成对素材。"""
    src = os.path.join(SA, "music")
    if not os.path.isdir(src):
        print("跳过 music: 目录不存在 %s" % src, flush=True)
        return
    outdir = os.path.join(OUT, "music")
    os.makedirs(outdir, exist_ok=True)
    ipath = os.path.join(OUT, "index_music.json")
    index = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    bs = bundles(src)
    if limit:
        bs = bs[:limit]
    t0 = time.time()
    n_new = n_skip = n_err = 0
    srs: dict[int, int] = {}
    for bp in bs:
        try:
            clips = list(_clips(bp))
        except Exception as e:
            print("LOAD FAIL", os.path.basename(bp), type(e).__name__, e, flush=True)
            n_err += 1
            continue
        for stem, vid, data in clips:
            key = "%s/%s" % (stem, vid)
            dst = os.path.join(outdir, "%s__%s.wav" % (stem, vid))
            if key in index and os.path.exists(dst):
                n_skip += 1
                continue
            if not data:
                continue
            try:
                x, src_sr = decode(data)
            except Exception:
                n_err += 1
                continue
            sr = int(SR_OUT) or int(src_sr)
            y = x[:int(maxsec * sr)]
            wav_write(dst, y, sr)
            srs[sr] = srs.get(sr, 0) + 1
            index[key] = {"bundle": stem, "name": vid, "file": dst, "sr": sr,
                          "dur": round(len(y) / sr, 4), "src_sr": int(src_sr),
                          "kind": "music", "lang": "", "loop": vid.endswith("_loop"),
                          "voiceIndex": None, "voiceTitle": "", "placeType": ""}
            n_new += 1
    json.dump(index, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    print("DONE music bundles=%d new=%d skip=%d err=%d 采样率分布=%s  %.0fs"
          % (len(bs), n_new, n_skip, n_err, srs, time.time() - t0), flush=True)


def build_refs() -> None:
    """long44/index_long.json：语音 + BGM，供长音池使用。"""
    items = []
    for name in ("index_voice_battle_jp.json", "index_voice_battle_cn.json",
                 "index_voice_battle_en.json", "index_voice_battle_custom.json",
                 "index_music.json"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            items += list(d.values()) if isinstance(d, dict) else d
    # sfx/ 里本来就有的长素材（环境音/剧情音/长音效），沿用原索引但补上 kind
    for g in ("root", "player", "enemy", "custom_se"):
        p = os.path.join(D, "sfx", "index_%s.json" % g)
        if not os.path.exists(p):
            continue
        for v in json.load(open(p, encoding="utf-8")).values():
            b = v.get("bundle", "")
            if b in ("ambience", "dialog"):
                items.append(dict(v, kind="ambience" if b == "ambience" else "dialog"))
            elif float(v.get("dur") or 0) > 2.5:
                items.append(dict(v, kind="long_se"))
    keep = ("name", "bundle", "file", "sr", "dur", "kind", "voiceIndex", "voiceTitle",
            "placeType", "lang", "loop", "src_sr")
    items = [{k: v[k] for k in keep if k in v} for v in items]
    json.dump(items, open(os.path.join(OUT, "index_long.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    import collections
    c = collections.Counter(v["kind"] for v in items)
    durs = np.array([float(v["dur"]) for v in items])
    total_h = sum(float(v["dur"]) for v in items if v.get("sr", 16000) == 16000) / 3600
    print("index_long.json: %d 条 %s" % (len(items), dict(c)), flush=True)
    print("  dur min=%.2f p50=%.2f max=%.2f  total=%.1f min"
          % (durs.min(), np.median(durs), durs.max(), durs.sum() / 60), flush=True)
    del total_h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["voice-jp", "voice-cn", "voice-en", "voice-custom",
                                     "music", "refs", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxsec", type=float, default=0.0, help="0 = 用各自默认上限")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "all":
        extract_voice("jp", a.limit, a.maxsec or MAXSEC_VOICE)
        extract_voice("cn", a.limit, a.maxsec or MAXSEC_VOICE)
        extract_music(a.limit, a.maxsec or MAXSEC_MUSIC)
        build_refs()
    elif a.what == "music":
        extract_music(a.limit, a.maxsec or MAXSEC_MUSIC)
    elif a.what == "refs":
        build_refs()
    else:
        lang = a.what.split("-")[1]
        extract_voice(lang, a.limit, a.maxsec or MAXSEC_VOICE)
