"""解包：语音(保留 44.1 kHz) + BGM。**不使用 multiprocessing**（受限环境下命名管道会被拒绝），
并行由外部启动多个进程、每个处理一个分片实现：

    # 8 个分片并行（每个进程串行处理 1/8 的 bundle）
    for ($i=0; $i -lt 8; $i++) { Start-Process python -ArgumentList "... --shard $i --shards 8" }
    # 全部结束后合并
    python x_extract_long44_mp.py refs

输出到 long44/，文件名 `<bundle>__<clip>.wav`（一个 bundle 可能有多个 AudioClip，必须带 clip 名）。
保留 AudioClip 的原始采样率（语音 44100），不做任何降采样。
"""
import argparse
import glob
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

SR_OUT = 0
BATTLE_IDX = set(range(17, 33))
LANG_DIR = {"jp": "voice", "cn": "voice_cn", "en": "voice_en", "custom": "voice_custom"}
MAXSEC = {"voice": 60.0, "music": 180.0}
TAGS = ("voice_battle_jp", "voice_battle_cn", "voice_battle_en", "voice_battle_custom", "music")

_CFG: dict = {}


def _titles() -> dict:
    m = {}
    for p in (os.path.join(D, "_data", "charword_table.json"),
              os.path.join(D, "samples", "_data", "charword_table.json")):
        if os.path.exists(p):
            for e in json.load(open(p, encoding="utf-8"))["charWords"].values():
                try:
                    i = int(e["voiceIndex"])
                except Exception:
                    continue
                m.setdefault(i, (e.get("voiceTitle", ""), e.get("placeType", "")))
            break
    return m


def decode(data: bytes):
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


def bundles(root: str) -> list:
    out = []
    for dp, _dn, fn in os.walk(root):
        out += [os.path.join(dp, f) for f in fn if f.endswith(".ab")]
    return sorted(out)


def _one(bp: str):
    """一个 bundle -> (索引条目列表, 采样率计数, 错误数)。"""
    what = _CFG["what"]
    tag = _CFG["tag"]
    titles = _CFG["titles"]
    is_voice = what.startswith("voice")
    maxsec = MAXSEC["voice" if is_voice else "music"]
    stem = os.path.splitext(os.path.basename(bp))[0]
    out, srs, err = [], {}, 0
    try:
        env = load(bp)
    except Exception:
        return out, srs, 1
    for obj in env.objects:
        if obj.type.name != "AudioClip":
            continue
        try:
            d = obj.read()
        except Exception:
            err += 1
            continue
        # 先按名字筛选，再解码 samples：一个 char bundle 里有几十条语音，只要战斗那 16 条，
        # 若直接访问 d.samples 会把它们全部解码 —— 这是主要的耗时来源。
        raw_name = getattr(d, "m_Name", "") or ""
        vid0 = raw_name[:-4] if raw_name.endswith(".wav") else raw_name
        vi = None
        if is_voice:
            try:
                vi = int(vid0.split("_")[1])
            except Exception:
                continue
            if vi not in BATTLE_IDX:
                continue
        try:
            samples = d.samples or {}
        except Exception:
            err += 1
            continue
        for fn, data in samples.items():
            vid = fn[:-4] if fn.endswith(".wav") else fn
            if is_voice and vid != vid0:
                continue
            dst = os.path.join(OUT, tag, "%s__%s.wav" % (stem, vid))
            if not os.path.exists(dst):
                if not data:
                    continue
                try:
                    x, src_sr = decode(data)
                except Exception:
                    err += 1
                    continue
                sr = int(SR_OUT) or int(src_sr)
                y = x[:int(maxsec * sr)]
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                wav_write(dst, y, sr)
                nfr = len(y)
            else:
                with wave.open(dst, "rb") as w:
                    sr, nfr = w.getframerate(), w.getnframes()
                src_sr = sr
            srs[sr] = srs.get(sr, 0) + 1
            key = "%s/%s" % (stem, vid)
            meta = {"bundle": stem, "name": vid, "file": dst, "sr": sr,
                    "dur": round(nfr / float(sr), 4), "src_sr": int(src_sr)}
            if is_voice:
                ti, pl = titles.get(vi, ("", ""))
                meta.update({"voiceIndex": vi, "voiceTitle": ti, "placeType": pl,
                             "kind": "voice_battle", "lang": what.split("-")[1]})
            else:
                meta.update({"kind": "music", "lang": "", "loop": vid.endswith("_loop"),
                             "voiceIndex": None, "voiceTitle": "", "placeType": ""})
            out.append((tag, key, meta))
    return out, srs, err


def run(what: str, limit: int = 0, shard: int = 0, shards: int = 1) -> None:
    if what.startswith("voice"):
        lang = what.split("-")[1]
        root = os.path.join(PD, LANG_DIR[lang])
        tag = "voice_battle_%s" % lang
    else:
        root = os.path.join(SA, "music")
        tag = "music"
    if not os.path.isdir(root):
        print("跳过 %s: 目录不存在 %s" % (what, root), flush=True)
        return
    bs = bundles(root)
    if limit:
        bs = bs[:limit]
    if shards > 1:
        bs = bs[shard::shards]
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    _CFG["what"] = what
    _CFG["tag"] = tag
    _CFG["titles"] = _titles()
    suff = ".part%d" % shard if shards > 1 else ""
    ipath = os.path.join(OUT, "index_%s%s.json" % (tag, suff))
    index = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    print("[%s] shard %d/%d: %d 个 bundle -> %s" % (what, shard, shards - 1, len(bs), ipath),
          flush=True)
    tot_srs, n_err = {}, 0
    t0 = time.time()
    for n, bp in enumerate(bs, 1):
        items, srs, err = _one(bp)
        for _tg, key, meta in items:
            index[key] = meta
        for k, v in srs.items():
            tot_srs[k] = tot_srs.get(k, 0) + v
        n_err += err
        if n % 10 == 0 or n == len(bs):
            el = time.time() - t0
            print("  %4d/%d %5.1f%%  %5.0fs  eta %4.0fs  条目 %d  err %d  %s"
                  % (n, len(bs), 100 * n / max(len(bs), 1), el, el / n * (len(bs) - n),
                     len(index), n_err, tot_srs), flush=True)
            json.dump(index, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(index, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    print("[%s] DONE shard %d 条目 %d err %d %s 用时 %.0fs"
          % (what, shard, len(index), n_err, tot_srs, time.time() - t0), flush=True)


def merge_shards() -> None:
    """把 index_<tag>.part*.json 合并成 index_<tag>.json。"""
    for tag in TAGS:
        parts = sorted(glob.glob(os.path.join(OUT, "index_%s.part*.json" % tag)))
        if not parts:
            continue
        merged = {}
        for p in parts:
            d = json.load(open(p, encoding="utf-8"))
            if isinstance(d, dict):
                merged.update(d)
        out = os.path.join(OUT, "index_%s.json" % tag)
        json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("合并 %s: %d 片 -> %d 条" % (tag, len(parts), len(merged)), flush=True)


def build_refs() -> None:
    """long44/index_long.json：语音 + BGM + sfx 里的长素材，供长音池使用。"""
    items = []
    for tag in TAGS:
        p = os.path.join(OUT, "index_%s.json" % tag)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            items += list(d.values()) if isinstance(d, dict) else d
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
    print("index_long.json: %d 条 %s" % (len(items), dict(c)), flush=True)
    print("  dur min=%.2f p50=%.2f max=%.2f  total=%.1f min"
          % (durs.min(), np.median(durs), durs.max(), durs.sum() / 60), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+",
                    choices=["voice-jp", "voice-cn", "voice-en", "voice-custom",
                             "music", "refs", "merge"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for w in a.what:
        if w == "refs":
            build_refs()
        elif w == "merge":
            merge_shards()
        else:
            run(w, a.limit, a.shard, a.shards)
