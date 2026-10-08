"""从已导出的 wav 目录生成训练脚本要读的索引文件。

解包工具因人而异（AssetStudio / UnityPy / 其它），本脚本不假设任何一种，只负责最后一跳：
把"一堆 wav"变成 `build_bank.py` / `build_long.py` 需要的 JSON 索引。

    # SFX 模板库的索引（四组，文件名随意，group 由脚本传入）
    python make_index.py sfx --dir <wav目录> --group player --out ../sfx/index_player.json
    python make_index.py sfx --dir <wav目录> --group root   --out ../sfx/index_root.json
    python make_index.py sfx --dir <wav目录> --group custom_se --out ../sfx/index_custom_se.json
    python make_index.py sfx --dir <wav目录> --group enemy  --out ../sfx/index_enemy.json

    # 长音池的索引（环境音/剧情音/干员战斗语音；可给多个 --dir）
    python make_index.py long --dir <环境音目录> --kind ambience --out ../long/index_long.json
    python make_index.py long --dir <语音目录> --kind voice_battle --append --out ../long/index_long.json

字段规范
--------
sfx 索引（dict，键 = 文件名）每项：
    file    wav 的路径（写绝对路径最稳）
    name    模板名（默认取文件名去扩展名；`--name-from` 可改成"目录名"或"bundle__name"）
    bundle  来源分组名（默认取 name 里第一个 `__` 之前的部分）
    dur     时长（秒）
    sr      采样率
long 索引（list）每项：
    file / name / bundle / dur / sr / kind / lang / loop

`kind` 决定它被当作长音床还是人声：
    长音床 = music / ambience / dialog / long_se        人声 = voice_battle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

__all__ = ["wav_info", "scan"]


def wav_info(path: str) -> tuple[float, int]:
    """返回 (时长秒, 采样率)。只读 wav 头，不解码。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        return w.getnframes() / float(sr if sr else 1), sr


def scan(dirs, exts=(".wav", ".WAV")) -> list[str]:
    out: list[str] = []
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            raise SystemExit("不是目录: %s" % d)
        for root, _sub, files in os.walk(d):
            for f in sorted(files):
                if f.endswith(exts):
                    out.append(os.path.join(root, f))
    return out


def split_name(path: str, name_from: str) -> tuple[str, str]:
    """返回 (name, bundle)。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if name_from == "dir":
        bundle = os.path.basename(os.path.dirname(path)) or "misc"
        return stem, bundle
    if "__" in stem:
        left, right = stem.split("__", 1)
        return stem, left
    return stem, "misc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("make_index")
    ap.add_argument("mode", choices=["sfx", "long"])
    ap.add_argument("--dir", action="append", required=True, help="wav 所在目录（可多次）")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--group", default="", help="sfx 模式：写进 group 字段（player/root/custom_se/enemy）")
    ap.add_argument("--kind", default="ambience",
                    help="long 模式：music/ambience/dialog/long_se/voice_battle")
    ap.add_argument("--lang", default="", help="long 模式：语言标记（jp/cn/en…）")
    ap.add_argument("--loop", action="store_true", help="long 模式：标记为循环素材")
    ap.add_argument("--name-from", default="stem", choices=["stem", "dir"])
    ap.add_argument("--append", action="store_true", help="并入已存在的 out（long 模式常用）")
    ap.add_argument("--min-dur", type=float, default=0.0, help="短于该时长的条目丢弃（秒）")
    ap.add_argument("--max-dur", type=float, default=0.0, help="长于该时长的条目丢弃（0=不限）")
    ap.add_argument("--relative", action="store_true", help="file 写相对 out 的路径（默认写绝对）")
    a = ap.parse_args(argv)

    files = scan(a.dir)
    if not files:
        raise SystemExit("这些目录里没有 wav: %s" % a.dir)
    out = os.path.abspath(a.out)
    base = os.path.dirname(out)

    items, skipped = [], 0
    for p in files:
        try:
            dur, sr = wav_info(p)
        except Exception:
            skipped += 1
            continue
        if dur < a.min_dur or (a.max_dur > 0 and dur > a.max_dur):
            skipped += 1
            continue
        name, bundle = split_name(p, a.name_from)
        rel = os.path.relpath(p, base).replace("\\", "/") if a.relative else os.path.abspath(p)
        items.append({"file": rel, "name": name, "bundle": bundle, "dur": round(dur, 4),
                      "sr": int(sr)})

    os.makedirs(base, exist_ok=True)
    if a.mode == "sfx":
        old = {}
        if a.append and os.path.isfile(out):
            old = json.load(open(out, encoding="utf-8"))
        table = dict(old)
        for it in items:
            it["group"] = a.group or it["bundle"]
            table[os.path.basename(it["file"])] = it
        json.dump(table, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("sfx 索引: %d 条（本次新增 %d，跳过 %d）-> %s"
              % (len(table), len(items), skipped, out))
    else:
        old = []
        if a.append and os.path.isfile(out):
            old = json.load(open(out, encoding="utf-8"))
        for it in items:
            it.update({"kind": a.kind, "lang": a.lang, "loop": bool(a.loop),
                       "voiceIndex": None, "voiceTitle": "", "placeType": ""})
        allitems = old + items
        json.dump(allitems, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        kinds: dict[str, int] = {}
        for it in allitems:
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        print("long 索引: %d 条（本次 +%d，跳过 %d）%s -> %s"
              % (len(allitems), len(items), skipped,
                 "  " + " ".join("%s=%d" % kv for kv in sorted(kinds.items())), out))
        print("  长音床 = music/ambience/dialog/long_se，人声 = voice_battle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
