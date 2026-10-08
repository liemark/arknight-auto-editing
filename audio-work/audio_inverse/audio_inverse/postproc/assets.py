"""训练脚本要读的数据文件：检查在不在、补齐缺的、打印资产状态。

训练脚本在 `<项目根>/train/`，数据根是 `<项目根>/data/atoms/`（音频源、长音、录音）。
**采样率 44.1 kHz**：`train/core.py` 的 `SR`、模板库/长音池的 `sr`、`bg44.npy` 必须一致，
不一致时波形会被当另一个采样率用（音高与时序都错），而且不报错。本模块会交叉校验。

| 需要的文件 | 谁生成 | 怎么补 |
|---|---|---|
| `train/bank_pool.npy` `bank_offs.npy` `bank_lens.npy` `bank_index.json` `bank_meta.json` | `train/build_bank.py` | `python train/build_bank.py`（SFX + 战斗语音，约 2.5 分钟） |
| `long44/index_long.json` | 解包脚本 refs（需要游戏安装目录） | 本脚本 `long-index`：用已有的 `long44/index_*.json` + `sfx/index_*.json` 拼 |
| `train/long_pool.npy` `long_offs.npy` `long_lens.npy` `long_meta.json` `long_summary.json` | `train/build_long.py` | `python train/build_long.py`（吃上面那个 index，约 4 分钟） |
| `train/bg44.npy` | **本脚本 `bgrec`**，`train/synth.py` 首次用到时也会自动生成 | `python -m audio_inverse.postproc.assets bgrec` |
| `train/variants.npy` `variants_lens.npy` `variants_ncc.npy` | `train/gen_variants.py`（audiomentations） | 可选、很慢（数 GB）；没有时脚本打印告警并退化为干净模板 |
| `train/clusters_512.npy` `clusters_512.json` | `train/make_clusters.py` | 不需要：训练用 `--label-mode template` |

    # 看齐了没有（本页就是"现在能不能开跑"的答案）
    python -m audio_inverse.postproc.assets status

    # 造 bg44.npy（--bg-mode recording 的前提）
    python -m audio_inverse.postproc.assets bgrec

    # 造 long44/index_long.json（build_long.py 的输入）
    python -m audio_inverse.postproc.assets long-index
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

from ..config import PKG_ROOT, load_cfg

SR = 44100                      # 必须与 train/core.py 的 SR 一致（status 会校验）
BG_NAME = "bg%d.npy" % (SR // 1000)
BED_BUNDLES = ("ambience", "ambient", "dialog")
LONG_SE_MIN_DUR = 2.5
VOICE_TAGS = ("voice_battle_jp", "voice_battle_cn")
MUSIC_TAG = "music"
KEEP_FIELDS = ("name", "bundle", "file", "sr", "dur", "kind", "voiceIndex",
               "voiceTitle", "placeType", "lang", "loop", "src_sr")


def _atom_root(cfg) -> str:
    return cfg.abspath("atom_root")


def _train_dir(cfg) -> str:
    """训练/识别脚本目录：模板库、长音池、bg44.npy、ckpt_tmpl 都在这里。"""
    return os.path.join(PKG_ROOT, "train")


def _core_sr(cfg) -> int:
    """从 train/core.py 读 SR，用于校验各处采样率是否一致（不导入 torch）。"""
    p = os.path.join(_train_dir(cfg), "core.py")
    try:
        m = re.search(r"^SR\s*=\s*(\d+)", open(p, encoding="utf-8").read(), re.M)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


# ------------------------------------------------------------------- bg（录音）
def make_bg(cfg, *, src: str = "") -> str:
    """真实录音 -> `train/bg44.npy`（`--bg-mode recording` 的背景）。

    不传 `--src` 时直接调用 `synth._recording_pool()`：这样产物与训练时自动生成的
    完全一致（候选源：`nl_mono.wav` / `real/ui_demo_mono48k.wav` / `v82_mono.wav`）。
    传 `--src` 时只用这一个录音，处理链与 synth 相同：30 Hz~0.475*min(sr,SR) 带通 -> 线性插值到 SR。
    """
    tr = _train_dir(cfg)
    dst = os.path.join(tr, BG_NAME)
    if not src:
        if os.path.isfile(dst):
            x = np.load(dst, mmap_mode="r")
            print("bgrec: 已存在 %s（%.1f s，%d 采样）" % (dst, len(x) / SR, len(x)))
            return dst
        sys.path.insert(0, tr)
        import synth                                              # noqa: E402
        p = synth._recording_pool()
        x = np.load(p, mmap_mode="r")
        print("bgrec: %s  %.1f s @%d Hz  (%.1f MB)"
              % (p, len(x) / synth.SR, synth.SR, os.path.getsize(p) / 1e6))
        return p
    if not os.path.isfile(src):
        raise SystemExit("找不到录音源: %s" % src)
    root = _atom_root(cfg)
    sys.path.insert(0, root)
    import alab                                                   # noqa: E402
    x, sr = alab.wav_read(src, mono=True)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if int(sr) != SR:
        x = alab.bandpass(x, int(sr), 30.0, min(int(sr), SR) * 0.475)
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR,
                      np.arange(len(x)) / sr, x).astype(np.float32)
    os.makedirs(tr, exist_ok=True)
    np.save(dst, x.astype(np.float32))
    print("bgrec: %s  %.1f s @%d Hz -> %s  (%.1f MB)"
          % (os.path.basename(src), len(x) / SR, SR, dst, x.nbytes / 1e6))
    return dst


# --------------------------------------------------------------- long-index
def _load_items(p: str) -> list:
    d = json.load(open(p, encoding="utf-8"))
    return list(d.values()) if isinstance(d, dict) else list(d)


def make_long_index(cfg, *, out: str = "") -> str:
    """现有索引 -> `long44/index_long.json`（`build_long.py` 的输入）。

    与解包脚本的 `refs` 等价（解包需要游戏安装目录，这里只用已经解出来的索引）：
      * `long44/index_voice_battle_{jp,cn}.json` + `long44/index_music.json` 原样收；
      * `sfx/index_{root,player,enemy,custom_se}.json` 里 bundle 为 ambience/dialog 的算长音床，
        其余时长 > 2.5 s 的算 long_se（短音效已经在模板库里，不必再进长音池）。
    只收文件真实存在的条目，并打印被丢掉的条数。
    """
    root = _atom_root(cfg)
    l44 = os.path.join(root, "long44")
    items: list[dict] = []
    for tag in VOICE_TAGS + (MUSIC_TAG,):
        p = os.path.join(l44, "index_%s.json" % tag)
        if not os.path.isfile(p):
            print("  [跳过] 没有 %s" % p)
            continue
        kind = "voice_battle" if tag.startswith("voice_battle") else tag
        for v in _load_items(p):
            items.append(dict(v, kind=v.get("kind") or kind))
    for g in ("root", "player", "enemy", "custom_se"):
        p = os.path.join(root, "sfx", "index_%s.json" % g)
        if not os.path.isfile(p):
            continue
        for v in _load_items(p):
            b = str(v.get("bundle", "")).lower()
            if b.startswith(BED_BUNDLES):
                items.append(dict(v, kind="ambience" if b.startswith("ambient") else "dialog"))
            elif float(v.get("dur") or 0.0) > LONG_SE_MIN_DUR:
                items.append(dict(v, kind="long_se"))

    n_all = len(items)
    items = [v for v in items if v.get("file") and os.path.isfile(str(v["file"]))]
    out_items = [{k: v[k] for k in KEEP_FIELDS if k in v} for v in items]
    dst = out or os.path.join(l44, "index_long.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out_items, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    kinds: dict = {}
    for v in out_items:
        kinds[str(v.get("kind"))] = kinds.get(str(v.get("kind")), 0) + 1
    durs = np.array([float(v.get("dur") or 0.0) for v in out_items])
    print("long-index: %d 条（丢掉文件不存在的 %d 条）%s  共 %.1f 分钟 -> %s"
          % (len(out_items), n_all - len(out_items), kinds, durs.sum() / 60.0, dst))
    if durs.size:
        print("  dur min=%.2f p50=%.2f max=%.2f"
              % (durs.min(), np.median(durs), durs.max()))
    print("  下一步：cd %s && python build_long.py" % _train_dir(cfg))
    return dst


# -------------------------------------------------------------------- status
# (位置, 相对路径, 分组, 生成者, 说明)   位置: "train" = 训练脚本目录, "data" = 数据根
ROWS = [
    ("train", "bank_pool.npy", "核心", "build_bank.py", "变长模板流 fp16 拼接（训练必需）"),
    ("train", "bank_offs.npy", "核心", "build_bank.py", "每条模板在池中的起始采样"),
    ("train", "bank_lens.npy", "核心", "build_bank.py", "每条模板长度（采样）"),
    ("train", "bank_index.json", "核心", "build_bank.py", "类别元数据（name/bundle/kind），顺序 = class id"),
    ("train", "bank_meta.json", "核心", "build_bank.py", "K / sr / 时长分位 / 源采样率分布 / n_err"),

    ("data", "long44/index_long.json", "long 档", "解包 refs 或本脚本 long-index", "build_long.py 的输入"),
    ("train", "long_pool.npy", "long 档", "build_long.py", "--bg-mode long 的背景床"),
    ("train", "long_offs.npy", "long 档", "build_long.py", "每条长音起始采样"),
    ("train", "long_lens.npy", "long 档", "build_long.py", "每条长音长度（采样）"),
    ("train", "long_meta.json", "long 档", "build_long.py", "与池顺序对齐的元数据（list）"),
    ("train", "long_summary.json", "long 档", "build_long.py", "长音池概况（人读）"),

    ("train", BG_NAME, "recording 档", "本脚本 bgrec / synth 自动", "--bg-mode recording 的背景"),
    ("data", "nl_mono.wav", "recording 档", "真实录音", "bgrec 的录音源（候选之一）"),

    ("train", "variants.npy", "可选增广", "gen_variants.py", "增广变体池（没有则用干净模板）"),
    ("train", "variants_lens.npy", "可选增广", "gen_variants.py", "每个变体的长度"),
    ("train", "variants_ncc.npy", "可选增广", "gen_variants.py", "每个变体的质检 NCC"),

    ("train", "clusters_512.npy", "不需要", "make_clusters.py", "--label-mode template 时不用"),
    ("train", "clusters_512.json", "不需要", "make_clusters.py", "--label-mode template 时不用"),
]
GROUP_ORDER = ["核心", "long 档", "recording 档", "可选增广", "不需要"]


def _fmt_size(p: str) -> str:
    try:
        if os.path.isdir(p):
            n = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(p) for f in fs)
            return "%.1f MB" % (n / 1e6)
        return "%.1f MB" % (os.path.getsize(p) / 1e6)
    except OSError:
        return "--"


def _bank_line(tr: str) -> str:
    p = os.path.join(tr, "bank_meta.json")
    if not os.path.isfile(p):
        return "  bank: （无 bank_meta.json）"
    m = json.load(open(p, encoding="utf-8"))
    return ("  bank: K=%d  %.1f 分钟  时长 min/p50/p90/max = %s/%s/%s/%s s  "
            "源采样率 %s  n_err=%s"
            % (m.get("n", 0), m.get("total_min", 0.0), m.get("dur_min"), m.get("dur_p50"),
               m.get("dur_p90"), m.get("dur_max"), m.get("src_sr_hist"), m.get("n_err")))


def _long_line(tr: str) -> str:
    p = os.path.join(tr, "long_summary.json")
    if os.path.isfile(p):
        m = json.load(open(p, encoding="utf-8"))
        return ("  long: %d 条  %.1f 分钟  kinds=%s  dur p50/max = %s/%s s  cap=%s bad=%s"
                % (m.get("n", 0), m.get("total_min", 0.0), m.get("kinds"),
                   m.get("dur_p50"), m.get("dur_max"), m.get("cap"), m.get("bad")))
    lp = os.path.join(tr, "long_lens.npy")
    if os.path.isfile(lp):
        lens = np.load(lp)
        return ("  long: %d 条  %.1f 分钟  dur max = %.1f s（无 long_summary.json）"
                % (len(lens), float(lens.sum()) / SR / 60.0, float(lens.max()) / SR))
    return "  long: （无长音池）"


def _sr_line(cfg, tr: str) -> tuple[str, bool]:
    core = _core_sr(cfg)
    metas = {}
    for name, key in (("bank_meta.json", "sr"), ("long_summary.json", "sr")):
        p = os.path.join(tr, name)
        if os.path.isfile(p):
            d = json.load(open(p, encoding="utf-8"))
            if isinstance(d, dict) and key in d:
                metas[name] = d[key]
    seen = {"train/core.py": core, **metas}
    ok = (core == SR) and all(v == SR for v in metas.values())
    return ("  采样率: %s  -> %s" % ("  ".join("%s=%s" % kv for kv in seen.items()),
                                     "一致" if ok else "★不一致（先修这个再开训）")), ok


def status(cfg) -> int:
    root = _atom_root(cfg)
    tr = _train_dir(cfg)
    print("=" * 76)
    print("训练资产状态    SR=%d    train=%s" % (SR, tr))
    print("                atom_root=%s" % root)
    print("=" * 76)
    found: dict = {}
    for group in GROUP_ORDER:
        rows = [r for r in ROWS if r[2] == group]
        if not rows:
            continue
        print("[%s]" % group)
        for where, rel, _g, who, why in rows:
            p = os.path.join(tr, rel) if where == "train" else os.path.join(root, rel)
            e = os.path.exists(p)
            found[rel] = e
            print("  %s %-24s %-10s %-28s %s"
                  % ("OK  " if e else "缺失", rel, _fmt_size(p) if e else "--", who, why))
    print("-" * 76)
    print("池概况")
    print(_bank_line(tr))
    print(_long_line(tr))
    srl, sr_ok = _sr_line(cfg, tr)
    print(srl)
    print("-" * 76)
    print("结论")
    core_missing = [r[1] for r in ROWS if r[2] == "核心" and not found.get(r[1])]
    if core_missing:
        print("  × 核心资产缺 %d 个：%s" % (len(core_missing), " ".join(core_missing)))
        print("    -> cd %s && python build_bank.py" % tr)
    else:
        print("  OK 核心齐：可以开训（--bg-mode mixed / noise / silence）")
    if not sr_ok:
        print("  × 采样率不一致：core.py 与库的 SR 必须相同，否则波形会按错的采样率使用")
    long_missing = [r[1] for r in ROWS if r[2] == "long 档" and not found.get(r[1])]
    print("  %s --bg-mode long：%s"
          % ("OK" if not long_missing else "×",
             "已就绪" if not long_missing else "缺 " + " ".join(long_missing)
             + "（assets long-index -> python build_long.py）"))
    bg_ok = found.get(BG_NAME) or found.get("nl_mono.wav")
    print("  %s --bg-mode recording：%s"
          % ("OK" if bg_ok else "×",
             "%s 已就绪" % BG_NAME if found.get(BG_NAME)
             else ("可生成（有录音源，跑 assets bgrec）" if bg_ok else "缺录音源与 " + BG_NAME)))
    opt_missing = [r[1] for r in ROWS if r[2] == "可选增广" and not found.get(r[1])]
    if opt_missing:
        print("  · 可选未生成：%s（gen_variants.py，很慢；没有则用干净模板）"
              % " ".join(opt_missing))
    return 1 if (core_missing or not sr_ok) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("audio_inverse.postproc.assets")
    ap.add_argument("cmd", choices=["status", "bgrec", "long-index", "all"])
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--src", default="", help="bgrec 指定的录音源（覆盖候选顺序）")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = load_cfg(a.config, list(a.overrides))
    if a.cmd == "status":
        return status(cfg)
    if a.cmd in ("bgrec", "all"):
        make_bg(cfg, src=a.src)
    if a.cmd in ("long-index", "all"):
        make_long_index(cfg)
    print()
    return status(cfg)


if __name__ == "__main__":
    sys.exit(main())
