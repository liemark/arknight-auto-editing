"""构建数据: manifest.json + atomlib/。P1 入口。

    python -m audio_inverse.build_data --config configs/data_atoms.yaml
    python -m audio_inverse.build_data --rebuild-manifest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .atomlib import load_or_build_atomlib
from .config import load_cfg
from .manifest import build_manifest, review_groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("build_data", description="构建清单与原子库")
    ap.add_argument("--config", default="data_atoms.yaml")
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--rebuild-atomlib", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value 覆盖配置")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config, args.overrides)
    data_root = cfg.abspath("data_root")
    atom_root = cfg.abspath("atom_root")
    os.makedirs(data_root, exist_ok=True)

    man_path = os.path.join(data_root, "manifest.json")
    rebuild_man = args.rebuild_manifest or bool(cfg.get("build.rebuild_manifest", False))
    if os.path.exists(man_path) and not rebuild_man:
        from .manifest import Manifest
        man = Manifest.load(man_path)
        print(f"[build_data] 复用已有清单 {man_path} ({man.n} 原子)")
    else:
        print(f"[build_data] 扫描 {atom_root} ...")
        man = build_manifest(
            atom_root, man_path,
            indexes=cfg.get("atom_indexes"),
            ratios=tuple(cfg.get("split_ratios", [0.8, 0.1, 0.1])),
            salt=str(cfg.get("split_salt", "ai-v1")),
            verify=bool(cfg.get("build.verify_wav_headers", True)),
        )
    if cfg.get("build.review_groups", True):
        print("[build_data] 分组划分抽样:")
        print(review_groups(man))

    lib_dir = cfg.abspath("atomlib_dir")
    lib = load_or_build_atomlib(man, lib_dir,
                               rebuild=args.rebuild_atomlib
                               or bool(cfg.get("build.rebuild_atomlib", False)))
    print(f"[build_data] 原子库 {len(lib)} 条 @ {lib_dir}")

    # 自检: 随机抽样比对 池内波形 vs 原始 WAV (防止 offset/长度写错)
    rng = np.random.default_rng(0)
    probe = rng.choice(len(lib), size=min(24, len(lib)), replace=False)
    bad = 0
    for i in probe:
        x_pool, sr_pool = lib.raw(int(i))
        from .audio import read_wav
        x_src = read_wav(man.atoms[int(i)].file, mono=True)
        m = min(x_pool.size, x_src.size)
        if m == 0:
            bad += 1
            continue
        err = float(np.max(np.abs(x_pool[:m] - x_src[:m])))
        if err > 1e-4 or sr_pool != man.atoms[int(i)].sr:
            bad += 1
            print(f"  [自检失败] id={i} sr={sr_pool}/{man.atoms[int(i)].sr} maxerr={err:.2e}")
    print(f"[build_data] 抽样自检 {len(probe)} 条, 失败 {bad}")

    if cfg.get("report.show_voice_hi8k", True):
        k = lib.stat_names.index("hi8k_ratio")
        voice = np.array([a.kind == "voice_battle" for a in man.atoms])
        if voice.any():
            hi = lib.stats[voice, k]
            print(f"[build_data] 语音原子 8kHz 以上能量占比: "
                  f"max={hi.max():.4f} p99={np.percentile(hi,99):.4f} "
                  f"(≈0 证实 16k 带宽上限, 该频段无法反相)")

    print(f"[build_data] 完成。stats = {json.dumps(man.stats, ensure_ascii=False)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
