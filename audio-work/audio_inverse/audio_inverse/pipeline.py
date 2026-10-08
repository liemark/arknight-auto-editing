"""audio_inverse 的最小端到端框架：一个入口, 把各步骤按序串起来。

设计原则：**薄驱动**——这里不重新实现任何东西, 只是用固定好的默认值去调用已有的脚本/模块,
所以每一步都能单独跑、单独 debug, 不会出现"框架里跑得通、单独跑不通"。

    python -m audio_inverse.pipeline status              # 资产状态（能不能开跑）
    python -m audio_inverse.pipeline assets              # 补 bg44.npy + long44/index_long.json
    python -m audio_inverse.pipeline bank                # 建模板库（约 2.5 分钟）
    python -m audio_inverse.pipeline longpool            # 建长音池（约 4 分钟）
    python -m audio_inverse.pipeline train --tag 44k2    # 训练（默认 6G 显存预算）
    python -m audio_inverse.pipeline eval                # 分档评测（自动跟随 checkpoint）
    python -m audio_inverse.pipeline infer --wav data/atoms/v82_mono.wav
    python -m audio_inverse.pipeline export --timeline <json> --wav <wav>
    python -m audio_inverse.pipeline render --timeline <json> --wav <wav>

训练默认值（按本机 8GB 卡实测定的 6G 预算）：
    attn 编码器 T=20s: bs=8→1.85GB  bs=12→2.74GB  bs=16→3.62GB  bs=20→4.50GB
    所以默认 --bs 16（3.62GB）；显存紧张就 12/8，富余就 20。
    编码器可选 --enc attn|crnn|cnn|bcresnet（实测质量 attn > crnn > cnn/bcresnet，成本反过来）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PKG = os.path.dirname(os.path.abspath(__file__))          # .../audio_inverse/audio_inverse
ROOT = os.path.dirname(PKG)                                # .../audio_inverse
TRAIN = os.path.join(ROOT, "train")
PY = sys.executable

TRAIN_DEFAULTS = [
    "--arch", "v2", "--enc", "attn", "--label-mode", "template",
    "--layers", "4", "--d", "192", "--nh", "4", "--emb", "256",
    "--max-ev", "192", "--T", "20", "--bs", "16", "--amp",
    "--lr", "8e-4", "--lr-sched", "cosine", "--warmup", "500",
    "--pool-mode", "mean", "--pool-frames", "12",
    "--onset-shape", "gauss", "--onset-len", "5", "--onset-sigma", "2.5",
    "--peak-w", "1.0", "--peak-win", "25", "--peak-margin", "2.0",
    "--bnd-loss", "bce", "--pos-weight", "10", "--off-w", "1.0", "--le-w", "0.5",
    "--bg-mode", "mixed", "--steps", "200000", "--out", "./ckpt_tmpl",
    "--best-metric", "top1", "--ckpt-every", "500", "--summary-every", "2000",
    "--snapshot-every", "20000", "--logevery", "100", "--workers", "10", "--prefetch", "4",
]
ENC_CHOICES = ("attn", "crnn", "cnn", "bcresnet")


def _set_default(argv: list, flag: str, value: str) -> list:
    """框架默认值可被命令行覆盖。"""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            argv[i + 1] = value
            return argv
    return argv + [flag, value]


def _run(argv: list, cwd: str) -> int:
    print("$ (cwd=%s)\n  %s" % (cwd, " ".join(argv)), flush=True)
    r = subprocess.run(argv, cwd=cwd)
    if r.returncode != 0:
        print("!! 退出码 %d" % r.returncode, flush=True)
    return r.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("audio_inverse.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="资产状态（这一步就是「现在能不能开跑」的答案）")
    sub.add_parser("assets", help="补 bg44.npy + long44/index_long.json")
    sub.add_parser("bank", help="建模板库 bank_pool/offs/lens/index/meta")
    sub.add_parser("longpool", help="建长音池 long_pool/offs/lens/meta")

    p = sub.add_parser("train", help="训练（默认 6G 显存预算, bs=16）")
    p.add_argument("--tag", required=True, help="本次训练的标签（决定 ckpt/log 文件名）")
    p.add_argument("--enc", default="attn", choices=ENC_CHOICES)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--bg-mode", default="mixed", choices=["mixed", "noise", "silence", "recording", "long"])
    p.add_argument("extra", nargs="*", help="其余参数原样透传给 train.py")

    p = sub.add_parser("eval", help="分档评测（池化口径自动跟随 checkpoint）")
    p.add_argument("--ckpt", default="", help="默认取 train/ckpt_tmpl/best_*.pt 最新那个")
    p.add_argument("--n", type=int, default=800, help="每档窗口数")

    p = sub.add_parser("infer", help="推理 -> timeline（采样级重拟合已默认打开）")
    p.add_argument("--wav", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--k", type=int, default=4, help="候选数; 4 比 8 快 25 倍(NNLS 枚举 2^n-1)")

    for name, extra_help in (("export", "试听轨：matched/cuts/pairs/resid"),
                             ("render", "反相轨 + PSR 报告")):
        p = sub.add_parser(name, help=extra_help)
        p.add_argument("--timeline", required=True)
        p.add_argument("--wav", required=True)
        p.add_argument("--out-dir", default="out/export" if name == "export" else "out/postproc")

    a = ap.parse_args(argv)

    if a.cmd == "status":
        return _run([PY, "-X", "utf8", "-u", "-m", "audio_inverse.postproc.assets", "status"], ROOT)
    if a.cmd == "assets":
        return _run([PY, "-X", "utf8", "-u", "-m", "audio_inverse.postproc.assets", "all"], ROOT)
    if a.cmd == "bank":
        return _run([PY, "-X", "utf8", "-u", "build_bank.py"], TRAIN)
    if a.cmd == "longpool":
        return _run([PY, "-X", "utf8", "-u", "build_long.py"], TRAIN)

    if a.cmd == "train":
        av = list(TRAIN_DEFAULTS)
        av = _set_default(av, "--enc", a.enc)
        av = _set_default(av, "--bs", str(a.bs))
        av = _set_default(av, "--T", str(a.T))
        av = _set_default(av, "--steps", str(a.steps))
        av = _set_default(av, "--bg-mode", a.bg_mode)
        # 0.0 表示"auto = 整个窗口", 即素材可以从窗口前开始播（窗口内听到中段）
        av += ["--left-pad", "0.0", "--tag", a.tag]
        av += list(a.extra or [])
        return _run([PY, "-X", "utf8", "-u", "train.py"] + av, TRAIN)

    if a.cmd == "eval":
        ck = a.ckpt
        if not ck:
            d = os.path.join(TRAIN, "ckpt_tmpl")
            cands = [os.path.join(d, f) for f in os.listdir(d)
                     if f.startswith("best_") and f.endswith(".pt")] if os.path.isdir(d) else []
            if not cands:
                print("train/ckpt_tmpl 下没有 best_*.pt，请先训练或用 --ckpt 指定", file=sys.stderr)
                return 2
            ck = max(cands, key=os.path.getmtime)
            print("自动选 checkpoint: %s" % os.path.basename(ck), flush=True)
        return _run([PY, "-X", "utf8", "-u", "eval.py", "--ckpt", ck, "--T", "0", "--n", str(a.n)], TRAIN)

    if a.cmd == "infer":
        av = ["infer.py", "--wav", a.wav, "--k", str(a.k),
              "--max-refit-s", "2.0", "--pool-frames", "12", "--refr", "1"]
        if a.ckpt:
            av += ["--ckpt", a.ckpt]
        return _run([PY, "-X", "utf8", "-u"] + av, TRAIN)

    if a.cmd == "export":
        return _run([PY, "-X", "utf8", "-u", "-m", "audio_inverse.postproc.export",
                     "--timeline", a.timeline, "--wav", a.wav, "--out-dir", a.out_dir,
                     "--mode", "all", "--pair-top", "12"], ROOT)
    if a.cmd == "render":
        return _run([PY, "-X", "utf8", "-u", "-m", "audio_inverse.postproc.render",
                     "--timeline", a.timeline, "--wav", a.wav, "--out-dir", a.out_dir,
                     "--gain-mode", "refit", "--top-per-event", "4", "--gain-thr", "0.05"], ROOT)
    return 2


if __name__ == "__main__":
    sys.exit(main())
