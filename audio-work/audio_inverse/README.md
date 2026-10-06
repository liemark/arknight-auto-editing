# audio_inverse

明日方舟音效 / 战斗语音的**反相抵消**：把游戏解包的音效与视频音频对齐，把匹配到的音效
反相叠回原音，使原音近乎静音。

```
x  ≈  Σ_i  g_i · T_θ(a_i)  +  background
```

不是语音分离，而是在变换族上的**稀疏逼近**；验收看**事件实例级零错配**（存在性 + atom_id + 时延）。

## 目录

| 路径 | 内容 |
|---|---|
| `train/` | 训练 / 识别 / 评测脚本（16 kHz；PCEN 前端 + 卷积 stem + 注意力栈 + onset/原型/offset 头 + RoPE；6576 类模板库；合成数据现场生成） |
| `audio_inverse/` | Python 包：配置与数据读取（`config/audio/atomlib/manifest`）+ 后处理（`postproc/`）+ 渲染链（`models/ detector/ synth/`） |
| `data/` | 音频源（`atoms/` 22328 条 / 4.6 GB）、`atomlib/`、`manifest.json` |
| `configs/` `docs/` | 配置；`docs/PIPELINE.md` 是**交接文档**，`docs/TRAIN_SCRIPTS.md` 是各脚本职责 |
| `../_unused_20260918.zip` | 归档：不参与当前流程的代码与参考（需要时解压取回） |

## 三步

```powershell
# ① 训练（cwd = train/）
cd audio-work\audio_inverse\train
..\..\..\.venv\Scripts\python.exe -X utf8 -u train.py --arch v2 --label-mode template `
  --layers 4 --d 192 --nh 4 --emb 256 --T 10 --bs 16 --amp `
  --lr 8e-4 --lr-sched cosine --warmup 500 --pool-mode mean --pool-frames 12 `
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 `
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 `
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 --bg-mode mixed `
  --steps 200000 --out ./ckpt_tmpl --best-metric top1 `
  --ckpt-every 500 --summary-every 2000 --snapshot-every 20000 `
  --logevery 100 --workers 4 --prefetch 2 --tag 4L

# ② 识别（同上目录）
..\..\..\.venv\Scripts\python.exe -X utf8 -u infer.py --ckpt ckpt_tmpl/best_4L.pt

# ③ 后处理 → 反相轨（cwd = audio_inverse/）
cd ..
..\..\.venv\Scripts\python.exe -X utf8 -u -m audio_inverse.postproc.render `
  --timeline data/atoms/timeline_nl_mono_4L_s200000.json `
  --wav data/atoms/samples/nl_mono.wav --out-dir out/postproc --gain-mode refit
```

**听 `out/postproc/residual.wav`**（= 原音 + 反相轨）。

VS Code 里对应的任务是 `AI · 00`–`AI · 15`（`AI · 05` 训练 / `AI · 14` 后处理）。

## 先读

- `docs/PIPELINE.md` —— 数据资产、任务表、三档背景的对照实验、后处理细节、
  `variants.npy` 的现状与生成方法、历史实测结论与死路。
- `docs/TRAIN_SCRIPTS.md` —— `train/` 下每个脚本干什么。

## 环境

```powershell
.\.venv\Scripts\python.exe      # 仓库根目录的 venv；torch 2.11+cu128 / RTX 4060 8GB
```

训练脚本是 16 kHz 的独立工作流（依赖 torch / numpy / scipy / audiomentations / ffmpeg）；
包侧（后处理）另需 `soundfile` `soxr` `pyyaml`。见 `pyproject.toml` 与 `requirements-cu128.txt`。
