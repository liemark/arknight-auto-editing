# audio_lab — 游戏音效检测器训练包

从解包的音效 / 语音素材出发，训练一个**事件定位 + 模板识别**模型：

```
输入 16 kHz 音频，10 秒窗口
输出 每一帧是不是事件起始(onset) + 事件边界(offset) + 事件属于哪个模板(class)
```

模型 = PCEN 梅尔前端 + 卷积 stem + 注意力栈（RoPE）+ onset 头 + 原型余弦识别头 + offset 头。
训练数据**现场合成**：从模板库里抽音效、按随机间隔重复、叠加背景、加通道失真，不需要人工标注。

---

## 目录

```
audio_lab/
├─ alab.py                音频读写与常用 DSP（FFT/带通/峰值/内积，被 ml/ 下的脚本 import）
├─ ml/
│   ├─ core.py            模型定义（FrontEnd / Enc / PoolAttn / OffsetHead / Detector）
│   ├─ synth.py           训练数据合成器（模板 → 混合音频 + 帧级标签）
│   ├─ train.py           训练主程序
│   ├─ test_losses.py     损失不变量单测（纯 CPU，秒级，改损失前后各跑一次）
│   ├─ build_bank.py      模板库构建（默认 16 kHz，2.5 s 上限）
│   ├─ build_long.py      长音池构建（环境音/剧情/人声，供背景使用）
│   ├─ gen_variants.py    变体增广库构建（可选）
│   ├─ aug_spec.py        增广规格与一致性检查
│   ├─ make_index.py      从 wav 目录生成索引 JSON
│   ├─ eval.py            分档评测（easy/mid/hard/dense + PCEN 消融）
│   ├─ loc_eval.py        事件级定位 P/R/F1 + 召回上限
│   ├─ recall_at_k.py     全库排序 recall@k（识别上限）
│   ├─ onset_diag.py      onset 概率分布诊断（合成 / 真实 / 静音）
│   └─ infer.py           推理 → 事件时间轴 JSON/TXT
├─ sfx/                   （放你的 SFX wav + index_*.json）
├─ long/                  （放你的长音 wav + index_long.json）
└─ docs/TRAINING.md       完整训练手册（先读这个）
```

`ml/` 同时是**工作目录**：模板库、长音池、变体库、checkpoint 都写在这里。

---

## 快速开始

```powershell
# 0) 依赖：python>=3.11, torch, numpy, scipy, audiomentations；模板库的 AAC 档需要 ffmpeg
pip install torch numpy scipy audiomentations

# 1) 用任意解包工具把音频导成 wav，然后生成索引（见 docs/TRAINING.md §2）
python ml/make_index.py sfx --dir D:\sfx\player --group player --out sfx/index_player.json
python ml/make_index.py sfx --dir D:\sfx\root   --group root   --out sfx/index_root.json

# 2) 模板库（约 1 分钟，产出 sfx 里 0.05–2.5 s 的模板）
python ml/build_bank.py

# 3) 自检 + 训练
python ml/test_losses.py
python ml/train.py --arch v2 --label-mode template --layers 4 --d 192 --nh 4 --emb 256 `
  --T 10 --bs 16 --amp --lr 8e-4 --lr-sched cosine --warmup 500 `
  --pool-mode mean --pool-frames 12 `
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 `
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 `
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 `
  --bg-mode mixed --steps 200000 --out ./ckpt_tmpl --best-metric top1 `
  --ckpt-every 500 --summary-every 2000 --snapshot-every 20000 `
  --logevery 100 --workers 4 --prefetch 2 --tag run1

# 4) 评测 / 推理
python ml/eval.py       --ckpt ckpt_tmpl/best_run1.pt --T 0 --n 1500
python ml/loc_eval.py   --ckpt ckpt_tmpl/best_run1.pt --T 0 --windows 900
python ml/recall_at_k.py --ckpt ckpt_tmpl/best_run1.pt --T 0 --windows 900
python ml/infer.py      --ckpt ckpt_tmpl/best_run1.pt --wav <真实录音.wav>
```

工作目录约定：`ml/` 里的脚本都按 `ML = 脚本所在目录`、`D = ML 的上一层` 解析路径，
所以请保持上面的目录结构，并从 `audio_lab/` 或 `audio_lab/ml/` 下运行。

---

## 先读

**`docs/TRAINING.md`** —— 数据准备、索引字段规范、推荐配方、三档背景的对照实验、
评测判据、以及"什么会导致什么"的注意事项（用问题 → 方案的方式写）。
