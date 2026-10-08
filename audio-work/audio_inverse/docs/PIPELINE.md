# audio_inverse — 明日方舟音效/语音反相抵消

把游戏解包的音效 SFX / 角色战斗语音与视频音频对齐，把匹配到的音效**反相**叠回原音，
使原音近乎静音。形式化目标：

```
x  ≈  Σ_i  g_i · T_θ(a_i)  +  background
```

不是语音分离，而是在变换族上的稀疏逼近；验收看**事件实例级零错配**（存在性 + atom_id + 时延）。

---

## 目录

```
audio-work/audio_inverse/
├─ train/                训练 / 识别 / 评测（16 kHz，PCEN 前端 + 卷积 stem + 注意力栈
│   │                    + onset/原型/offset 头 + RoPE；6576 类模板库；纯合成现场生成）
│   ├─ train.py          训练主程序
│   ├─ core.py synth.py  模型定义 / 数据合成器
│   ├─ build_bank.py     模板库（6576 类，16k）
│   ├─ build_long.py     长音池（环境/剧情/干员语音，当背景用）
│   ├─ gen_variants.py   变体增广库（可选，见 §4）
│   ├─ eval.py loc_eval.py recall_at_k.py onset_diag.py   评测与诊断
│   ├─ infer.py          推理 → timeline JSON/TXT
│   ├─ test_losses.py    损失数值不变量单测（纯 CPU，秒级）
│   └─ ckpt_tmpl/        训练产物（日志 / ckpt / best / snap）
├─ audio_inverse/        Python 包
│   ├─ config.py audio.py atomlib.py manifest.py   配置与数据读取
│   ├─ postproc/         后处理：timeline → 反相轨（48 kHz 渲染链）+ 资产准备
│   ├─ detector/ models/ synth/                    渲染链与混音参数
├─ data/
│   ├─ atoms/            音频源 4.6 GB（sfx / voice / assets / real / samples）
│   ├─ atomlib/          int16 打包池 + 索引 + 统计（3.0 GB）
│   ├─ manifest.json     22328 条原子清单 + train/val/test 划分
│   └─ ckpt_v2/          早期实验留下的权重（2.0 GB，与当前流程无关）
├─ configs/  docs/  pyproject.toml
└─ ../_unused_20260918.zip   归档：不参与当前流程的代码与参考（212 条目，见 §7）
```

`.vscode/tasks.json` 里有 16 个任务，编号 `AI · 00`–`AI · 15`，见 §3。

---

## 1. 三步主流程

| 步骤 | 命令 / 任务 | 产物 |
|---|---|---|
| ① 训练 + 识别 | `train/train.py` → `train/infer.py`（AI·05 → AI·13） | `ckpt_tmpl/*.pt`、`data/atoms/timeline_*.json` |
| ② 后处理 | `audio_inverse.postproc.render`（AI·14） | `out/postproc/{cancel.wav, residual.wav, report.json}` |
| ③ 听 | 直接听 `residual.wav`（= 原音 + 反相轨） | — |

**为什么后处理单独一层**：识别模型与模板库是 **16 kHz** 的（`train/core.py: SR=16000`，
`bank_clean.npy` 也是 16 kHz 重采样）。48 kHz 原始录音里 **8–24 kHz** 那一段用 16 kHz 模板
永远抵消不掉。本包的 `atomlib` 是**原始采样率**（16k/44.1k/48k 原样混存）的池子，渲染链全程
48 kHz —— 所以识别结果必须交回本包来渲染，才可能真正做到"近乎静音"。

---

## 2. 数据资产

| 文件 | 生成者 | 实测 | 说明 |
|---|---|---|---|
| `data/atoms/` | 解包（已有） | 22328 条 / 4.6 GB | SFX 8240 + 战斗语音 14088；语音全是 16 kHz ⇒ 8 kHz 以上不可反相 |
| `data/atomlib/`、`manifest.json` | `audio_inverse.build_data` | 22328 条 | 后处理靠它取 48 kHz 原始波形 |
| `train/bank_clean.npy` `bank_lens/offs.npy` `bank_index.json` | `train/build_bank.py` | **6576 类**，37 s | 取 `sfx/index_*.json` 里 0.05–2.5 s 的片段 |
| `train/bank_96k.npy` `bank_48k.npy` | 同上 | 各 526 MB | AAC 降质档，给变体增广用（需 ffmpeg） |
| `train/bg16.npy` | `audio_inverse.postproc.assets bg16` | **323.3 s / 20.7 MB** | `v82_mono.wav` 降采样到 16k，`--bg-mode recording` 的前提 |
| `long/index_long.json` | `... assets long-index` | 14228 条 / 834 分钟 | `build_long.py` 的输入（解包脚本需要游戏安装目录，本机没有，改用现有索引拼成约定格式） |
| `train/long_pool.npy` 等 | `train/build_long.py` | **14228 条 / 1602 MB / bad=0 / 13 s** | `--bg-mode long` 的前提 |
| `train/variants.npy` | `train/gen_variants.py` | **未生成**（可选） | 见 §4 |
| `train/clusters_512.json` | 自带 | — | 只有 `--label-mode cluster` 才用；本项目一律 `template` |

重建顺序：`AI·02 → AI·03 → AI·04`（都已跑完，可跳过）。

---

## 3. 任务表（`.vscode/tasks.json`）

**准备**：`AI·00` 环境自检（跑 `test_losses.py`）· `AI·01` 资产状态 · `AI·02` 造 bg16/long-index ·
`AI·03` 模板库 ✅ · `AI·04` 长音池 ✅

**训练**（模型/损失/采样参数完全相同，**只有背景不同** ⇒ 背景的单变量对照）：

| 任务 | 背景 | 说明 |
|---|---|---|
| `AI·05` ★ | `mixed` | 80% 白/粉噪 + 20% 数字静音。参数照抄 `run_night.sh`，去掉 `--resume/--resize`，`--layers` 定回 4（8 层实测倒退：oracle top-1 58.10% → 47.91%） |
| `AI·06` ★ | `recording` | 拿真实录音当背景。`run_bgrec.sh` 写好的实验，当年**没有结果文件**，`bg16.npy` 现已就位 |
| `AI·07` | `long` | 长音床（环境/剧情）+ 0.65 概率叠干员战斗语音，同样没跑过 |
| `AI·08` | 续训 | `--resume latest`（只 glob `ckpt_*.pt`，不会误取 best/snap） |

**评测**：`AI·09` `eval.py` 分档 + PCEN 消融 · `AI·10` `loc_eval.py` 事件级 P/R/F1 + 召回硬上限 ·
`AI·11` `recall_at_k.py` oracle recall@k · `AI·12` `onset_diag.py` 域差诊断

**推理与后处理**：`AI·13` `infer.py` → timeline · `AI·14` ★ `postproc.render` → 反相轨 + PSR 报告

---

## 4. `variants.npy`：当前**没有**生成（可选项）

**它是什么**：`train/gen_variants.py` 用 audiomentations 为每条模板预生成 V 个"音色一致"的
增广变体，**离线**产物、训练时直接 memmap 读；`synth.py` 在文件存在时优先用它
（`synth.py:178`），不存在则退回 `bank_clean/96k/48k` 三档干净模板（仍会在线跑 `chan_fx` 增广）。

**是否参与训练**：参与。`train.py` 构造 `SynthDS` 时没有传 `use_variants`（默认 `True`），
只有 `no_fx=True` 才会关闭。所以没有它 = 少一层离线音色增广（当年默认配方是吃它的）。

**当年的参数**（`train/variants_meta.json` 里记着）：`V=12 / profile=full / qc_ncc=0.60`
→ 6576×12 = **78912 条 × 16000 采样 fp16 ≈ 2.53 GB**，预计 20–30 分钟（纯 CPU，多进程）。

```powershell
cd audio-work\audio_inverse\train
..\..\..\.venv\Scripts\python.exe -X utf8 -u gen_variants.py --V 12 --profile full --workers 10
```

**生成前必须知道的三件事**（都已实测/已修）：

1. `profile=full` 单独用会毁掉模板（实测 NCC p50 = **0.02**）。`gen_variants.py` 里自带
   **NCC 质检**：低于 0.60 就用 `safe` 规格重做，所以最终统计才是 p50 0.95 / 99.5%>0.5。
   本机 audiomentations 0.43.1 与 `aug_spec` **完全兼容**（full 17/17、tamed 16/16、
   safe 11/11 个变换全部装上，0 跳过）。
2. **短模板的静默退化已修**：`LoudnessNormalization`（pyloudnorm）要求 ≥400 ms，而
   `bank_lens` 里 **11.5% 的模板短于 400 ms**。以前这个异常被 `_work()` 的 `except` 吞掉，
   退化成 `y = w.copy()`，其 NCC 恰好 = 1.0 被当成"合格变体" —— 那批模板等于没有增广，日志里
   还看不出来。现在 `aug_spec.MIN_SAMPLES` 按长度过滤该变换，`_work` 把兜底次数带回主进程，
   结尾打印 `兜底(原样复制) N/78912 条 (x%)`；`variants_meta.json` 也记下 `minlen` / `n_fallback`。
3. **NCC 门限原来只是"两条候选挑更好的那条"，没有硬门限**。已补上**硬兜底**：full 与 safe
   都不达标时，回退到干净模板本身（NCC 记 1.0）并计入 `n_rejected`。实测触发率 **1/400 = 0.25%**；
   加上它之后，"入库变体恒满足 `NCC ≥ 0.60`"成为不变量 —— 300 条抽样：
   `min 0.61 / p10 0.81 / p50 0.95`，**低于门限 0 条**。

**本轮移除的一段**：原来的"故意 2 倍速"（`rng.random() < 0.08`，施加在质检**之后**）已删除。
实测（10 条模板取中位；判据用模型前端 PCEN 谱的逐帧余弦，也就是识别头实际看到的东西）：

| 变速 | 时长比 | 波形 NCC | PCEN 谱余弦 |
|---|---|---|---|
| 不变速（对照） | 1.00 | 1.000 | 1.000 |
| 抗混叠 0.90 | 0.90 | 0.001 | 0.723 |
| 抗混叠 1.10 | 1.10 | −0.032 | 0.720 |
| 抗混叠 1.25 | 1.25 | 0.020 | 0.653 |
| 原版做法 2.00（无抗混叠抽取） | 0.50 | −0.039 | **0.350** |

两个结论：

- **波形 NCC 对任何变速都会立刻归零**（连 0.90 倍速都是 0.001）—— 它是固定时间对齐的判据，
  天生不能给变速样本做质检。这解释了原作者为什么把 2 倍速放在质检**之后**：代价是它
  **绕过了唯一的质检**，从未被验证过。
- 该看的是 **PCEN 谱余弦**：2.0 只有 0.35（身份已破 = 标签噪声）；而真实素材的变速域
  0.90–1.10 有 0.72–0.78，与正常变体（0.80–0.95）同级。那一段由训练时的在线 `chan_fx`
  （±0.6%）覆盖；要更宽应当在 `chan_fx` 里调，而不是往变体库里塞伪影
  （原版是**无抗混叠的线性插值抽取**，注释里的 `unchanged pitch` 也与实际不符）。

---

## 5. 后处理细节（`audio_inverse.postproc.render`）

```
train/infer.py → timeline_*.json        （onset 峰 → 不应期 → 240ms 池化 → 原型 top-k → NNLS 重拟合）
                        │  class = 模板下标（0..6575）
                        ▼
postproc.render
   1. class → 本包 atom id   用 (name, bundle) 作键；实测 **6576/6576 全部命中、无歧义**
   2. 增益                    默认 `--gain-mode refit`：用本包渲染链**重新最小二乘拟合**
                              （timeline 里的 gain 口径不同：bank_clean 的模板没做 RMS 归一化）
   3. 渲染反相轨              复用 models/synthesize.py::render_events(invert=True)
   4. 报告                    PSR + **分子带 PSR**（20-500 / 500-2k / 2k-8k / 8k-16k / 16k-24k）
```

分子带 PSR 是"有没有真的全带宽抵消"的照妖镜：高频那一条若为负，说明反相在该频段**增加了**
能量（16 kHz 模板反相 48 kHz 录音时的典型症状）。

---

## 6. 历史实测结论（跑之前先知道）

- **背景域差是端到端失败的主因**：同一个模型在合成窗口上 onset 概率 p50=0.001（10.1% 的帧 >0.5），
  在真实录音 `nl_mono.wav` 上 **p50=0.448、45.8% 的帧 >0.5**；而孤立模板 0.011、数字静音 0.001
  —— 不是检测头坏了，是它没见过真实背景。症状：自适应阈值静默贴到上限 0.90，吐出上千个事件、
  NNLS 增益 0.00–0.11。`--bg-mode recording` 这个开关存在却从没被用过 ⇒ 就是 `AI·06`。
- **PCEN 是决定性组件**：关掉后 hard 档 top-1 51.2%→1.8%（512 类）、86.1%→1.2%（6576 类）。
- **验收必须用事件级指标**：帧级 F1 测不出"每个音效在哪里"；而且把 onset 目标改尖会让帧 F1
  **下降**（真值变窄），别被它误导。阈值也应按事件 F1 选。
- **99% 的 onset 假峰落在正在响的事件内部**（中位相对位置 0.49）——`offset` 边界头与
  `--peak-w` hinge 的来由。
- **`empty_frac` 修虚警**：加之前精确率 0.559–0.657（dense 档过检 73%），加了之后约 0.80。
- **死路（已归档，不要重走）**：谱峰对指纹路线（净负收益，混音里 hash 重合度 0.99→0.43→0.08）、
  512 簇模式、激进增广（未加质检时 73% 的变体与自身标签不相关、中位 NCC 0.061、top-1 卡在 2–3%）、
  8 层（比 4 层低 10.19 点）、用 presence-run 当池化窗。
- **分档结果**（6576 类，T=10，n=1500/档，seed 4242）：step 48000 时 top-1
  easy 96.9% / mid 95.5% / hard 89.8% / **dense 69.6%**，事件 F1 0.863/0.958/0.926/0.866。
  原始报告在 `docs/eval_reports/`。

---

## 7. 归档 zip：`audio-work/_unused_20260918.zip`

212 个条目，装的是**不参与当前流程**的东西，需要时解压取回即可：

| 组 | 内容 |
|---|---|
| `ref/` | 早期 lab 源码副本、更早的 v1 包归档、旧目录布局说明 |
| `docs_old/` | 描述已废弃路线的文档（`PROJECT.md`、`NCC_RETIREMENT.md`、`RETRIEVAL_PRECHECK.md`、旧 README） |
| `pkg_unused/` | 包里不参与运行的模块（早期 48k 检测器与检索路线、其诊断脚本、旧合成数据管线） |
| `train_unused/` | `train/` 里不参与工作流的脚本：指纹路线（`fp_*` `hash_*`）、一次性探针（`*_probe`、`pool_sweep`、`timing_probe`…）、`build_confusion`、`warmstart_template`、`launch_detached` 等 |

**注意**：`train_unused/` 里被保留下来的脚本**不 import** 其中的任何模块（已核对）；
唯一例外是 `gen_variants.py` 需要 `aug_spec.py`，后者已放回 `train/`。

---

## 8. 常用命令

```powershell
# 训练（= AI·05；cwd 必须是 train/）
cd audio-work\audio_inverse\train
..\..\..\.venv\Scripts\python.exe -X utf8 -u train.py `
  --arch v2 --label-mode template --layers 4 --d 192 --nh 4 --emb 256 `
  --T 10 --bs 16 --amp --lr 8e-4 --lr-sched cosine --warmup 500 `
  --pool-mode mean --pool-frames 12 `
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 `
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 `
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 `
  --bg-mode mixed --steps 200000 --out ./ckpt_tmpl --best-metric top1 `
  --ckpt-every 500 --summary-every 2000 --snapshot-every 20000 `
  --logevery 100 --workers 4 --prefetch 2 --tag 4L

# 后处理（= AI·14；cwd 换回 audio_inverse）
cd ..
..\..\.venv\Scripts\python.exe -X utf8 -u -m audio_inverse.postproc.render `
  --timeline data/atoms/timeline_nl_mono_4L_s200000.json `
  --wav data/atoms/samples/nl_mono.wav --out-dir out/postproc --gain-mode refit
```

> `--workers 4 --prefetch 2` 是实测最佳：12×4 = 48 个 batch 在途会把 mmap 页缓存挤出去，
> 吞吐从 8.4 it/s 衰减到 5.5；降到 4×2 后是 6.45 → 10.98 it/s。
>
> 各脚本职责见 `docs/TRAIN_SCRIPTS.md`（原文，路径按 `train/` 理解）。
