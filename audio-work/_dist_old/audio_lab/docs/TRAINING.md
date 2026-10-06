# 训练手册

本文按"数据 → 训练 → 评测 → 推理"的顺序给出完整流程，并在 §6 集中列出**参数取值依据**
与**会导致什么后果**的注意事项（每条都是"问题 → 方案"的形式，可直接照着做）。

---

## 1. 环境

```powershell
python -m pip install torch numpy scipy audiomentations
```

| 依赖 | 用途 | 备注 |
|---|---|---|
| torch | 训练/推理 | CUDA 版；训练前会做 CUDA 预检 |
| numpy / scipy | 合成、DSP | scipy.fft 在渲染管线里比 numpy.fft 快 2–4 倍，代码已用 |
| audiomentations | 变体增广（可选） | 只有 `gen_variants.py` 用 |
| ffmpeg | 模板库的 AAC 降质档 | 缺失时 `build_bank.py` 的降质档会跳过，不影响干净模板 |

工作目录约定：`ml/` 里的脚本按 `ML = 脚本所在目录`、`D = ML 的上一层` 解析路径。
请保持 `audio_lab/{alab.py, ml/, sfx/, long/}` 结构。

---

## 2. 数据准备

### 2.1 目录与索引字段

```
audio_lab/
├─ alab.py
├─ sfx/                    ← SFX 模板的 wav + index_<group>.json
└─ long/                   ← 长音素材的 wav + index_long.json
```

**SFX 索引**（`sfx/index_<group>.json`，JSON **object**，键 = 文件名）：

| 字段 | 说明 |
|---|---|
| `file` | wav 路径（绝对路径最稳） |
| `name` | 模板名（用于日志/清单） |
| `bundle` | 来源分组（同一角色/武器的多个音效通常同 bundle） |
| `dur` | 时长（秒） |
| `sr` | 采样率 |

`build_bank.py` 会遍历索引，取 `0.05 s ≤ dur ≤ 2.5 s` 的条目作为模板，并把 `--group` 写进清单。

**长音索引**（`long/index_long.json`，JSON **array**）：

| 字段 | 说明 |
|---|---|
| `file` / `name` / `bundle` / `dur` / `sr` | 同上 |
| `kind` | `music` / `ambience` / `dialog` / `long_se` → 当作**长音床**；`voice_battle` → 当作**人声** |
| `lang` / `loop` | 语言、是否循环（可选，写进元数据） |

### 2.2 生成索引

解包工具因人而异（AssetStudio、UnityPy 等），本包只负责最后一跳：

```powershell
# SFX：四组，分别生成
python ml/make_index.py sfx --dir D:\sfx\player --group player --out sfx/index_player.json
python ml/make_index.py sfx --dir D:\sfx\root   --group root   --out sfx/index_root.json
python ml/make_index.py sfx --dir D:\sfx\custom_se --group custom_se --out sfx/index_custom_se.json
python ml/make_index.py sfx --dir D:\sfx\enemy  --group enemy  --out sfx/index_enemy.json

# 长音：环境音 / 剧情音 / 战斗语音，可多次追加到同一个文件
python ml/make_index.py long --dir D:\long\ambience --kind ambience --out long/index_long.json
python ml/make_index.py long --dir D:\long\voice    --kind voice_battle --lang jp --append --out long/index_long.json
```

`--name-from dir` 用目录名当 bundle；`--min-dur/--max-dur` 可按长度过滤；
`--relative` 让 `file` 写相对路径（默认绝对）。

### 2.3 模板库（必需）

```powershell
python ml/build_bank.py
```

产出（写在 `ml/` 下）：

| 文件 | 内容 |
|---|---|
| `bank_clean.npy` | `[K, 40000]` float16，每条模板 16 kHz 波形（右侧补零） |
| `bank_lens.npy` | `[K]` int32，每条的真实长度 |
| `bank_offs.npy` | `[K]` int64，在拼接流里的偏移 |
| `bank_index.json` | `[K]` 元数据（name / bundle / group / dur） |
| `bank_96k.npy` / `bank_48k.npy` | 同一批模板经 AAC(96k/48k) 编解码后的版本，供变体增广制造编解码多样性 |

`K` = 通过 0.05–2.5 s 过滤的模板数。

### 2.4 长音池（用 `--bg-mode long` 时需要）

```powershell
python ml/build_long.py
```

产出 `long_pool.npy`（float16 拼接流）+ `long_lens.npy` / `long_offs.npy` / `long_meta.json`。
训练时由 `synth.py` 按 `kind` 分成"床"和"人声"两类抽样。

### 2.5 真实录音背景（用 `--bg-mode recording` 时需要）

把一段真实录音（BGM + 环境音 + 人声都在的那种）重采样成 16 kHz 单声道，存成 `ml/bg16.npy`：

```python
import numpy as np, wave
x, sr = <读取你的录音为 float32, 单声道>
x = <重采样到 16000>          # 线性插值即可，先做 30–7500 Hz 带通
np.save("ml/bg16.npy", x.astype(np.float32))
```

### 2.6 变体增广库（可选）

```powershell
python ml/gen_variants.py --V 12 --profile full --workers 10
```

为每条模板预生成 `V` 个音色变体，训练时随机取用（`synth.py` 检测到 `variants.npy` 就优先用它）。
`V=12` 时约 `K*12` 条 × 16000 采样 fp16。

---

## 3. 训练

### 3.1 推荐配方

```powershell
python ml/train.py `
  --arch v2 --label-mode template `
  --layers 4 --d 192 --nh 4 --emb 256 `
  --T 10 --bs 16 --amp `
  --lr 8e-4 --lr-sched cosine --warmup 500 `
  --pool-mode mean --pool-frames 12 `
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 `
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 `
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 `
  --bg-mode mixed `
  --steps 200000 --out ./ckpt_tmpl --best-metric top1 `
  --ckpt-every 500 --summary-every 2000 --snapshot-every 20000 `
  --logevery 100 --workers 4 --prefetch 2 --tag run1
```

先跑一次 3 步自检确认形状/显存/耗时：

```powershell
python ml/train.py --quick --bs 4 --T 10 --workers 0 --amp --tag smoke
```

### 3.2 背景档（背景是影响真实录音表现的主因，建议做成对照实验）

| `--bg-mode` | 背景内容 | 适用 |
|---|---|---|
| `mixed` | 80% 白/粉噪 + 20% 数字静音 | 基线 |
| `recording` | 真实录音片段（`ml/bg16.npy`） | 目标域与真实录音接近时 |
| `long` | 长音床（环境/剧情）+ 0.65 概率叠 1–3 条战斗语音 | 素材里有长音、需要覆盖"背景一直在响" |

三档只改背景、其余参数保持一致，才能把差异归因到背景。

> 电平口径：`mixed/recording` 的 `--bg-lo/--bg-hi` 是**峰值** dBFS；`long` 档是 **RMS** dBFS。
> 同一个数值在 `long` 档下响得多（音乐比噪声底密），不要跨档直接比较数值。

### 3.3 显存与吞吐

| 配置 | 显存峰值 | 说明 |
|---|---|---|
| T=10 / bs=16 | ≈ 1.3 GB | 8 GB 卡的安全档 |
| T=10 / bs=32 | ≈ 2.5 GB，可能 `cudaErrorUnknown` | 超过 8 GB 卡能力 |
| T=4 / bs=32 | ≈ 1.1 GB | 需要小窗口时用 |

`--workers` × `--prefetch` 不是越大越好：合成单窗口约 9.3 ms(热)/35.3 ms(冷)，单 worker 可产
约 108 窗/s，而训练只需约 10.5 窗/s。在途 batch 过多会占用数 GB 内存，把 mmap 的模板库
页缓存挤出去，合成就退化为冷读。推荐 `--workers 4 --prefetch 2`。

### 3.4 日志怎么读

```
              步/总数        | 已用/剩余 | lo 检测BCE  pk onset-margin  off 边界BCE  le 识别CE  F1 帧级 | top1 top5 | it/s
[####......] 100/200000 0.1% | 已用 12s   | lo 1.203  pk 2.416  off 0.958  le 8.852  F1 0.239 | top1 0.0% top5 0.0% | 1.7 it/s
```

- `lo` / `off`：onset / offset 的 BCE，应持续下降。
- `pk`：onset 必须高出后续持续段的 hinge，开启 `--peak-w` 后出现。
- `le`：模板识别的交叉熵，随机猜测基线是 `ln(K)`（K=6576 时约 8.79）。它降到基线以下才说明在学。
- `F1`：帧级检测 F1（训练批内），只作趋势参考 —— 它**不反映"每个音效在哪里"**。
- `top1 / top5`：识别准确率（相对模板库全库）。

`--eval` 相关：脚本按 `--summary-every` 对最近若干步取滑动平均来判定 `best`，
避免用单批噪声决定最佳点。

### 3.5 checkpoint 与续训

| 文件 | 语义 |
|---|---|
| `ckpt_<tag>.pt` | 每 `--ckpt-every` 步**覆盖**写，含优化器状态 |
| `best_<tag>.pt` | 按 `--best-metric` 的滑动平均另存，不覆盖 |
| `snap_<tag>_s<N>.pt` | 每 `--snapshot-every` 步另存一份，不覆盖 |

```powershell
# 续训（--steps 必须大于 checkpoint 里的 step）
python ml/train.py --resume latest --steps 400000 ... --tag run1

# 扩大模型后继续训练：必须显式 --resize，否则按 checkpoint 记录的 d/emb/layers/nh 续跑
python ml/train.py --resume ckpt_tmpl/best_run1.pt --resize --layers 8 ... 
```

`best` 记录带可比性签名（metric / K / tag）：签名变化时旧记录会自动改名留档、`best` 从 -1 重新计，
以免不同口径的分数互相覆盖。

---

## 4. 评测

### 4.1 分档评测

```powershell
python ml/eval.py --ckpt ckpt_tmpl/best_run1.pt --T 0 --n 1500
```

`--T 0` 表示**用 checkpoint 里记录的窗口长度**（不要手填，窗口不一致会让数字失去意义）。

输出四档（easy/mid/hard/dense）+ 干净锚点，并附一行 **PCEN 消融**（`--no-pcen`）作为对照。

### 4.2 事件级定位（验收口径）

```powershell
python ml/loc_eval.py --ckpt ckpt_tmpl/best_run1.pt --T 0 --windows 900
```

- 扫 (阈值 × 不应期 × 容差)，报告**事件级** P / R / F1；阈值按事件 F1 选，不按帧 F1。
- 同时给出每个不应期下的**召回上限**：两两间距 ≥ (refr+1) 帧的最大真值子集。
  `R` 贴近上限说明瓶颈在解码器（不应期/池化），`R` 远低于上限才是模型的问题。
- 附假峰分析：假峰落在事件区间内的比例，以及与"随机时刻"基线的对比。

### 4.3 识别上限

```powershell
python ml/recall_at_k.py --ckpt ckpt_tmpl/best_run1.pt --T 0 --windows 900
```

两条曲线：**oracle**（用真值事件段池化后去全库排序 = 识别天花板）与 **detected**
（用模型自己的 onset 峰 + 固定池化窗 = 推理实际拿到的）。两者的差就是检测误差 + 池化选择的代价。

### 4.4 域差诊断

```powershell
python ml/onset_diag.py --ckpt ckpt_tmpl/best_run1.pt --wav <真实录音.wav> --windows 16
```

比较同一模型在四类输入上的 onset 概率分布：合成窗口 / 真实录音 / 孤立模板(无背景) / 数字静音与噪声底，
并把合成音效叠在真实背景上逐档抬电平。判读：孤立模板与静音上安静、只有真实录音上饱和 ⇒ 是**背景域差**，
应换训练背景（§3.2），而不是调电平、加容量或调解码器。

---

## 5. 推理

```powershell
python ml/infer.py --ckpt ckpt_tmpl/best_run1.pt --wav <录音.wav> --refr 1 --pool-frames 12 --k 8
```

输出 `timeline_<录音名>_<tag>_s<step>.json`（每帧的事件时间、候选 class 与增益）与同名 `.txt`。

它会打印 `onset 概率 p50/p90/p99 -> 阈值`：若阈值贴到上限（`--thr-hi`，默认 0.90），说明概率分布已饱和，
这份时间轴不可信，先按 §4.4 定位原因。

`--refr`（不应期）与 `--pool-frames`（事件识别池化窗）必须与训练时一致（默认 12 = 240 ms）。

---

## 6. 参数取值依据与注意事项（问题 → 方案）

**前端**
1. **采样率**：语音类素材常见为 16 kHz，8 kHz 以上没有能量，任何匹配/反相在该频段都无法工作。
   若素材有更高采样率的原始版本，优先使用；改用其它采样率需要同时调整 `core.py` 的
   `SR/N_FFT/HOP/N_MELS` 与 `build_bank.py` 的 `SR`。
2. **PCEN 不可关**：把前端换成纯 log-mel 会让识别准确率大幅下降（`eval.py` 的消融行会直接给出差距）。
   `--no-pcen` 只用于消融对照，正式训练保持开启。
3. **逐 clip 标准化**（PCEN 之后的 `(x-mean)/std`）是前端的一部分：它让同一模板在不同录音电平下
   的表示保持一致，去掉后域差会放大。

**数据合成**
4. **模板库长度上限 2.5 s**：更长素材会被过滤掉。BGM / 环境音 / 长语音应走 `long/` 作为**背景**，
   而不是当模板（当模板也无法逐事件定位）。
5. **空窗必须存在**：训练窗口若总有事件，模型没有"静音"概念，实测表现是精确率低、过检多。
   用 `--empty-frac 0.2`（默认）保留 20% 纯背景窗口。
6. **onset 目标用脉冲**：整段目标下连续重复的音效无法逐个分辨，用
   `--onset-shape gauss --onset-len 5 --onset-sigma 2.5`（≈50 ms 脉冲）。
   注意：目标变尖会让**帧级 F1 下降**（真值变窄），这是正常的，不要据此回退。
7. **持续音内部的假峰**：onset 头只在事件起始处被监督，会在持续音中间产生假峰。
   解法是 `--arch v2` 的 offset 头 + `--peak-w 1.0 --peak-win 25 --peak-margin 2.0`
   （要求 onset 高出其后 500 ms 内的最大值 2.0）。
8. **背景内容决定真实录音上的表现**：只用白/粉噪与数字静音训练，模型遇到持续有 BGM/环境音/人声的
   真实录音会把大量帧判成 onset。用 `--bg-mode recording` 或 `long`（§3.2）。
9. **源模型**：同一个源 = 同一个音效按随机间隔重复（模拟连续触发），同一源共用同一变体以保持音色一致；
   源的活跃区间可以越出窗口（中途插入/退出），onset 落在窗口外的事件只标 span、不给脉冲 ——
   这样模型才会见过"声音已经在响"的输入。
10. **自重复与近邻**：短音效（<0.6 s）在同窗内重复、以及同 bundle 的近邻音效混入，都能提高
    对真实场景的覆盖；难度用 `--dense-frac / --min-src / --max-src / --src-rate-ref` 控制。
11. **混音标定**：逐实例增益 + 源内抖动 + 整体缩放，峰值超限时用 **tanh 软限幅**。
    不要用硬削波：硬削波是非线性，会让本可完美抵消的信号也无法对齐。
12. **`--src-rate-ref`**：源数按 `max(1, T/该值)` 缩放，使**事件密度**不随窗口长度变化；
    想更密就调小它。

**增广**
13. **逐条质检**：增广的前提是"变换后仍属于同一类"。同一组参数对不同模板效果差异很大
    （能量集中的冲击音可能被带通整段滤掉，宽带噪声型几乎不受影响），所以 `gen_variants.py`
    对每条变体用零均值归一化相关（NCC，`aug_spec.py` 的 `QC_NCC`，默认 0.60）与干净模板比对：
    不达标换 SAFE 规格重做；两条都不达标则回退到干净模板本身。这样"入库变体的 NCC 恒 ≥ 门限"
    成为不变量，识别头不会收到与自身标签负相关的样本。
14. **变速样本不进变体库**：NCC 是固定时间对齐的判据，对任何变速都会立刻失效，
    因此变速无法通过质检；真实素材的变速域（约 0.90–1.10 倍速）由训练时的在线 `chan_fx`
    覆盖（±0.6% 轻微变速 + EQ + 饱和）。需要更宽的范围时应在 `chan_fx` 里调。
15. **有最小长度要求的变换**：`LoudnessNormalization`（pyloudnorm）需要 ≥ 400 ms。
    `aug_spec.MIN_SAMPLES` 会按长度剔除它；短模板若让它抛异常再由调用方兜住，
    那条变体会退化成"原样复制"而其 NCC=1.0 被判为合格，等于悄悄失去增广。

**训练循环**
16. **识别头的池化窗**：`--pool-frames`（默认 12 = 240 ms）在训练与推理两侧必须一致。
17. **损失权重硬相加**：`loss = lo + le_w·le + off_w·off + peak_w·pk`。减小 `--le-w` 会让识别项
    占比下降，过大则压制检测；识别项与检测项共享同一主干，两者需要通过权重平衡。
18. **`--bnd-loss focal` 的量级**：focal 形式的边界损失比 `pos_weight BCE` 小约 15 倍，
    切换时必须同步调大 `--ool-w`，否则相当于关掉边界监督。
19. **类别数很大时用集合目标**（可选）：`--tir-conf` 让"与真实标签波形相似度 ≥ 阈值的其它模板"
    也算正确，降低近重复模板带来的标签噪声；0 = 关闭（普通 CE）。
20. **数据顺序**：块打乱（`--shuffle-block`）保证每次重启看到的数据顺序不同，
    否则 `SynthDS` 的样本是 `(seed, index)` 的确定函数，重启后 step 与数据完全绑定。
21. **`--steps` 必须大于 checkpoint 的 step**，否则没有可执行的步。

**运行环境**
22. **显存**：T=10/bs=16 约 1.3 GB；T=10/bs=32 在 8 GB 卡上会 `cudaErrorUnknown`。
    `--T × --bs > 200` 时脚本会打印显存警告。
23. **DataLoader 不宜过大**：见 §3.3。
24. **长音池写入中断**：`build_long.py` 先按完整长度建 memmap 再逐条填，中断会让尾部保持为 0。
    `synth.py` 会扫描真实内容边界并把越界/全零条目判为无效（并在日志里报告可用条数）；
    要得到完整池就重跑 `build_long.py`。
25. **日志**：训练写文本日志 `ml/ckpt_tmpl/train_<tag>.log`，并同时留在终端。

**指标**
26. **不要用帧级 F1 当验收**：它无法反映事件的**位置与数量**。验收用 `loc_eval.py` 的事件级
    P/R/F1，并对照它给出的召回上限。
27. **阈值按事件 F1 选**，不按帧 F1。
28. **识别能力用 `recall_at_k.py` 的 oracle 曲线单独衡量**，它不受检测误差影响。
29. **背景虚警要在纯背景上量**：合成噪声底 + 真实长音两档都要看，只看一档会低估虚警。

---

## 7. 常见失败模式速查

| 现象 | 最可能的原因 | 处理 |
|---|---|---|
| `le` 长期停在 `ln(K)` 附近、`top1` 为 0 | 识别头还没开始学（K 较大时前期正常）；或候选/池化窗配置不一致 | 继续观察；确认 `--pool-frames` 两侧一致；用 `recall_at_k.py` 的 oracle 曲线确认上限 |
| 检测 F1 尚可但 `top1` 很低 | 训练集里混入了与标签不再对应的增广样本 | 检查 `gen_variants.py` 的 NCC 统计；不要关闭质检 |
| 真实录音上事件数量异常多、阈值贴到上限 | 训练背景与真实录音的**内容**差异过大 | 换 `--bg-mode recording` 或 `long`（§3.2），用 `onset_diag.py` 验证 |
| 大量假峰落在持续音内部 | 只有 onset 监督，没有边界监督 | 用 `--arch v2` + `--peak-w/--peak-win/--peak-margin` |
| 过检多、精确率低 | 训练窗口全是"有事件"，模型没有静音概念 | `--empty-frac 0.2` |
| 窗口里 95% 是空的 | 源的起始时刻没有铺满窗口 | 检查 `--max-src / --src-rate-ref`；确认 `mixer` 的铺开逻辑未被关闭 |
| 显存溢出 | `--T × --bs` 过大 | 降 `--bs` 或 `--T`（§3.3） |
| 吞吐随训练下降 | DataLoader 在途 batch 过多，挤掉了 mmap 页缓存 | `--workers 4 --prefetch 2` |
| 续训后模型形状变了 | 命令行形状与 checkpoint 不同且未加 `--resize` | 加 `--resize`，或按 checkpoint 的形状继续 |
