# _audio_lab/ml — 音效检测库

纯合成训练的 SFX 定位+识别。数据是**现场合成**的（不落盘），模板来自游戏解包。

## 主链路

| 文件 | 干什么 |
|---|---|
| `core.py` | **模型定义**。`FrontEnd`(STFT→64 mel→PCEN) + `Enc`(卷积 stem + 注意力栈) + 双头(`onset` / `proto` 余弦) + `OffsetHead`(边界头) + RoPE |
| `synth.py` | **合成数据生成器** `SynthDS`。源=同一音效按随机间隔重复，可中途插入/退出、可跨窗口边界被切 |
| `build_long.py` | **长音池**构建：`../long/index_long.json` → `long_pool.npy`（BGM/环境/干员战斗语音，16 kHz） |
| `train.py` | 训练主程序（argparse 在 main() 里，DataLoader worker 不会重复解析） |
| `eval.py` | 分档评测 easy/mid/hard/dense + 干净锚点 + 去 PCEN + 纯背景虚警 |
| `loc_eval.py` | **验收指标**：事件级定位 P/R/F1（帧级 F1 测不出"每个音效在哪里"） |
| `infer.py` | 全片推理 → 时间轴 JSON/TXT（onset 峰 → 池化 → 原型 top-k → NNLS 重拟合） |
| `recall_at_k.py` | recall@k 曲线（top 多少能到 99%） |
| `test_losses.py` | `set_ce` / `onset_margin` 不变量单测（纯 CPU，秒级） |

## 数据资产（都已被 .gitignore 排除）

| 文件 | 大小 | 谁生成 | 谁用 |
|---|---|---|---|
| `bank_clean.npy` + `bank_lens/offs.npy` + `bank_index.json` | 502 MB | `build_bank.py`（从 `../sfx/`） | 训练/推理的模板库 |
| `bank_96k.npy` / `bank_48k.npy` | 各 502 MB | `build_bank.py`（AAC 降质缓存） | 只有 `gen_variants.py` |
| `variants.npy` + `variants_lens/ncc.npy` | 2.4 GB | `gen_variants.py` | 训练默认取它（音色一致的增广） |
| `warm6576_src.pt` | 15.7 MB | `warmstart_template.py` | 任务 0 / 3 的暖启动 |
| `tir_6576.npz` | 1.2 MB | `build_confusion.py` | `--tir-conf` 集合目标 |
| `fp6576.npz` | 34 MB | `fp_build.py` | `hash_*.py` 指纹路线 |
| `long_pool.npy` + `long_offs/lens.npy` + `long_meta.json` | 2.7 GB | `build_long.py` | `SynthDS(bg_mode="long")` 的长音背景 |
| `ckpt_tmpl/` | 运行中 | 训练 | **训练在写，别删** |

## 长音测试集（BGM / 环境 / 干员战斗语音）

模板库 `build_bank.py` 把 clip 硬卡在 **2.5 s**（`LMAX=40000`），所以**所有长音都被静默滤掉了**
—— 干员语音（0.5~25 s）、环境音（≤20 s）、BGM（1~3 min）一个都没进过训练和测试。
而真实录音里这几样**一直在响**。补齐链路：

```
_audio_lab/x_extract_long.py all       # 解包 → _audio_lab/long/（voice_battle_jp/cn + music）
_audio_lab/ml/build_long.py            # → ml/long_pool.npy + long_meta.json
```

| 来源 | 内容 | 量 |
|---|---|---|
| `PersistentData/.../audio/sound_beta_2/voice{, _cn}/char_*.ab` | **所有干员的战斗语音**：charword `voiceIndex` 17–32（编入队伍/任命队长/行动出发/行动开始/选中干员1-2/部署1-2/作战中1-4/4星·3星·非3星结束/行动失败） | 488 人 × 16 条 × 2 语种 |
| `StreamingAssets/.../audio/sound_beta_2/music/**` | BGM（`m_sys_` 界面循环 / `m_bat_` 战斗 / `m_avg_` 剧情），16 kHz 单声道，每条截前 60 s | 565 条 |
| 已有 `sfx/` 里 `bundle=="ambience"/"dialog"` | 环境循环音 / 剧情音 | 140 条 |
| 已有 `sfx/` 里 `dur > 2.5 s` 的其余 clip | 长音效（`long_se`） | 视索引 |

> 索引 `../long/index_long.json` 的 `voiceTitle` 来自社区解包表 `_data/charword_table.json`；
> `voiceIndex → 标题` 在 483 个干员上**完全一致**，所以选择规则可以直接用序号（皮肤 AB 没有表项也能选）。

**怎么用**：`SynthDS(bg_mode="long")` 取 1~2 条长音床叠加，再以 `long_voice_p`(默认 0.65) 概率
叠 1~3 条干员战斗语音。`eval.py` 里对应 `long` 档（事件密度/电平与 `mid` 完全相同，**只换背景**，
所以 `long − mid` 就是"背景换成录音里的长音"的代价），末尾还有一条**纯长音背景虚警率**。

**电平约定（别直接横比）**：`long` 档的 `bg_lo/bg_hi` 是 **RMS dBFS**，噪声档是**峰值 dBFS**。
同一个 −50，音乐在 RMS 意义下比噪声底密得多（`long_probe.py` 会打出两者的实测差）。

## 诊断脚本（按需跑，都不在训练路径上）

**架构/数值**：`arch_report.py`(架构+参数量+感受野+置换等变性) · `rope_check.py`(RoPE 是否生效 / 旧 checkpoint 能否载入) · `domain_probe.py`(时域还是频域) · `frontend_range.py`(前端在不同窗口长度下的数值上限) · `step_cost.py`(单步显存/耗时)

**数据分布**：`long_probe.py`(长音池构成 + 两种背景模式的实际电平/削波率) · `edge_probe.py`(事件/秒、跨边界比例) · `budget_diag.py`(噪声/重叠预算) · `grid_ceiling.py`(帧网格理论上限) · `aug_ablate.py`+`aug_spec.py`(哪个增广破坏了模板身份)

**指标拆解**：`pool_sweep.py`(池化窗扫描) · `timing_probe.py`(时间精度) · `depth_probe.py`(逐层探针) · `headtohead.py` · `modelfree_match.py`(无模型对照) · `rerank_probe.py`(波形重排上限)

**指纹路线**：`fp.py` · `fp_build.py` · `fp_run.py` · `hash_rank.py` · `hash_ab.py` · `hash_ablate.py` · `hash_diag.py` · `hash_peaks.py`

**其它**：`prof_speed.py`/`prof2.py`(性能剖析) · `render_timeline.py`+`verify_export.py`+`synth_listen.py`(听得见的产物) · `x_export.py` / `x_extract_long.py`(上级) · `x_extract_sfx.py`(抽 SFX, 在上上级 `_audio_lab/`)

## 已停用的历史路线

- **簇模式**（512 类）：`make_clusters.py` + `clusters_512.npy/json`。现在是 6576 全库 template 模式，不再分堆。`headtohead.py` 仍会用它做对照。
- **指纹**（谱图峰值对）：做过，结论是净负收益，脚本留着备查。

## 约定

- 训练/推理的窗口长度必须一致：`eval.py`/`loc_eval.py` 的 `--T 0` = 自动跟随 checkpoint 记的值。
- 架构以 checkpoint 的 `args.arch`/`args.rope` 为准（`core.ckpt_rope` 兜底），别用代码默认值去前向旧 checkpoint。
- `best_*.pt` 记录只在 **同样的 T 和架构** 之间可比；不一致会被自动留档改名并从 -1 重新计。
- 别删 `ckpt_tmpl/`（训练在写）、`variants.npy`、`bank_clean.npy`。
