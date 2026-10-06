# _audio_lab 源码包

- 打包时间：2026-09-17
- 文件数：73
- 未压缩体积：4062.7 KB

## 包含范围

- 根目录 `*.py`：`alab.py`（音频 IO/渲染工具库）、`x_extract_sfx.py`、`x_extract_long.py`、`x_export.py`（数据导出）
- `ml/*.py`：训练 / 合成 / 评测 / 推理全部脚本
- `ml/*.sh`：长跑启动脚本（`run_night.sh`、`run_bgrec.sh`）
- `ml/*.md`：工程说明 `README.md` 与各 checkpoint 的 `eval_report_*.md`
- `ml/*.json`：数据索引与评测结果（`bank_index.json`、`clusters_512.json`、`eval_results*.json`、`variants_meta.json`、`long_meta.json`）

## 未包含（体积原因，非源码）

| 类别 | 路径 | 说明 |
|---|---|---|
| 模型权重 | `ml/ckpt_tmpl/*.pt`、`ml/ckpt/` | 约 1.4 GB，含 base4L_s6000_T10 等关键存档 |
| 特征池 | `ml/*.npy`、`ml/*.npz` | `bank_clean.npy` / `variants.npy` / `long_pool.npy` / `fp6576.npz` 等，约 6 GB |
| 音频素材 | `sfx/`、`long/`、`*.wav` | 8240 + 14088 个 wav |
| 推理产物 | `timeline_*.json`、`out/*.wav`、`out/montage_*`、`out/listen/` | timeline / 试听包 |
| 训练日志 | `ml/ckpt_tmpl/train_*.log` | 可复现记录，体积较大 |
| 参考文献 | `papers/` | arXiv 论文源（2506.21086 等），非本项目代码 |
| 会话记忆 | `.workbuddy-ai/` | 项目约定笔记 |

## 重建所需数据（本包不含）

运行前需先准备：`sfx/`（SFX 库）、`long/`（长音池）、`ml/bank_clean.npy` 等特征池、
以及 `ml/ckpt_tmpl/` 下的 checkpoint。生成顺序见 `ml/README.md`。

## 文件清单

| 文件 | 字节 | sha256[:12] |
|---|---:|---|
| `alab.py` | 6821 | `5c8abeffe6e9` |
| `x_export.py` | 3730 | `998ccf5e71f1` |
| `x_extract_long.py` | 9193 | `ebd62b3d9475` |
| `x_extract_sfx.py` | 3374 | `115f293c23c8` |
| `ml/README.md` | 6273 | `611efbb7116f` |
| `ml/arch_report.py` | 6352 | `8a9c452f1b28` |
| `ml/aug_ablate.py` | 3011 | `26c61d82dd21` |
| `ml/aug_spec.py` | 4410 | `1392c7a20bd7` |
| `ml/bank_index.json` | 578678 | `d14f527be2c7` |
| `ml/bench_synth.py` | 4463 | `0fd2750736af` |
| `ml/budget_diag.py` | 3166 | `e147cc80a3c1` |
| `ml/build_bank.py` | 3662 | `849ed4893e74` |
| `ml/build_confusion.py` | 7567 | `7c14f7b3af74` |
| `ml/build_long.py` | 2719 | `d74242c0ca0c` |
| `ml/clusters_512.json` | 1814 | `fcb0538c5378` |
| `ml/core.py` | 9257 | `4e8d5f167cff` |
| `ml/depth_probe.py` | 6170 | `e8d0ddb17e73` |
| `ml/domain_probe.py` | 5300 | `9735fc2aaeae` |
| `ml/e2e_verify.py` | 12350 | `e2be68dc3383` |
| `ml/edge_probe.py` | 2964 | `4e01a44bed8e` |
| `ml/eval.py` | 11941 | `f1dcd30b3704` |
| `ml/eval_report.md` | 1982 | `ece1106452aa` |
| `ml/eval_report_tmpl6576_s48000.md` | 971 | `d4ad83e10671` |
| `ml/eval_report_tmpl6576_s6000.md` | 986 | `a06157d803f1` |
| `ml/eval_report_tmpl6576_s88000.md` | 998 | `f1b253ceed43` |
| `ml/eval_report_tmpl6576_s93000.md` | 970 | `499bd7d39875` |
| `ml/eval_results.json` | 2503 | `5ae125c3e9e9` |
| `ml/eval_results_tmpl6576_s48000.json` | 2574 | `f2b52a6ddd95` |
| `ml/eval_results_tmpl6576_s6000.json` | 2579 | `3b84931b3935` |
| `ml/eval_results_tmpl6576_s88000.json` | 2580 | `3bd342aaedc8` |
| `ml/eval_results_tmpl6576_s93000.json` | 2531 | `f5856a5bc3a5` |
| `ml/fp.py` | 7301 | `67bb29cb2b51` |
| `ml/fp_build.py` | 879 | `32cdc0d868bb` |
| `ml/fp_run.py` | 5626 | `c42c7b547b36` |
| `ml/frontend_range.py` | 1904 | `7ccf15f4f3a0` |
| `ml/gen_variants.py` | 4205 | `1393cd84da0e` |
| `ml/grid_ceiling.py` | 1544 | `0ca1bd8b83fb` |
| `ml/hash_ab.py` | 1961 | `5ea80e71ee0d` |
| `ml/hash_ablate.py` | 4851 | `d321d9a51e49` |
| `ml/hash_diag.py` | 3715 | `860185000e56` |
| `ml/hash_peaks.py` | 5362 | `b27bdaf73949` |
| `ml/hash_rank.py` | 6486 | `bfe5c9b58233` |
| `ml/headtohead.py` | 4149 | `628b13ff5f4a` |
| `ml/infer.py` | 10527 | `758eba38b240` |
| `ml/launch_detached.py` | 2048 | `2e88cf58e2ce` |
| `ml/loc_eval.py` | 16770 | `f102647519aa` |
| `ml/long_meta.json` | 3193136 | `a67733e189ec` |
| `ml/long_probe.py` | 3254 | `7927cb0d85ae` |
| `ml/long_voice_titles.md` | 764 | `24b642fb94f7` |
| `ml/make_clusters.py` | 3664 | `17350ab71c86` |
| `ml/modelfree_match.py` | 6704 | `14302e8d42a8` |
| `ml/onset_diag.py` | 6925 | `6e2a7bb085fa` |
| `ml/peel_probe.py` | 5841 | `0993a241c06e` |
| `ml/peel_probe2.py` | 5429 | `ff0d54654bd4` |
| `ml/pool_sweep.py` | 4153 | `be0626a3f313` |
| `ml/pres_probe.py` | 3537 | `28901f2f70b7` |
| `ml/prof2.py` | 1172 | `d1013a4d3bca` |
| `ml/prof_speed.py` | 4900 | `7edfad3e9eb3` |
| `ml/recall_at_k.py` | 6793 | `abada8b4f140` |
| `ml/render_timeline.py` | 3836 | `1af829668788` |
| `ml/rerank_probe.py` | 5658 | `9106bf875acf` |
| `ml/rope_check.py` | 2102 | `30277da4d4aa` |
| `ml/run_bgrec.sh` | 5806 | `5467fa4e8206` |
| `ml/run_night.sh` | 5149 | `cd7f248cd0b3` |
| `ml/step_cost.py` | 6917 | `f7b198f8fddc` |
| `ml/synth.py` | 24460 | `098eb5263467` |
| `ml/synth_listen.py` | 4445 | `5cf750444635` |
| `ml/test_losses.py` | 5740 | `3d423578eeed` |
| `ml/timing_probe.py` | 5521 | `bb07f0fa8e31` |
| `ml/train.py` | 51225 | `59f00337fed1` |
| `ml/variants_meta.json` | 96 | `1854b8cf2511` |
| `ml/verify_export.py` | 2247 | `85c285ee67af` |
| `ml/warmstart_template.py` | 1487 | `801b35ef822b` |
