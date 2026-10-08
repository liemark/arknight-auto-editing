# 最小端到端框架（audio_inverse.pipeline）

一个入口，八步全串起来。**薄驱动**：不重新实现任何东西，只用固定好的默认值调用已有脚本，
所以每一步都能单独跑、单独 debug。

```powershell
cd F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse
$py = "F:\杂七杂八\arknight-auto-editing\.venv\Scripts\python.exe"

& $py -m audio_inverse.pipeline status                    # ① 资产状态：现在能不能开跑
& $py -m audio_inverse.pipeline assets                    # ② 补 bg44.npy + long44/index_long.json
& $py -m audio_inverse.pipeline bank                      # ③ 建模板库（≈2.5 分钟）
& $py -m audio_inverse.pipeline longpool                  # ④ 建长音池（≈4 分钟）
& $py -m audio_inverse.pipeline train --tag 44k2          # ⑤ 训练（默认 6G 显存预算）
& $py -m audio_inverse.pipeline eval                      # ⑥ 分档评测（自动跟随 checkpoint）
& $py -m audio_inverse.pipeline infer --wav data/atoms/v82_mono.wav   # ⑦ 推理 → timeline
& $py -m audio_inverse.pipeline export --timeline <json> --wav <wav>  # ⑧ 试听轨
& $py -m audio_inverse.pipeline render --timeline <json> --wav <wav>  # ⑨ 反相轨 + PSR
```

## 显存预算（6G）

attn 编码器、T=20 s、bf16、前向+反传，本机 4060(8GB) 实测：

| bs | 峰值显存 |
|---|---|
| 8 | 1.85 GB |
| 12 | 2.74 GB |
| 16 | **3.62 GB** ← 框架默认 |
| 20 | 4.50 GB |

`--bs` 可覆盖（`train --bs 12`）；crnn / cnn / bcresnet 编码器比 attn 更省。
**注意瓶颈通常不是显存而是数据管线**：单进程生成约 12 个 20 s 窗口/秒，所以 `--workers 10 --prefetch 4`
（默认）比加 bs 更能提吞吐。

## 编码器开关（同一个 `--enc`，其它参数全一样 → 单变量对照）

| `--enc` | 结构 | 编码器参数 | bs=8 显存 / 前反向 | 12k 步时的质量 |
|---|---|---|---|---|
| `attn` | 4 层自注意力 + RoPE（原架构） | 2.68M | 2.90 GB / 179 ms | **最好**：le 6.88、top1 15.0% |
| `crnn` | 卷积 stem + 扩张残差卷积 + 2 层 BiGRU | 1.57M | 2.65 GB / 142 ms | 次之：le 7.43、top1 10.1% |
| `cnn` | 同上但无循环 | 1.24M | 2.63 GB / 142 ms | 未跑长 |
| `bcresnet` | 2D 卷积 ⊗ 时间向 1D 卷积广播相加 | 0.56M | 1.84 GB / 100 ms | 未跑长（最省） |

结论：**T=20 s 下注意力在质量上最划算**；线性复杂度（crnn/bcresnet）只有在把窗口拉到 40~60 s 时才回本。

## 已经验证过的经典基线（同样口径：dense 档、oracle onset、全库 K=22326）

| 方法 | R@1 | 备注 |
|---|---|---|
| 训练好的 v3d（实例级 22k 类） | **91.9%** | 当前最强 |
| PCEN patch 余弦（不训练） | 11.4~26.1% | 前端本身有信息量 |
| 谱特征 + 随机森林（11 类粗分类） | 52.2% 分类准确率 | 类别这一层就被模型超过 |
| CED-mini 冻结（预训练打标签骨干） | 4.1% | 打标签的语义特征对"哪一条变体"无用 |
| MFCC patch 余弦 | 0.3% | 倒谱截断丢掉区分变体的细节 |

**未解决的关键问题**：`render` 出来的 `residual.wav` 目前 **PSR≈0.1 dB（几乎没抵消）**。
缺的不是分类器，而是波形域的两件事：
1. **亚帧延迟**：输出时间在 20 ms 栅格上，反相抵消要求亚采样对齐（相关峰抛物线插值 / 互谱相位斜率 / 分数延迟滤波器）。
2. **短 FIR 信道估计**（正则化 LS / NLMS），吸收重采样、EQ、编码、混响差异；现在整条链只有一个标量增益。

另外 `audio-work/_bench_fp.py`（音频指纹）**实现有 bug**（合成真值 8 条只认回 2 条），
修好之前不能用它下任何结论；指纹是唯一还没被证伪的经典路线。

## 目录速查

| 位置 | 内容 |
|---|---|
| `audio_inverse/pipeline.py` | 本框架（薄驱动） |
| `audio_inverse/postproc/{assets,export,render}.py` | 资产状态 / 试听轨 / 反相渲染 |
| `train/{core,synth,train,eval,infer,loc_eval,recall_at_k}.py` | 模型与训练/评测 |
| `train/build_{bank,long}.py` | 模板库 / 长音池 |
| `docs/PIPELINE.md` | 更详细的后处理与历史说明 |
| `.vscode/tasks.json` | `AI · 00`~`AI · 18` 全套任务（含 4 种编码器） |
