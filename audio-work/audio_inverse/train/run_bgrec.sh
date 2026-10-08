#!/usr/bin/env bash
# ============================================================================
# 实验: 背景域修正  (4 层 / T=10 / mean 池化 / bg-mode: mixed -> recording)
#
# 【为什么跑这一跑】2026-09-16 端到端实测把真正的瓶颈钉死了 —— 不是模型容量。
#
# 症状: infer.py 在真实录音 nl_mono.wav 上, 4 层和 8 层模型的自适应阈值都被 clamp 到
#   上限 0.900, 分别吐出 1023 / 1127 个事件, NNLS 增益普遍 0.00~0.11。
#   09-13 的旧 timeline 同症状 (779 / 1286 个) -> 端到端从来就没工作过。
#
# 诊断 (ml/onset_diag.py, 同一个模型喂四组输入, 看 onset 概率分布):
#   A 合成窗口(训练同分布)   p50=0.001   >0.5 的帧 10.1%
#   B 真实录音               p50=0.448   >0.5 的帧 45.8%     << 饱和
#   C 孤立模板(无背景)       p50=0.011   >0.5 的帧 10.9%
#   D 数字静音 / 噪声底      p50=0.001   >0.5 的帧  0.0%     << 完全干净
#   D 那两行是决定性的: 模型在静音和噪声上【一点都不响】, 它不是"无脑乱响",
#   它只是在真实录音上把 45.8% 的帧判成了 onset。这是纯 domain gap。
#
# 根因坐实: 训练全程 --bg-mode mixed, 而 mixed 只产生【数字静音(30%) + 白噪/粉噪(70%)】
#   (synth.py 第 290-292 行)。--bg-mode recording (拿真实录音 bg16.npy 当背景) 这个开关
#   【存在但从来没被用过】。真实游戏录音里 BGM + 环境音占绝对主导, 模型一次都没见过。
#
# 便宜的验证 (onset_diag.py 的 E 组, 1 分钟, 不用训练):
#   合成 SFX + 真实录音背景, 逐档抬电平:
#     bg 默认(-70..-40 dBFS)  p50=0.227   >0.5 的帧 29.9%
#     bg -45 dBFS             p50=0.347   >0.5 的帧 36.7%
#     bg -32 dBFS             p50=0.418   >0.5 的帧 41.5%
#     bg -26 dBFS             p50=0.437   >0.5 的帧 43.5%
#     真实录音(参照)          p50=0.448   >0.5 的帧 45.8%
#   -> 只把背景从白噪换成真实录音(电平都不用动), p50 就从 0.001 跳到 0.227;
#      背景抬到真实电平后, 模拟出的 0.418 和真实录音的 0.448 几乎重合。
#
# 电平不用调 (所以 --bg-db 不加, 保住单变量):
#   真实录音逐帧 RMS = p5 -78.8 / p25 -54.9 / p50 -37.8 / p90 -26.4 / p99 -22.6 dBFS,
#   而训练背景档 -70..-40 dBFS 的均值正好 -55 —— 对 p25=-54.9 只差 0.1 dB, 本来就对得上。
#   问题在背景的【内容】, 不在【电平】。
#
# 【单变量】相对 4 层基线 best_tmpl6576_s6000_T10.0_v2_d192e256l4.pt 只改 bg-mode:
#   - resume 源 base4L_s6000_T10.pt 就是那个基线 (step 6000, oracle top-1 58.10%)。
#   - 【不加 --resize】: 形状完全一致, 让 Adam 动量完整继承 (119/119 张量)。
#     加了 --resize 反而会按 shape 筛选暖启动、把动量清零, 平白多一个变量。
#   - label_mode 自动从 checkpoint 继承 (template), 不用手写 —— train.py 第 331-335 行。
#   - 其余参数逐个核对过, 与基线 ckpt 里的 args 完全一致 (max_ev 128 / ivl_med 1.0 /
#     ivl_sig 1.2 / min_src 1 / max_src 5 / src_span_lo 0.25 / p_mid 0.3 /
#     dense_frac 0.6 / silence_frac 0.2 / empty_frac 0.2 / le_w 0.5 / peak_* 同值)。
#   - 唯一已知的次要差异: 基线当时是恒定 lr (--lr-sched 那时还不存在), 这里用 cosine+warmup500,
#     即本项目长跑的既定配置。主要终点是真实录音的 onset p50 (与调度无关), 合成得分只作护栏。
#
# 【判据】不看合成得分, 看两个数:
#   1) onset_diag.py 里真实录音的 p50 有没有从 0.448 明显降下来  <- 主要终点
#   2) recall_at_k.py 的 oracle top-1 有没有从 58.10% 崩掉        <- 护栏
#   快照每 10000 步一个, 事后按真实指标挑, 不要信训练内的 top1。
#
# 【命名隔离】--tag bgr -> best_bgr.pt / ckpt_bgr.pt / snap_bgr_s*.pt / train_bgr.log。
#   现有 best_tmpl6576.pt(定位最好 F1 0.8737) 等一概不会被碰 —— 归档逻辑只动 best_<tag>.pt。
#
# 用法:  bash ml/run_bgrec.sh
# 崩溃后续跑: SRC=ckpt_tmpl/ckpt_bgr.pt bash ml/run_bgrec.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-}
if [ -z "$PY" ]; then
  for cand in "E:/Python313/python.exe" "/e/Python313/python.exe"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  [ -n "$PY" ] || PY=python
fi

"$PY" -c "import sys, torch; print('[bgrec] 解释器 %s' % sys.executable); print('[bgrec] torch %s | cuda_available %s' % (torch.__version__, torch.cuda.is_available())); sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "[bgrec] 中止: 这个解释器用不了 CUDA. 换 PY=... 或先装 CUDA 版 torch." >&2
  exit 3
}

# 背景源检查: bg_mode=recording 靠 ml/bg16.npy (v82_mono.wav 降采样到 16k, 323.3s)
if [ ! -f bg16.npy ]; then
  echo "[bgrec] 中止: 没有 ml/bg16.npy, --bg-mode recording 会静默退化。" >&2
  exit 4
fi
echo "[bgrec] 背景源 ml/bg16.npy 就位 ($(stat -c %s bg16.npy 2>/dev/null || echo '?') 字节)"

SRC=${SRC:-ckpt_tmpl/base4L_s6000_T10.pt}
if [ ! -f "$SRC" ]; then
  echo "[bgrec] 中止: resume 源不存在: $SRC" >&2
  exit 5
fi
echo "[bgrec] resume 源: $SRC  (4 层基线 step6000, oracle top-1 58.10%)"

exec "$PY" train.py \
  --resume "$SRC" \
  --arch v2 \
  --T 10 --bs 16 --amp \
  --lr 8e-4 --lr-sched cosine --warmup 500 \
  --pool-mode mean --pool-frames 12 \
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 \
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 \
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 \
  --bg-mode recording \
  --steps 120000 \
  --tag bgr \
  --out ./ckpt_tmpl --best-metric top1 \
  --ckpt-every 500 --summary-every 2000 --snapshot-every 10000 \
  --logevery 100 --workers 4 --prefetch 2
