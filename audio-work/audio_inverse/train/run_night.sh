#!/usr/bin/env bash
# ============================================================================
# 过夜长跑脚本  (v2 / T=10 / 8 层 + mean 池化 —— 单变量实验: 只加深度)
#
# 背景 (2026-09-15 深夜的实测, 结论记在这里免得再绕回去):
#   白天的推断: "nl=4 的末块还有 +23 点增益, 远未饱和 -> 该加深度".
#   第一次跑 8 层时【同时】改了 3 个变量 (layers 4->8, pool_mode mean->attn, Adam 动量清零),
#   实测是倒退的 —— 同一口径 (recall_at_k.py 全库排序 oracle top-1, T=10) 下:
#       4 层 step 6000  -> 58.10%
#       8 层 step 18000 -> 47.91%      低 10.19 点
#   但三重变量混杂 -> 无法归因. 所以本次把池化切回基线验证过的 mean,
#   【只保留 "加深度" 这一个变量】, 让结论可解释.
#
#   评测口径提醒: 旧的 54.97% 是 recall_at_k.py 硬编码 T=4.0 测出来的, 和 T=10 训练的模型
#   不是一回事. 该脚本已修为自动跟随 checkpoint 的 T 和 label_mode (与 loc_eval.py 一致).
#   T=10 口径下的正确基线就是上面的 58.10%.
#
#   中间成果都没丢: best_tmpl6576_s22000_T10.0_v2_d192e256l8.pt (8 层, 训练内 top1 0.7436)、
#   snap_tmpl6576_s20000.pt、以及 4 层基线 best_tmpl6576_s6000_T10.0_v2_d192e256l4.pt
#   和它的副本 base4L_s6000_T10.pt, 都在 ckpt_tmpl/ 里.
#
# 为什么这么配:
#   1) --resize 必加:  否则续跑会【静默】把命令行里的 layers 改回 checkpoint 里的值
#      (旧日志那行 "[resume] layers: 8 -> 4" 就是它), 以前"想扩大模型"根本没生效.
#   2) resume 源 base4L_s6000_T10.pt: step 6000 的 v2/T=10 4 层权重, 训练内记录 top1 0.6654,
#      比 T=4/v1 那条线跑到 88000 步的 0.6241 还高. 单独复制一份, 免得被本次运行覆盖.
#   3) --snapshot-every:  ckpt-every 是【覆盖同一个文件】, 所以以前每次从 warm 重开
#      都会把长跑的 checkpoint 冲掉. 留快照是最重要的保险.
#   4) cosine + warmup:  恒定 lr 在十万步量级会在最优点附近来回跳, 汇总曲线一直抖
#      (60.1~62.9% 来回跳), best 很难刷新. 调度按【绝对 step】算, 中断续跑曲线连续.
#      从 step 6000 续跑时余弦因子还有 0.998, 相当于从接近峰值的 lr 开始衰减.
#   5) --pool-mode mean:  这是本次实验的关键控制变量. 4 层基线用的就是 mean, 而第一次 8 层
#      那一跑换成了 attn (attn 是 mean 的超集, 但要额外训一套池化器参数), 于是"加深度"的
#      效果和"换池化"的效果混在一起分不开. 这次固定成 mean, 让深度成为唯一变量.
#   6) workers=4 prefetch=2 (不是 12x4):  实测单窗口合成只要 9.3ms 热 / 35.3ms 冷,
#      单 worker 就能产 108 窗/s, 而训练最多只要 ~10.5 窗/s —— 12 个 worker 是 120 倍
#      冗余. 12x4 = 48 个 batch 在途, 白占 ~3.5GB 内存, 把 mmap 的 bank 页缓存挤出去,
#      合成退化成冷读, 吞吐从 8.4 it/s 一路衰减到 5.5 (空闲内存 764MB -> 568MB).
#      降到 4x2 后 16GB 机器内存松出来, 页缓存能常驻, 速度反而稳.
#
# 用法:  bash ml/run_night.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 解释器: 优先用 E:\Python313 —— 那是装了 CUDA 版 torch 的那一份.
# C 盘系统 Python 里是 torch 2.6.0+cpu, 拿它跑只会在 .to(dev) 处失败.
# 想换回别的解释器: PY=/path/to/python.exe bash ml/run_night.sh
PY=${PY:-}
if [ -z "$PY" ]; then
  for cand in "E:/Python313/python.exe" "/e/Python313/python.exe"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  [ -n "$PY" ] || PY=python
fi

# 启动前先验 CUDA, 免得把数据集建完才发现跑不了
"$PY" -c "import sys, torch; print('[night] 解释器 %s' % sys.executable); print('[night] torch %s | cuda_available %s' % (torch.__version__, torch.cuda.is_available())); sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "[night] 中止: 这个解释器用不了 CUDA. 换 PY=... 或先装 CUDA 版 torch." >&2
  exit 3
}

# 把 resume 源固定成一份不可变副本, 防止本次运行把它覆盖掉.
# 崩溃后重启时改成指向最新 checkpoint 更划算:
#   SRC=ckpt_tmpl/ckpt_tmpl6576.pt bash ml/run_night.sh
SRC=${SRC:-}
if [ -z "$SRC" ]; then
  SRC=ckpt_tmpl/base4L_s6000_T10.pt
  if [ ! -f "$SRC" ]; then
    cp ckpt_tmpl/best_tmpl6576_s6000_T10.0_v2_d192e256l4.pt "$SRC"
    echo "[night] 已备份 resume 源 -> $SRC"
  fi
fi
echo "[night] resume 源: $SRC"

exec "$PY" train.py \
  --resume "$SRC" \
  --resize --layers 8 --nh 4 --d 192 --emb 256 \
  --arch v2 \
  --T 10 --bs 16 --amp \
  --lr 8e-4 --lr-sched cosine --warmup 500 \
  --pool-mode mean --pool-frames 12 \
  --onset-shape gauss --onset-len 5 --onset-sigma 2.5 \
  --peak-w 1.0 --peak-win 25 --peak-margin 2.0 \
  --bnd-loss bce --pos-weight 10 --off-w 1.0 --le-w 0.5 \
  --bg-mode mixed \
  --steps 200000 \
  --out ./ckpt_tmpl --best-metric top1 \
  --ckpt-every 500 --summary-every 2000 --snapshot-every 20000 \
  --logevery 100 --workers 4 --prefetch 2
