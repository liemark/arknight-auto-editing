"""44.1 kHz 变长流模板库 + SynthDS 的开训前自检（不训练，纯 CPU，秒级）。

检查三件事：
  1. 池与索引自洽：offs 严格递增、offs+lens 不越界、长度>0、索引条数 == K、SR 一致;
  2. 标签栅格与窗口一致：po 的帧数必须 == 1 + n // HOP（HOP 与 core 一致），
     否则标签与模型输出会静默错位;
  3. 前端前向形状：FrontEnd(20 s) 的输出帧数 == 期望的模型输出帧数。
"""
import json
import os
import sys

import numpy as np
import torch

T = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, T)
os.chdir(T)
import core                                                     # noqa: E402
import synth                                                    # noqa: E402

fail = []


def check(cond, msg):
    print("  %s %s" % ("OK  " if cond else "FAIL", msg))
    if not cond:
        fail.append(msg)


print("=" * 72)
print("1) 模板池与索引")
lens = np.load("bank_lens.npy")
offs = np.load("bank_offs.npy")
pool = np.load("bank_pool.npy", mmap_mode="r")
meta = json.load(open("bank_meta.json", encoding="utf-8"))
index = json.load(open("bank_index.json", encoding="utf-8"))
K = len(lens)
print("  K=%d  池 %.2f GB / %.1f 分钟  最长模板 %.2f s  源采样率分布 %s"
      % (K, pool.shape[0] * 2 / 1e9, pool.shape[0] / core.SR / 60.0,
         lens.max() / core.SR, meta.get("src_sr_hist")))
check(core.SR == int(meta["sr"]) == 44100, "SR 一致（core / bank_meta = 44100）")
check(K == len(index), "索引条数 == K（%d）" % K)
check(int((lens > 0).sum()) == K, "所有模板长度 > 0")
check(bool((np.diff(offs.astype(np.int64)) > 0).all()), "offs 严格递增")
check(bool((offs.astype(np.int64) + lens.astype(np.int64) <= pool.shape[0]).all()),
      "offs + lens 不越界")
check(int(meta.get("n_err", 0)) == 0, "建库时无读取错误（n_err=0）")
qs = np.percentile(lens / core.SR, [50, 90, 100])
print("  时长分布 p50 %.2f s  p90 %.2f s  max %.2f s" % tuple(qs))
long_frac = float((lens / core.SR > 2.5).mean())
print("  >2.5 s 的模板占比 %.1f%%（变长流下不再被截断）" % (100 * long_frac))
for w in ("bank_clean.npy", "bank_96k.npy", "bank_48k.npy"):
    check(not os.path.exists(w), "旧的定长矩阵已不在活动路径（%s）" % w)

print("\n2) SynthDS 标签栅格（T=20 s）")
ds = synth.SynthDS(T=20.0, length=64, seed=7, max_ev=192, dense_frac=0.6,
                   silence_frac=0.2, empty_frac=0.2, min_src=1, max_src=5,
                   label_mode="template", onset_shape="gauss", onset_len=5,
                   onset_sigma=2.5, bg_mode="mixed")
n = ds.n
nmel = 1 + n // synth.HOP
Tm = nmel // core.TIME_STRIDE
print("  窗口 %d 采样（%.1f s） -> 谱帧 %d -> 模型输出帧 %d"
      % (n, n / core.SR, nmel, Tm))
bad = 0
for i in range(8):
    _o = ds[i]
    mix, po, ev, lb = _o[0], _o[1], _o[2], _o[3]
    nv = int((lb >= 0).sum())
    f0r = (int(ev[lb >= 0][:, 0].min()), int(ev[lb >= 0][:, 0].max())) if nv else ("-", "-")
    same = po.shape[0] == nmel
    bad += 0 if same else 1
    print("  #%d mix %s peak %.3f | po %d/%d %s | 事件 %2d onset样本 %3d | f0 %s | len %d"
          % (i, mix.shape, float(np.abs(mix).max()), po.shape[0], nmel,
             "OK" if same else "错位", nv, int((po > 0).sum()), f0r, len(mix)))
check(bad == 0, "8 条样本的标签帧数都与窗口一致")
check(all(len(ds[i][0]) == n for i in range(3)), "混音长度 == 窗口长度")
check(all(ds[i][1].ndim == 1 for i in range(3)), "po 是一维帧级标签")

print("\n3) 前端前向")
fe = core.FrontEnd().eval()
with torch.no_grad():
    mel = fe(torch.from_numpy(ds[0][0])[None])
check(mel.shape[1] == core.N_MELS, "mel 维数 == N_MELS（%d）" % core.N_MELS)
check(mel.shape[2] // core.TIME_STRIDE == Tm,
      "FrontEnd 输出帧数//stride == 期望模型帧数（%d）" % Tm)
print("  FrontEnd(20 s) -> %s" % (tuple(mel.shape),))

print("\n" + "=" * 72)
if fail:
    print("不通过 %d 项：" % len(fail))
    for f in fail:
        print("  × %s" % f)
    sys.exit(1)
print("全部通过 —— 可以开训")
