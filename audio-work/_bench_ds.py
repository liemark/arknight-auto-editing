"""量合成数据生成吞吐（CPU，多进程 worker 就是干这个的）。

目的：判断 44.1k 训练变慢是卡在 GPU 还是卡在数据管线。
训练时 it/s 需要的数据率 = it/s * bs 个窗口/秒。
"""
import os
import sys
import time

import numpy as np

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import synth                                                      # noqa: E402
import core                                                       # noqa: E402

def bench(T, once_len, n=24, fx=True):
    ds = synth.SynthDS(T=T, length=n + 4, seed=3, max_ev=192, dense_frac=0.6,
                       silence_frac=0.2, empty_frac=0.2, min_src=1, max_src=5,
                       ivl_med=1.0, ivl_sig=1.2, once_len=once_len,
                       label_mode="template", onset_shape="gauss", onset_len=5,
                       onset_sigma=2.5, bg_mode="mixed", left_pad=0.0, p_mid=0.3,
                       src_rate_ref=7.5, no_fx=not fx)
    for i in range(4):                       # 预热(打开 memmap / 首次分配)
        ds[i]
    t0 = time.time()
    ev = 0
    for i in range(4, 4 + n):
        o = ds[i]
        ev += int((o[3] >= 0).sum())
    el = time.time() - t0
    return n / el, ev / n, o[0].shape[0] / core.SR

print("单进程生成吞吐（worker 数=4 时约乘 4；采样率 %d）" % core.SR)
print("%-34s %10s %10s %10s" % ("配置", "窗口/秒", "事件/窗", "秒音频/窗"))
for T, ol, fx, tag in ((20.0, 1.0, True,  "T=20 once_len=1.0 fx=on（当前训练口径）"),
                       (20.0, 0.0, True,  "T=20 once_len=0   fx=on（旧口径）"),
                       (20.0, 1.0, False, "T=20 once_len=1.0 fx=off"),
                       (10.0, 1.0, True,  "T=10 once_len=1.0 fx=on")):
    wps, ev, sec = bench(T, ol, fx=fx)
    print("%-34s %10.1f %10.0f %10.1f" % (tag, wps, ev, sec))
print()
print("训练实测 it/s = 4.0~6.1（bs=8, T=20）-> 需要 32~49 个窗口/秒")
print("若单进程只有 %.1f 窗口/秒, 4 个 worker = %.1f 窗口/秒 -> 数据管线就是瓶颈"
      % (bench(20.0, 1.0)[0], bench(20.0, 1.0)[0] * 4))
