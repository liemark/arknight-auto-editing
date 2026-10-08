"""诊断：合成数据里【事件重叠度】随 once_len(长素材只播一次) 的变化。纯 CPU，不碰 GPU。

背景：44.1k 换库后 le(识别CE) 卡在 ln K 不动、而 lo/F1 正常降。测出来是因为
模板时长中位从 0.94s(SFX) 变成 2.09s(含战斗语音)，而"源"仍按中位 1.0s 间隔重复 ->
每个源自己叠自己，240ms 池化窗 71% 混着别的类别 -> 类别不是输入的函数。
这里对比三种配置的重叠度。
"""
import os
import sys

import numpy as np

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import synth                                                      # noqa: E402
import core                                                       # noqa: E402

PF = 12                     # --pool-frames 12 = 240 ms
N = 8
BASE = dict(T=20.0, length=N, seed=1, max_ev=192, dense_frac=0.6, silence_frac=0.2,
            empty_frac=0.2, min_src=1, max_src=5, ivl_med=1.0, ivl_sig=1.2,
            label_mode="template", onset_shape="gauss", onset_len=5, onset_sigma=2.5,
            bg_mode="mixed", left_pad=0.0, p_mid=0.3, src_rate_ref=7.5, no_fx=True)
CONFIGS = [
    ("once_len=0   （旧行为：全部按 ivl 重复）", dict(once_len=0.0)),
    ("once_len=1.0 （新默认，src_rate_ref=7.5）", dict(once_len=1.0)),
    ("once_len=1.0 + src_rate_ref=12（略降源数）", dict(once_len=1.0, src_rate_ref=12.0)),
    ("once_len=1.0 + src_rate_ref=20（源数=1~5）", dict(once_len=1.0, src_rate_ref=20.0)),
    ("once_len=1.0 + src_rate_ref=30（源数=1~3.3）", dict(once_len=1.0, src_rate_ref=30.0)),
]
lens = np.load("bank_lens.npy")

print("=" * 92)
print("T=20s, dense 60%%, 1~5 源 x T/src_rate_ref, ivl 中位 1.0s, K=22326, 池化窗 240ms")
print("=" * 92)
print("%-42s %6s %6s %7s %8s %8s %8s" % ("配置", "事件/窗", "类别/窗", "同时响", "污染率", "别的类", ">1s占比"))
for name, kw in CONFIGS:
    ds = synth.SynthDS(**{**BASE, **kw})
    ev_cnt, cl_cnt, span_c, contam, oth_avg = [], [], [], [], []
    dur = []
    for i in range(N):
        mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
        v = lb >= 0
        ne = int(v.sum())
        if ne == 0:
            continue
        Tm2 = len(po) // 2
        f0 = ev[v][:, 0]; f1 = ev[v][:, 1]; cl = lb[v].astype(np.int64)
        st = np.maximum(f0, 0) // 2
        sf = st.copy()
        ef = np.minimum((np.maximum(f1, 0) + 1) // 2, Tm2)
        en = np.maximum(np.minimum(ef, st + PF), st + 1)
        dur += list(lens[cl] / core.SR)
        acc = np.zeros(Tm2 + 1, np.int32)
        np.add.at(acc, sf, 1); np.add.at(acc, np.minimum(ef, Tm2), -1)
        span_c.append(np.cumsum(acc)[:Tm2].mean())
        oth = []
        for j in range(ne):
            m = np.ones(ne, bool); m[j] = False
            hit = (st[m] < en[j]) & (st[j] < en[m])
            oth.append(len(np.unique(cl[m][hit])))
        oth = np.array(oth)
        ev_cnt.append(ne); cl_cnt.append(len(np.unique(cl)))
        contam.append(float((oth > 0).mean())); oth_avg.append(oth.mean())
    dur = np.array(dur)
    print("%-42s %6.0f %6.0f %7.2f %7.1f%% %8.2f %7.0f%%" % (
        name, np.median(ev_cnt), np.median(cl_cnt), np.mean(span_c),
        100 * np.mean(contam), np.mean(oth_avg), 100 * (dur > 1.0).mean()))
print("=" * 92)
print("列义：同时响=每一瞬间平均几个原子在响；污染率=池化窗里混着别的类别的事件占比；别的类=平均几个")
