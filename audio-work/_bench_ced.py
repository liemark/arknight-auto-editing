"""C 选项：预训练骨干 CED（冻结）当特征，量 recall@k。

CED 的调用链（官方）：波形 16kHz --CedFeatureExtractor--> mel (B,64,T) --CedModel.encoder--> 隐状态。
我们素材是 44.1k，所以两边都重采样到 16k。
口径与 _bench_frontends.py 一致：dense 档、oracle onset、全库 K=22326、窗长 0.6s。
对照：PCEN patch 余弦 R@1 11.4% / R@50 26.1%；训练好的 v3d oracle R@1 91.9%。
"""
import os
import sys
import time

import numpy as np
import torch
import torchaudio

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402
import synth                                                      # noqa: E402

dev = torch.device("cuda")
SR, SRB = core.SR, 16000
WIN_S, NWIN, BATCH = 0.6, 60, 128
KS = [1, 5, 20, 50]
REPO = "mispeech/ced-mini"

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification   # noqa: E402
fe = AutoFeatureExtractor.from_pretrained(REPO, trust_remote_code=True)
m = AutoModelForAudioClassification.from_pretrained(REPO, trust_remote_code=True).to(dev).eval()
for p in m.parameters():
    p.requires_grad_(False)
print("CED %s  参数 %.2fM" % (REPO, sum(p.numel() for p in m.parameters()) / 1e6))

# 分类模型的 forward 不返回隐藏状态, 也不收 output_hidden_states -> 用 hook 抓 encoder 末尾的 norm
_store = {}
m.encoder.norm.register_forward_hook(lambda mod, i, o: _store.__setitem__("h", o))


def feat(x44: torch.Tensor, dim_cache={}):
    """(B, N@44.1k) -> (B, D) 归一化特征。"""
    x16 = torchaudio.functional.resample(x44, SR, SRB)
    inp = fe(list(x16.cpu().numpy()), sampling_rate=SRB, return_tensors="pt")["input_values"].to(dev)
    with torch.no_grad():
        m(inp)
    h = _store["h"]
    if dim_cache.get("t") is None:
        dim_cache["t"] = h.shape[1]
        print("  mel %s -> 隐状态 %s (每 token %.0f ms)" % (tuple(inp.shape), tuple(h.shape),
                                                      WIN_S * 1000 / h.shape[1]))
    v = h.mean(dim=1)
    return (v / (v.norm(dim=-1, keepdim=True) + 1e-9)).cpu().numpy()


lens = np.load("bank_lens.npy"); offs = np.load("bank_offs.npy")
bank = np.load("bank_pool.npy", mmap_mode="r")
K = len(lens)
NS = int(WIN_S * SR)
print("画廊 %d 条 ..." % K)
G = None; t0 = time.time()
for i0 in range(0, K, BATCH):
    xs = []
    for j in range(i0, min(i0 + BATCH, K)):
        L = int(min(int(lens[j]), NS))
        x = np.asarray(bank[int(offs[j]):int(offs[j]) + L], np.float32)
        if len(x) < NS:
            x = np.pad(x, (0, NS - len(x)))
        xs.append(x)
    f = feat(torch.from_numpy(np.stack(xs)).to(dev))
    if G is None:
        G = np.zeros((K, f.shape[1]), np.float32)
    G[i0:i0 + len(xs)] = f
    if (i0 // BATCH) % 20 == 0:
        print("  %5d/%d  %.0fs" % (i0 + len(xs), K, time.time() - t0), flush=True)

ds = synth.SynthDS(T=20.0, length=NWIN, seed=4242 + 404, dense_frac=0.6,
                   min_ev=8, max_ev=12, lo=-40.0, hi=-28.0, bg_lo=-45.0, bg_hi=-45.0,
                   empty_frac=0.0, silence_frac=0.2, min_src=1, max_src=5,
                   ivl_med=1.0, ivl_sig=1.2, once_len=1.0, label_mode="template",
                   onset_shape="gauss", onset_len=5, onset_sigma=2.5, bg_mode="mixed")
Q, lab = [], []
for i in range(NWIN):
    mix, po, ev, lb = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
    v = lb >= 0
    if not v.any():
        continue
    f0 = ev[v][:, 0]; cl = lb[v]
    for j in range(len(cl)):
        a = int(f0[j]) * core.HOP
        if a < 0 or a + NS > len(mix):
            continue
        Q.append(mix[a:a + NS]); lab.append(int(cl[j]))
lab = np.array(lab)
print("查询事件 %d 个" % len(lab))
Qf = np.zeros((len(Q), G.shape[1]), np.float32)
for i0 in range(0, len(Q), BATCH):
    Qf[i0:i0 + BATCH] = feat(torch.from_numpy(np.stack(Q[i0:i0 + BATCH])).to(dev))
Qf /= (np.linalg.norm(Qf, axis=1, keepdims=True) + 1e-12)
order = np.argsort(-(Qf @ G.T), axis=1)[:, :max(KS)]
print()
print("%-34s %8s %8s %8s %8s" % ("方法（冻结，不训练）", "R@1", "R@5", "R@20", "R@50"))
row = [float((order[:, :k] == lab[:, None]).any(1).mean()) for k in KS]
print("%-34s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("CED-mini 冻结特征", *[100 * r for r in row]))
print("%-34s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("（对照）PCEN patch 余弦", 11.4, 16.9, 22.9, 26.1))
print("%-34s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("（对照）训练好的 v3d", 91.9, 95.8, 97.6, 98.1))
