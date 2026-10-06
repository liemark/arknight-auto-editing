"""冒烟：三种编码器（attn / crnn / cnn）的形状、参数量、显存、前反向耗时。"""
import os
import sys
import time

import torch

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402

dev = torch.device("cuda")
print("T=20s -> 谱帧 %d, 输出帧 %d" % (1 + 20 * core.SR // core.HOP,
                                  (1 + 20 * core.SR // core.HOP) // core.TIME_STRIDE))
for enc in ("attn", "crnn", "cnn", "bcresnet"):
    for bs in (2, 8):
        m = core.Detector(1000, d=192, emb=256, nl=4, nh=4, rope=True, enc=enc).to(dev)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(bs, core.N_MELS, 2001, device=dev)
        for p in m.parameters():
            p.grad = None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            on, emb = m(x)
            (on.mean() + emb.mean()).backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3 * 1000
        print("  %-5s bs=%d  参数 %.2fM  峰值显存 %.2f GB  前+反 %.0f ms   输出 onset %s emb %s"
              % (enc, bs, n / 1e6, torch.cuda.max_memory_allocated() / 1e9, dt,
                 tuple(on.shape), tuple(emb.shape)))
        del m, x
        torch.cuda.empty_cache()
