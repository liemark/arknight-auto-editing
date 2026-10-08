"""打印两种编码器的逐层结构与参数量（K=22326, d=192, emb=256, nl=4, nh=4）。"""
import os
import sys

import torch

TRAIN = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import core                                                       # noqa: E402

K, d, emb, nl, nh = 22326, 192, 256, 4, 4


def show(tag, m):
    enc = sum(p.numel() for p in m.enc.parameters())
    heads = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("enc.") and n != "proto")
    proto = m.proto.numel()
    tot = sum(p.numel() for p in m.parameters())
    print("=" * 78)
    print("%s   编码器 %.2fM + 头 %.2fM + 原型 %.2fM = %.2fM" % (tag, enc / 1e6, heads / 1e6, proto / 1e6, tot / 1e6))
    print("=" * 78)
    x = torch.randn(1, core.N_MELS, 2001)
    for name, mod in m.enc.named_children():
        if name == "blocks":
            for i, b in enumerate(mod):
                y = None
                print("  blocks[%d]  AttnBlock(d=%d, nh=%d, RoPE)   %8d 参数" % (i, d, nh, sum(p.numel() for p in b.parameters())))
            continue
        if name == "convs":
            for i, c in enumerate(mod):
                print("  convs[%d]   Conv1d(%d,%d,k=3,dil=%d)      %8d 参数" % (i, d, d, c.dilation[0], sum(p.numel() for p in c.parameters())))
            continue
        print("  %-10s %-46s %8d 参数" % (name, str(mod).split("(")[0], sum(p.numel() for p in mod.parameters())))
    print("  头: onset Linear(%d,1)=%d; head Linear(%d,%d)=%d; OffsetHead Linear(%d,1)=%d"
          % (d, d + 1, d, emb, d * emb + emb, d, d + 1))
    print("  原型 proto: (%d, %d) = %d" % (K, emb, proto))


show("原架构  --enc attn", core.Detector(K, d=d, emb=emb, nl=nl, nh=nh, rope=True, enc="attn"))
show("新 CRNN   --enc crnn", core.Detector(K, d=d, emb=emb, nl=nl, nh=nh, rope=True, enc="crnn"))
