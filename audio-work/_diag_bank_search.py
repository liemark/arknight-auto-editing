"""诊断：录音里那个音效, 在我们的模板库里【到底有没有】对应的变体?

动机：波形 NCC 0.08 / 包络 0.5-0.65 / 频谱 0.5-0.6 —— 像"时间对、声音像、但不是同一条"。
游戏常给同一音效准备多个变体随机播; 我们的库也有变体, 但检测头只从 top-k 里挑。
所以: 用【频谱余弦】在全库 K=22326 条上粗筛 top-N, 再逐条做波形对齐(NCC)。
  * 若全库最好也只有 ~0.2  -> 这套录音的音效不在库里(版本/素材源不同) => 换录音或换素材
  * 若全库里有 0.8+ 的     -> 库里其实有正确变体, 是识别头挑错了兄弟 => 可修
纯 CPU。
"""
import json
import os
import sys

import numpy as np

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
TRAIN = os.path.join(AI, "train")
sys.path.insert(0, AI)
sys.path.insert(0, os.path.join(AI, "data", "atoms"))
from audio_inverse.atomlib import AtomLib                                   # noqa: E402
from audio_inverse.audio import read_wav                                    # noqa: E402
from audio_inverse.config import load_cfg                                    # noqa: E402
from audio_inverse.postproc.render import _map_templates                     # noqa: E402

SR = 44100
NF, HOP = 1024, 441
NFR = 48                       # 比较前 48 帧(~0.48s): 覆盖起音, 又不必整条模板
TOPN = 30
NEV = 3


def logspec(x):
    """前 NFR 帧的对数幅度谱, 归一化向量。"""
    n = NF + HOP * (NFR - 1)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(NF)[None, :] + HOP * np.arange(NFR)[:, None]
    X = np.abs(np.fft.rfft(x[idx] * np.hanning(NF), axis=1)) + 1e-6
    L = np.log(X).ravel()
    L -= L.mean()
    return (L / (np.linalg.norm(L) + 1e-12)).astype(np.float32)


def wave_ncc(x, seg, c, srch=SR // 33):
    x = x - x.mean(); nx = np.linalg.norm(x)
    if nx < 1e-9:
        return 0.0, 0
    best = (0.0, 0)
    for lag in range(-srch, srch + 1, 4):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        w = seg[a:a + len(x)]; w = w - w.mean()
        r = float(x @ w) / (nx * np.linalg.norm(w) + 1e-12)
        if abs(r) > abs(best[0]):
            best = (r, lag)
    f = best
    for lag in range(best[1] - 4, best[1] + 5):
        a = c + lag
        if a < 0 or a + len(x) > len(seg):
            continue
        w = seg[a:a + len(x)]; w = w - w.mean()
        r = float(x @ w) / (nx * np.linalg.norm(w) + 1e-12)
        if abs(r) > abs(f[0]):
            f = (r, lag)
    return f


def resample44(path):
    import alab
    x, sr = alab.wav_read(path, mono=True)
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if int(sr) != SR:
        x = alab.bandpass(x, int(sr), 30.0, min(int(sr), SR) * 0.475)
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x)
    return x.astype(np.float32)


cfg = load_cfg("base.yaml", [])
lib = AtomLib(os.path.join(cfg.abspath("data_root"), "atomlib"))
mp = _map_templates(cfg)
lens = np.load(os.path.join(TRAIN, "bank_lens.npy"))
offs = np.load(os.path.join(TRAIN, "bank_offs.npy"))
bank = np.load(os.path.join(TRAIN, "bank_pool.npy"), mmap_mode="r")
bi = json.load(open(os.path.join(TRAIN, "bank_index.json"), encoding="utf-8"))
K = len(lens)
NEED = NF + HOP * (NFR - 1)

for tag, wavname in (("nl_mono", "nl_mono"), ("v82", "v82_mono")):
    tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_%s_v3d.json" % tag),
                        encoding="utf-8"))
    x = resample44(os.path.join(AI, "data", "atoms", wavname + ".wav"))
    evs = sorted([e for e in tl if e["cands"][0]["cat"] != "其他"],
                 key=lambda e: -e["cands"][0]["gain"])[:NEV]
    print("=" * 100)
    print("%s：在全库 K=%d 上做频谱粗筛 (top %d) + 波形对齐" % (wavname, K, TOPN))
    print("=" * 100)
    for e in evs:
        t = e["t"]; c0 = e["cands"][0]
        c = int(t * SR)
        seg = x[max(0, c - 2000): c + NEED + 2000].astype(np.float64)
        cc = c - max(0, c - 2000)
        if len(seg) < NEED + 100 or cc < 0:
            continue
        q = logspec(seg[cc:cc + NEED])
        scores = np.empty(K, np.float32)
        for i in range(0, K, 1024):
            blk = []
            for j in range(i, min(i + 1024, K)):
                L = int(min(lens[j], NEED))
                w = np.asarray(bank[int(offs[j]):int(offs[j]) + L], np.float32)
                blk.append(logspec(w))
            B = np.stack(blk)
            scores[i:i + len(blk)] = B @ q
        top = np.argsort(-scores)[:TOPN]
        best = (0.0, 0, -1)
        for j in top:
            L = int(min(lens[j], NEED))
            w0 = np.asarray(bank[int(offs[j]):int(offs[j]) + L], np.float64)
            r, lag = wave_ncc(w0, seg, cc)
            if abs(r) > abs(best[0]):
                best = (r, lag, j)
        j = best[2]
        print("  t=%7.2fs  检测头给的: %-22s(cos %.2f)   全库最好: %-22s 波形NCC %.3f (谱 %.3f, 延迟 %.1fms)"
              % (t, c0["template"][:22], c0["cos"],
                 (bi[j]["name"][:22] if j >= 0 else "-"), best[0],
                 scores[j] if j >= 0 else 0, best[1] / SR * 1000))
