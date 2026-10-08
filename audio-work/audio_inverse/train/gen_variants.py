
"""Pre-generate V augmented variants per template using audiomentations.

Identity QC (added after measuring ml/aug_ablate.py): the raw aggressive Compose left ~73% of
variants uncorrelated with the template they are labelled as (median NCC 0.06), which poisons
the classification/prototype loss -- detection (onset) still trained fine but top-1 stalled at
2-3%.  Each variant is now checked with zero-mean NCC against its clean source; if it falls
below QC_NCC the "safe" Compose is used instead.  Measured after the fix: p50 NCC 0.95,
99.5% > 0.5, 94% > 0.8.
"""
import os, sys, json, time, argparse
import numpy as np
import multiprocessing as mp
ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.dirname(ML))
from aug_spec import MIN_SAMPLES, build, ncc as _ncc, QC_NCC

LMAXV = 88200          # 每条变体最多保留 2 s
# 短于这个长度的模板要换用"过滤版"规格：有几个变换有最小长度要求，硬跑会抛异常，
# 而异常兜底会把变体变成"原样复制"(NCC=1.0) 却被当成合格 —— 见 aug_spec.MIN_SAMPLES 的注释。
MINLEN = max(MIN_SAMPLES.values()) if MIN_SAMPLES else 0
_B = {}


def _init(banks, lens, profile):
    _B["offs"] = np.load(os.path.join(ML, "bank_offs.npy"))
    _B["pool"] = np.load(banks, mmap_mode="r")
    _B["lens"] = lens
    # 长/短各备一套：短的那套自动剔除装不下的变换
    _B["aug"], sk1 = build(profile)
    _B["aug_short"], sk2 = build(profile, n_samples=MINLEN - 1)
    _B["safe"], sk3 = build("safe")
    _B["safe_short"], sk4 = build("safe", n_samples=MINLEN - 1)
    if sk1 or sk2 or sk3 or sk4:
        print("[aug] 过滤/跳过: %s" % "; ".join(sk1 + sk2 + sk3 + sk4), flush=True)


def _work(job):
    k, v, seed = job
    rng = np.random.default_rng(seed)
    L0 = int(_B["lens"][k])
    off = int(_B["offs"][k])
    _pool = _B["pool"]
    L = min(L0, LMAXV)
    w = np.asarray(_pool[off:off + L], dtype=np.float32).copy()
    if rng.random() < 0.30 and L > 1600:          # SFX cut off mid-way
        L = int(rng.uniform(0.35, 0.95) * L)
        w = w[:L].copy()
    L = len(w)
    short = L < MINLEN
    aug = _B["aug_short"] if short else _B["aug"]
    safe = _B["safe_short"] if short else _B["safe"]
    fb = 0                                        # 1 = 走了异常兜底（原样复制），要报出来
    try:
        y = np.asarray(aug(w.copy(), sample_rate=44100), dtype=np.float32)
    except Exception:
        y = w.copy()
        fb = 1
    c = _ncc(y, w)
    rej = 0                                       # 1 = 两条候选都不达标 -> 回退干净模板
    if c < QC_NCC:
        try:
            y2 = np.asarray(safe(w.copy(), sample_rate=44100), dtype=np.float32)
        except Exception:
            y2 = w.copy()
            fb = 1
        if _ncc(y2, w) > c:
            y, c = y2, _ncc(y2, w)
        # ★ 硬兜底：full 与 safe 都不达标时，回退到干净模板本身并把 NCC 记 1.0。
        #   原来这里【没有】这一层 —— 只有"两条里挑更好的"，所以实测仍有 9.7% 的变体
        #   带着 NCC < 0.60（最差 -0.11，即与自己的标签负相关）进库，等于往分类/原型损失里
        #   灌标签噪声。宁可这条变体等于没增广，也不能放噪声进去。
        if c < QC_NCC:
            y, c, rej = w.copy(), 1.0, 1
    # 这里原来有一段"故意 2 倍速"（rng.random() < 0.08，施加在质检【之后】）：
    #     y = np.interp(np.arange(len(y)/2)*2, np.arange(len(y)), y)
    # 已移除。实测（10 条模板取中位，判据用模型前端 PCEN 谱的逐帧余弦）：
    #     不变速 1.000 | soxr 0.90 -> 0.723 | 1.10 -> 0.720 | 1.25 -> 0.653 | 2.00 -> 0.350
    # 也就是 2 倍速已经把"这个波形还是不是它自己的标签"打到 0.35，属于标签噪声；
    # 而且它是【无抗混叠的线性插值抽取】，会引入不真实的混叠伪影；注释写的
    # "unchanged pitch" 也与实际不符（数学上就是 2 倍速播放）。
    # 真实素材的变速域是 0.90~1.10（PCEN 0.72~0.78，可接受），那一段由训练时的在线
    # chan_fx（±0.6%）覆盖；要更宽应在 chan_fx 里调，而不是在这里塞 8% 的伪影样本。
    # 注意：波形 NCC 对任何变速都会立刻归零（连 0.90 倍速都是 0.001），所以 NCC
    # 不能用来给变速样本做质检 —— 这也是原来把它放在质检之后的原因。
    tw = 0                                        # 统计口径保留（现在恒为 0）
    if len(y) > LMAXV:
        y = y[:LMAXV]
    n = len(y)
    out = np.zeros(LMAXV, dtype=np.float16)
    out[:n] = y.astype(np.float16)
    return k, v, out, n, c, fb, rej, tw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--profile", default="full", choices=["full", "tamed", "safe"])
    ap.add_argument("--qc", type=float, default=QC_NCC)
    a = ap.parse_args()
    V = a.V
    lens = np.load(os.path.join(ML, "bank_lens.npy"))
    K = len(lens)
    banks = os.path.join(ML, "bank_pool.npy")
    jobs = [(k, v, 1000 + k * 97 + v) for k in range(K) for v in range(V)]
    path = os.path.join(ML, "variants.npy")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(K * V, LMAXV))
    vlens = np.zeros(K * V, dtype=np.int32)
    nccs = np.zeros(K * V, dtype=np.float32)
    t0 = time.time()
    n = 0
    n_fb = n_rej = n_tw = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(banks, lens, a.profile)) as pool:
        for k, v, out, L, c, fb, rej, tw in pool.imap_unordered(_work, jobs, chunksize=64):
            arr[k * V + v] = out
            vlens[k * V + v] = L
            nccs[k * V + v] = c
            n += 1
            n_fb += fb
            n_rej += rej
            n_tw += tw
            if n % 20000 == 0:
                print("  %d/%d  %.0fs  兜底 %d 剔除 %d" % (n, len(jobs), time.time() - t0,
                                                            n_fb, n_rej), flush=True)
    arr.flush()
    np.save(os.path.join(ML, "variants_lens.npy"), vlens)
    np.save(os.path.join(ML, "variants_ncc.npy"), nccs)
    json.dump({"V": V, "LMAXV": LMAXV, "K": int(K), "lib": "audiomentations",
               "profile": a.profile, "qc_ncc": a.qc, "minlen": int(MINLEN),
               "n_fallback": int(n_fb), "n_rejected": int(n_rej), "n_twice": int(n_tw)},
              open(os.path.join(ML, "variants_meta.json"), "w"))
    ok = nccs[np.asarray(vlens) > 0]              # 全部入库变体
    print("NCC vs clean template: p10 %.2f p50 %.2f p90 %.2f | >0.5 %.1f%%  >0.8 %.1f%%"
          % (np.percentile(ok, 10), np.median(ok), np.percentile(ok, 90),
             100 * (ok > 0.5).mean(), 100 * (ok > 0.8).mean()), flush=True)
    print("兜底(Compose 抛异常 -> 原样)  %d/%d 条 (%.2f%%)" % (n_fb, len(jobs),
                                                             100.0 * n_fb / max(len(jobs), 1)))
    print("剔除(两条候选都不达标 -> 回退干净模板, NCC 记 1.0)  %d/%d 条 (%.2f%%)"
          % (n_rej, len(jobs), 100.0 * n_rej / max(len(jobs), 1)))
    print("故意 2 倍速(质检之后做, NCC 天然偏低, 不算污染)  %d/%d 条 (%.2f%%)"
          % (n_tw, len(jobs), 100.0 * n_tw / max(len(jobs), 1)))
    print("DONE %d variants (%.2f GB) in %.0fs" % (K * V, K * V * LMAXV * 2 / 1e9, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
