
"""为每条模板预生成 V 个增广变体（audiomentations）。

增广会改变音色，但必须保持“这条波形仍然是该模板”。同一组变换参数对不同模板的效果
差异很大，所以每条变体都用零均值归一化相关（NCC）与它的干净模板比对：低于 QC_NCC
就换用 SAFE 规格重做；两条都不达标时回退到干净模板本身，使“入库变体的
NCC 恒 >= QC_NCC”成为不变量。
"""
import os, sys, json, time, argparse
import numpy as np
import multiprocessing as mp
ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.dirname(ML))
from aug_spec import MIN_SAMPLES, build, ncc as _ncc, QC_NCC

LMAXV = 16000
# 短于这个长度的模板要换用"过滤版"规格：有几个变换有最小长度要求，硬跑会抛异常，
# 而异常兜底会把变体变成"原样复制"(NCC=1.0) 却被当成合格 —— 见 aug_spec.MIN_SAMPLES 的注释。
MINLEN = max(MIN_SAMPLES.values()) if MIN_SAMPLES else 0
_B = {}


def _init(banks, lens, profile):
    _B["banks"] = [np.load(b, mmap_mode="r") for b in banks]
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
    bv = _B["banks"][int(rng.integers(0, len(_B["banks"])))]
    L = min(L0, LMAXV)
    w = np.asarray(bv[k, :L], dtype=np.float32).copy()
    if rng.random() < 0.30 and L > 1600:          # SFX cut off mid-way
        L = int(rng.uniform(0.35, 0.95) * L)
        w = w[:L].copy()
    L = len(w)
    short = L < MINLEN
    aug = _B["aug_short"] if short else _B["aug"]
    safe = _B["safe_short"] if short else _B["safe"]
    fb = 0                                        # 1 = 走了异常兜底（原样复制），要报出来
    try:
        y = np.asarray(aug(w.copy(), sample_rate=16000), dtype=np.float32)
    except Exception:
        y = w.copy()
        fb = 1
    c = _ncc(y, w)
    rej = 0                                       # 1 = 两条候选都不达标 -> 回退干净模板
    if c < QC_NCC:
        try:
            y2 = np.asarray(safe(w.copy(), sample_rate=16000), dtype=np.float32)
        except Exception:
            y2 = w.copy()
            fb = 1
        if _ncc(y2, w) > c:
            y, c = y2, _ncc(y2, w)
        # 两条候选都不达标 -> 回退到干净模板本身，NCC 记 1.0。这样“入库变体的
        #   NCC 恒 >= 门限”才成立。少一条增广样本的代价，远小于把一个与自身标签
        #   负相关的波形放进训练集：对原型余弦头，同类样本方向相反会互相抵消，
        #   受损的是全部类别的类中心。
        if c < QC_NCC:
            y, c, rej = w.copy(), 1.0, 1
# 变速样本不放进变体库：波形 NCC 是固定时间对齐的判据，对任何变速都会立刻归零
# （0.90 倍速也只有 0.001），因此它无法给变速样本做质检。真实素材的变速域
# （0.90~1.10）由训练时的在线 chan_fx 覆盖；要更宽的范围应当在 chan_fx 里调，
# 而不是往变体库里放无法质检的样本。
    tw = 0                                        # 统计口径保留
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
    banks = [os.path.join(ML, b) for b in ["bank_clean.npy", "bank_96k.npy", "bank_48k.npy"]]
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
