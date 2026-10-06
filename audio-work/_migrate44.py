"""迁移到 44.1 kHz + 变长流模板库：一次性改完所有引用点（跑完即删）。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"

REC_OLD = '''def _recording_pool():
    p = os.path.join(ML, "bg16.npy")
    if not os.path.exists(p):
        import sys
        sys.path.insert(0, D)
        import alab
        x, sr = alab.wav_read(os.path.join(D, "v82_mono.wav"), mono=True)
        x = alab.bandpass(x, sr, 30.0, 7600.0)
        y = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x).astype(np.float32)
        np.save(p, y)
        return np.load(p, mmap_mode="r")
    return np.load(p, mmap_mode="r")'''

REC_NEW = '''def _recording_pool():
    """真实录音背景（--bg-mode recording）。

    文件名带采样率（bg44.npy / bg16.npy）：采样率不匹配时把 16 kHz 的缓冲当 44.1 kHz 用，
    音高与时序都会错。缓存不存在时从真实录音重新生成，优先用高采样率的录音源。
    """
    p = os.path.join(ML, "bg%d.npy" % (SR // 1000))
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    import sys
    sys.path.insert(0, D)
    import alab
    for name in ("nl_mono.wav", os.path.join("real", "ui_demo_mono48k.wav"),
                 os.path.join("samples", "nl_mono.wav"), "v82_mono.wav"):
        src = os.path.join(D, name)
        if not os.path.exists(src):
            continue
        x, sr = alab.wav_read(src, mono=True)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = alab.bandpass(x, sr, 30.0, min(sr, SR) * 0.475)
        y = np.interp(np.arange(int(len(x) * SR / sr)) / SR,
                      np.arange(len(x)) / sr, x).astype(np.float32)
        np.save(p, y)
        print("[synth] recording 背景 <- %s (%.0f s @%d Hz -> %d Hz)"
              % (name, len(y) / SR, sr, SR), flush=True)
        return np.load(p, mmap_mode="r")
    raise FileNotFoundError("找不到可用作 recording 背景的录音（nl_mono.wav 等）")'''

BANK_OLD = '''        self.bank_paths = [os.path.join(ML, b) for b in
                           ["bank_clean.npy", "bank_96k.npy", "bank_48k.npy"]
                           if os.path.exists(os.path.join(ML, b))]
        assert self.bank_paths, "no template bank found - run ml/build_bank.py first"
        self._bk = None'''

BANK_NEW = '''        # 模板库是【变长流】：bank_pool.npy(fp16 拼接) + bank_offs/lens.npy
        self.bank_pool_path = os.path.join(ML, "bank_pool.npy")
        assert os.path.exists(self.bank_pool_path), \\
            "no template bank found - run build_bank.py first"
        self.offs = np.load(os.path.join(ML, "bank_offs.npy"))
        self._pool = None'''

ATOM_OLD = '''                    L = int(self.lens[k])
                    if L < 800:
                        continue
                    bv = self._bk[int(rng.integers(0, len(self._bk)))]
                    w = np.asarray(bv[k, :L], dtype=np.float32)'''

ATOM_NEW = '''                    L = int(self.lens[k])
                    if L < int(0.05 * SR):          # 短于 50 ms 的模板不足以定位
                        continue
                    off = int(self.offs[k])
                    w = np.asarray(self._pool[off:off + L], dtype=np.float32)'''

READ_OLD = '''lens = np.load(os.path.join(ML, "bank_lens.npy"))
bank = np.load(os.path.join(ML, "bank_clean.npy"), mmap_mode="r")'''

READ_NEW = '''lens = np.load(os.path.join(ML, "bank_lens.npy"))
offs = np.load(os.path.join(ML, "bank_offs.npy"))
bank = np.load(os.path.join(ML, "bank_pool.npy"), mmap_mode="r")'''


def rw(name, pairs, count=1):
    p = os.path.join(T, name)
    t = open(p, encoding="utf-8").read()
    n_hit = 0
    for old, new in pairs:
        if old in t:
            n_hit += t.count(old)
            t = t.replace(old, new) if count == 0 else t.replace(old, new, count)
        else:
            print("  !! %s 未命中: %r" % (name, old[:70]))
    open(p, "w", encoding="utf-8").write(t)
    print("%-16s 命中 %d 处" % (name, n_hit))


# 1) 采样率常量
for f in ("synth.py", "build_long.py", "infer.py", "onset_diag.py"):
    rw(f, [("SR = 16000", "SR = 44100")])

# 2) 低通截止（16k 时代的 7.5k）
for f in ("build_long.py", "infer.py", "onset_diag.py"):
    rw(f, [("alab.bandpass(x, sr, 30.0, 7500.0)",
            "alab.bandpass(x, sr, 30.0, min(sr, SR) * 0.475)")])

# 3) synth.py：背景池、模板池、阈值
rw("synth.py", [
    (REC_OLD, REC_NEW),
    ('label_mode="cluster", nclust=512, lmax_fx=16000, use_variants=True,',
     'label_mode="cluster", nclust=512, lmax_fx=88200, use_variants=True,'),
    (BANK_OLD, BANK_NEW),
    ('        if self._bk is None:\n            self._bk = [np.load(p, mmap_mode="r") for p in self.bank_paths]',
     '        if self._pool is None:\n            self._pool = np.load(self.bank_pool_path, mmap_mode="r")'),
    (ATOM_OLD, ATOM_NEW),
    ('                        if rng.random() < 0.30 and L > 1600:',
     '                        if rng.random() < 0.30 and L > int(0.1 * SR):'),
    ('                    L = int(self.vlens[vi])\n                    if L < 800:',
     '                    L = int(self.vlens[vi])\n                    if L < int(0.05 * SR):'),
])

# 4) 评测/推理脚本的模板读取
for f in ("infer.py", "onset_diag.py"):
    rw(f, [(READ_OLD, READ_NEW)])
    rw(f, [("bank[k, :L]", "bank[int(offs[k]):int(offs[k]) + L]")], count=0)

# 5) gen_variants.py：采样率 + 池（变体库本身仍是定长，按 lmax_fx 截断）
rw("gen_variants.py", [
    ("LMAXV = 16000", "LMAXV = 88200          # 每条变体最多保留 2 s"),
    ("sample_rate=16000", "sample_rate=44100"),
    ('    _B["banks"] = [np.load(b, mmap_mode="r") for b in banks]',
     '    _B["offs"] = np.load(os.path.join(ML, "bank_offs.npy"))\n'
     '    _B["pool"] = np.load(banks, mmap_mode="r")'),
    ('    bv = _B["banks"][int(rng.integers(0, len(_B["banks"])))]',
     '    off = int(_B["offs"][k])\n    _pool = _B["pool"]'),
    ("    w = np.asarray(bv[k, :L], dtype=np.float32).copy()",
     "    w = np.asarray(_pool[off:off + L], dtype=np.float32).copy()"),
    ('    banks = [os.path.join(ML, b) for b in ["bank_clean.npy", "bank_96k.npy", "bank_48k.npy"]]',
     '    banks = os.path.join(ML, "bank_pool.npy")'),
])

print("\n=== 复查 ===")
for f in ("core.py", "synth.py", "build_bank.py", "build_long.py", "infer.py", "onset_diag.py", "gen_variants.py"):
    t = open(os.path.join(T, f), encoding="utf-8").read()
    bad = [w for w in ("16000", "bank_clean", "_bk", "7500.0") if w in t]
    print("  %-16s 残留: %s" % (f, bad or "无"))
