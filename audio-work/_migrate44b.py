"""第二批迁移：窗口长度 + 中段播放 + 44.1k 下的采样数阈值（跑完即删）。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train"


def rw(name, pairs):
    p = os.path.join(T, name)
    t = open(p, encoding="utf-8").read()
    hit = 0
    for old, new in pairs:
        if old in t:
            hit += t.count(old)
            t = t.replace(old, new)
        else:
            print("  !! %s 未命中: %r" % (name, old[:70]))
    open(p, "w", encoding="utf-8").write(t)
    print("%-16s 命中 %d 处" % (name, hit))


rw("train.py", [
    ('p.add_argument("--bs", type=int, default=16,',
     'p.add_argument("--bs", type=int, default=8,'),
    ('p.add_argument("--T", type=float, default=10.0,',
     'p.add_argument("--T", type=float, default=20.0,'),
    ('p.add_argument("--max-ev", type=int, default=128,',
     'p.add_argument("--max-ev", type=int, default=192,'),
    ('p.add_argument("--left-pad", type=float, default=0.25,',
     'p.add_argument("--left-pad", type=float, default=0.0,'),
])

rw("synth.py", [
    # 44.1 kHz 下 50 ms = 2205 采样（原来 800 是 16 kHz 的 50 ms）
    ("left_pad=0.25, p_mid=0.3, src_span_lo=0.25, src_rate_ref=7.5, min_ev_samp=800,",
     "left_pad=0.0, p_mid=0.3, src_span_lo=0.25, src_rate_ref=7.5, min_ev_samp=2205,"),
    # left_pad=0 -> 允许提前整个窗口长度（这样长素材可以在窗口前开始、窗口内听到中段）
    ("        self.left_pad = float(left_pad)          # 源的 onset 最早能比 0 提前多久(秒)",
     "        # left_pad<=0 -> 取整个窗口长度：素材可以从窗口前开始播，窗口里听到的是中段\n"
     "        self.left_pad = float(left_pad) if float(left_pad) > 0 else float(T)"),
])

print("\n=== 复查 ===")
for f in ("train.py", "synth.py"):
    t = open(os.path.join(T, f), encoding="utf-8").read()
    for key in ("default=20.0", "default=0.0", "min_ev_samp=2205", "else float(T)"):
        if key in t:
            print("  %-12s 含 %s" % (f, key))
