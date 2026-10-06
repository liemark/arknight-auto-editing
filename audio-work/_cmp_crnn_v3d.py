"""对比 crnn-a 与 v3d 的前 10000 步轨迹（从训练日志里解析同一组步数）。"""
import re
import os

T = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train\ckpt_tmpl"
PAT = re.compile(r"(\d+)/(\d+)\s*\|.*?lo\s+([\d.]+)\s+pk\s+([\d.]+)\s+off\s+([\d.]+)\s+"
                 r"le\s+([\d.]+)\s+F1\s+([\d.]+)\s*\|\s*top1\s+([\d.]+)%\s+top5\s+([\d.]+)%"
                 r"\s*\|\s*([\d.]+)\s*it/s")


def load(name):
    p = os.path.join(T, name)
    out = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if m:
            st = int(m.group(1))
            out[st] = dict(lo=float(m.group(3)), pk=float(m.group(4)), off=float(m.group(5)),
                           le=float(m.group(6)), f1=float(m.group(7)),
                           t1=float(m.group(8)), t5=float(m.group(9)), its=float(m.group(10)))
    return out


a = load("train_crnn-a-9月18日.log")     # CRNN（你刚启动的）
b = load("train_v3d-9月18日.log")        # transformer 基线
print("crnn 日志步数 %d 条, v3d 日志步数 %d 条" % (len(a), len(b)))
STEPS = [2000, 5000, 8000, 10000, 11000, 12000]


def near(d, s):
    """取最接近 s 的那条记录。"""
    ks = [k for k in d if k <= s]
    return d[max(ks)] if ks else None


print()
print("%-6s | %-38s | %-38s" % ("步数", "CRNN (crnn-a)", "Transformer (v3d)"))
print("%-6s | %-38s | %-38s" % ("", "lo    le     F1    top1   it/s", "lo    le     F1    top1   it/s"))
print("-" * 90)
for s in STEPS:
    ra, rb = near(a, s), near(b, s)
    fa = ("%.3f  %5.2f  %.3f  %5.1f%%  %.1f" % (ra["lo"], ra["le"], ra["f1"], ra["t1"], ra["its"])) if ra else "-"
    fb = ("%.3f  %5.2f  %.3f  %5.1f%%  %.1f" % (rb["lo"], rb["le"], rb["f1"], rb["t1"], rb["its"])) if rb else "-"
    print("%-6d | %-38s | %-38s" % (s, fa, fb))
print()
# top1 首次超过若干门槛的步数
print("top1 首次达到阈值（步数）：")
print("%-10s %-14s %-14s" % ("阈值", "CRNN", "Transformer"))
for thr in (1.0, 3.0, 5.0, 10.0, 20.0):
    fa = next((k for k in sorted(a) if a[k]["t1"] >= thr), None)
    fb = next((k for k in sorted(b) if b[k]["t1"] >= thr), None)
    print("%-10s %-14s %-14s" % ("%.0f%%" % thr, fa if fa else "未达到", fb if fb else "未达到"))
print()
print("le 下降速度（同一步数对比, le 越小越好）：")
for s in (2000, 5000, 10000):
    ra, rb = near(a, s), near(b, s)
    if ra and rb:
        print("  步 %5d:  CRNN %.3f   Transformer %.3f   差 %+.3f" % (s, ra["le"], rb["le"], ra["le"] - rb["le"]))
