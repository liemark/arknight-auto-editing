"""把 .vscode/tasks.json 里训练任务的参数同步到 44.1k / T=20 配置（跑完即删）。"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"F:\杂七杂八\arknight-auto-editing\.vscode\tasks.json"
d = json.load(open(P, encoding="utf-8"))
changed = []
for t in d["tasks"]:
    a = t.get("args")
    if not isinstance(a, list) or "train.py" not in a:
        continue
    before = list(a)
    for i, x in enumerate(a):
        if x == "--T" and i + 1 < len(a):
            a[i + 1] = "20"
        if x == "--bs" and i + 1 < len(a):
            a[i + 1] = "8"
    if "--max-ev" not in a and "--T" in a:
        a[a.index("--T"):a.index("--T")] = ["--max-ev", "192"]
    if a != before:
        changed.append(t["label"])
        # detail 里补一句新配置说明（避免与实际默认值不符）
        t["detail"] = (t.get("detail", "").rstrip() +
                       "   【44.1 kHz 配置】前端 SR=44100/N_FFT=1024/HOP=441/N_MELS=128，"
                       "输出帧仍 20 ms；模板库为变长流（K=22326，含战斗语音与长音效，不截断）；"
                       "窗口 T=20 s，素材可从窗口前开始（窗口内听到中段）。")
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("已更新任务: %s" % ", ".join(changed))
# 回读校验
d2 = json.load(open(P, encoding="utf-8"))
print("JSON 合法，任务数 %d" % len(d2["tasks"]))
for t in d2["tasks"]:
    if isinstance(t.get("args"), list) and "train.py" in t["args"]:
        a = t["args"]
        bits = []
        for k in ("--T", "--bs", "--max-ev", "--layers", "--bg-mode", "--steps", "--resume"):
            if k in a:
                bits.append("%s %s" % (k, a[a.index(k) + 1]))
        print("  %-46s %s" % (t["label"][:44], " ".join(bits)))
