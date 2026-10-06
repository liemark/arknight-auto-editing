"""诊断 render 的 模板->原子 映射：为什么有 68 个事件 no_mapping。纯 CPU，秒级。

复刻 render.py::_map_templates 的规则（键 = (name, bundle) 小写，且在 manifest 里不重复），
再按 group/kind 和 timeline 里的 top-1 类别分别统计未命中。
"""
import collections
import json
import os
import sys

AI = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse"
sys.path.insert(0, AI)
from audio_inverse.manifest import Manifest                            # noqa: E402

bi = json.load(open(os.path.join(AI, "train", "bank_index.json"), encoding="utf-8"))
man = Manifest.load(os.path.join(AI, "data", "manifest.json"))
proj, dup = {}, set()
for a in man.atoms:
    k = (str(a.name).lower(), str(a.bundle).lower())
    if k in proj:
        dup.add(k)
    proj.setdefault(k, a.id)
print("manifest 原子 %d 个  唯一(name,bundle) %d 个  重复键 %d 个" % (len(man.atoms), len(proj), len(dup)))
print("  manifest 里的采样率分布:", collections.Counter(int(a.sr) for a in man.atoms))

ok, miss = {}, collections.Counter()
miss_ex = collections.defaultdict(list)
for i, v in enumerate(bi):
    k = (str(v.get("name", "")).lower(), str(v.get("bundle", "")).lower())
    if k in proj and k not in dup:
        ok[i] = proj[k]
    else:
        g = str(v.get("group") or v.get("kind") or "?")
        miss[g] += 1
        if len(miss_ex[g]) < 3:
            miss_ex[g].append(k)
print("\n模板 -> 原子：%d/%d 命中，未命中 %d" % (len(ok), len(bi), len(bi) - len(ok)))
print("  未命中按 group:", dict(miss))
for g, ex in miss_ex.items():
    print("    %-12s 例: %s" % (g, ex))
print("  重复键(有歧义被排除) 例:", list(dup)[:5])

tl = json.load(open(os.path.join(AI, "data", "atoms", "timeline_nl_mono_v3d.json"), encoding="utf-8"))
cat_ok, cat_bad = collections.Counter(), collections.Counter()
for e in tl:
    t = e["cands"][0]
    k = (str(t["template"]).lower(), str(t["bundle"]).lower())
    if k in proj and k not in dup:
        cat_ok[t["cat"]] += 1
    else:
        cat_bad[t["cat"]] += 1
tot = sum(cat_ok.values()) + sum(cat_bad.values())
print("\ntimeline %d 个事件的 top-1 映射：命中 %d，未命中 %d" % (tot, sum(cat_ok.values()), sum(cat_bad.values())))
print("  未命中按类别:", dict(cat_bad.most_common()))
