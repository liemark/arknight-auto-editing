"""修 build_long.py 的输入索引路径，指向 44.1 kHz 版（跑完即删）。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"F:\杂七杂八\arknight-auto-editing\audio-work\audio_inverse\train\build_long.py"
t = open(p, encoding="utf-8").read()
old = 'ip = os.path.join(D, "long", "index_long.json")'
new = ('ip = os.path.join(D, "long44", "index_long.json")      # 44.1 kHz 版\n'
       'if not os.path.exists(ip):\n'
       '    ip = os.path.join(D, "long", "index_long.json")      # 回退：旧的 16 kHz 版')
if old not in t:
    if 'long44/index_long.json' in t:
        print("已经是 44.1 kHz 路径，无需修改")
    else:
        print("!! 未找到索引路径行，需人工检查 build_long.py")
        print([ln for ln in t.splitlines() if "index_long.json" in ln])
    sys.exit(0)
open(p, "w", encoding="utf-8").write(t.replace(old, new))
print("build_long.py 输入索引 -> long44/index_long.json（回退 long/）")
