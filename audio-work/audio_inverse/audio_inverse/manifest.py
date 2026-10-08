"""原子清单: 从解包索引 JSON 生成规范化 manifest.json。

关键设计:
  * 【划分按 group 分组】而不是按原子。`p_imp_3__p_imp_nailgun_h` 的 group 是 `p_imp`,
    同组音效高度相似 (同一角色的不同技能/同一武器的不同档位)。若按原子随机划分,
    测试集会混入训练时见过的近邻 -> "零错配"结论不可信。按 group 划分后,
    测试集里的 group 在训练中从未出现, 这才是可信的零错配。
  * 语音同理: bundle = charId, 划分时同一角色的所有语音(含 jp/cn)必须同侧。
  * 清单里的 sr/ch/dur 一律以【WAV 头实测值】为准, 不信 JSON (JSON 可能过期)。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from .audio import wav_info

# 默认索引文件 (相对 atom_root, 即 _audio_lab/ -> audio_inverse/data/atoms/)
# 顺序 = atom id 顺序, 不要重排: data/bank.npz 与 data/atomlib 的行号按此顺序固化。
DEFAULT_INDEXES = [
    "sfx/index_root.json",
    "sfx/index_player.json",
    "sfx/index_custom_se.json",
    "sfx/index_enemy.json",
    "voice/index_voice_battle_jp.json",
    "voice/index_voice_battle_cn.json",
]

# 已知的 bundle 尾号模式: p_imp_3 -> p_imp, p_skill_14 -> p_skill, act1arkhub -> act1arkhub
def group_of(bundle: str) -> str:
    """bundle 归组: 去掉结尾的 `_<数字>` 或 `_<单个字母>` 分片后缀。

    >>> group_of("p_imp_3"), group_of("p_skill_14"), group_of("general_10")
    ('p_imp', 'p_skill', 'general')
    """
    b = bundle
    while True:
        i = b.rfind("_")
        if i <= 0:
            return b
        tail = b[i + 1:]
        if tail.isdigit() or (len(tail) == 1 and tail.isalpha()):
            b = b[:i]
            continue
        return b


@dataclass
class Atom:
    id: int
    kind: str                 # sfx | voice_battle
    bundle: str
    name: str
    file: str
    sr: int                   # 实测采样率
    ch: int                   # 实测声道数
    dur: float
    frames: int
    group: str
    split: str                # train | val | test  (原子哈希, 覆盖 97%)
    split_group: str = "train"  # 按 group 哈希的旧划分 (测"泛化到新音效"时用它)
    shard: int = 0            # 在 atomlib 中的分片号 (build 时填)
    voice_index: int | None = None
    voice_title: str | None = None
    place_type: str | None = None
    lang: str | None = None


@dataclass
class Manifest:
    atoms: list[Atom] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    source: str = ""
    created: float = 0.0

    # ---- 便捷视图 ---------------------------------------------------------
    def by_id(self, i: int) -> Atom:
        return self.atoms[i]

    @property
    def n(self) -> int:
        return len(self.atoms)

    def ids(self, split: str | None = None, kind: str | None = None) -> list[int]:
        return [a.id for a in self.atoms
                if (split is None or a.split == split) and (kind is None or a.kind == kind)]

    def groups(self, split: str | None = None) -> list[str]:
        seen = {}
        for a in self.atoms:
            if split is None or a.split == split:
                seen[a.group] = 1
        return sorted(seen)

    def split_of_group(self, group: str) -> str:
        for a in self.atoms:
            if a.group == group:
                return a.split
        raise KeyError(group)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "source": self.source,
            "created": self.created,
            "stats": self.stats,
            "atoms": [asdict(a) for a in self.atoms],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        atoms = [Atom(**a) for a in p["atoms"]]
        return Manifest(atoms=atoms, stats=p.get("stats", {}),
                        source=p.get("source", ""), created=p.get("created", 0.0))


# ------------------------------------------------------------------ split 规则
def split_of(group: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
             salt: str = "ai-v1") -> str:
    """按 group 名做稳定哈希划分。同一 group 永远落同一侧。"""
    h = hashlib.sha1(f"{salt}:{group}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    a, b, _ = ratios
    return "train" if u < a else ("val" if u < a + b else "test")


def split_of_atom(key: str, ratios: tuple[float, float, float], salt: str) -> str:
    """按【单个原子】做稳定哈希划分。

    为什么需要它 (check_split.py 实测结论):
      * 按 group 划分的原本目的是"防近邻泄漏", 但全库 22328 条高度重复 ——
        实测测试原子到训练集的最近邻余弦中位 0.992/p10 0.980, 也就是说即使
        按 group 划分, 测试集里也几乎总能找到"几乎一样"的兄弟, 防泄漏目的
        基本没达到; 代价却是 21% 的原子 (4729 条) 从未进入训练, 白白缺监督。
      * 我们的目标是"在已知原子库里找准并抵消", 全库原子都该被模型见过。
        所以默认改成原子级划分 (97/1.5/1.5), 覆盖率 97%;
        同时保留 group 级划分的结果在 split_group 字段, 需要测"泛化到新音效"
        时把它当 split 用即可。
    """
    h = hashlib.sha1(f"{salt}:atom:{key}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    a, b, _ = ratios
    return "train" if u < a else ("val" if u < a + b else "test")


# ------------------------------------------------------------------- 主构建流程
def build_manifest(atom_root: str, out_path: str, *,
                   indexes: Iterable[str] = DEFAULT_INDEXES,
                   ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                   salt: str = "ai-v1",
                   verify: bool = True,
                   verbose: bool = True) -> Manifest:
    """读索引 JSON -> 规范化 -> 校验 WAV 头 -> 划分 -> 写 manifest.json。"""
    atoms: list[Atom] = []
    seen_names: dict[str, int] = {}
    missing: list[str] = []
    bad: list[str] = []
    sr_hist: dict[int, int] = {}
    ch_hist: dict[int, int] = {}
    by_kind: dict[str, int] = {}

    for rel in indexes:
        path = os.path.join(atom_root, rel.replace("/", os.sep))
        if not os.path.exists(path):
            if verbose:
                print(f"[manifest] 跳过不存在的索引: {rel}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.items() if isinstance(data, dict) else ((v.get("name"), v) for v in data)
        kind_default = "voice_battle" if "voice_battle" in rel else "sfx"
        for key, meta in items:
            fp = meta.get("file")
            if not fp or not isinstance(fp, str):
                bad.append(f"{rel}:{key} 缺少 file")
                continue
            if not os.path.exists(fp):
                missing.append(fp)
                continue
            info = wav_info(fp)
            if verify and info.frames <= 0:
                bad.append(f"{fp} 零长度")
                continue
            bundle = str(meta.get("bundle") or os.path.basename(os.path.dirname(fp)) or "unknown")
            name = str(meta.get("name") or os.path.splitext(os.path.basename(fp))[0])
            kind = str(meta.get("kind") or kind_default)
            grp = group_of(bundle)
            seen_names[name] = seen_names.get(name, 0) + 1
            sr_hist[info.sr] = sr_hist.get(info.sr, 0) + 1
            ch_hist[info.ch] = ch_hist.get(info.ch, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            atoms.append(Atom(
                id=len(atoms), kind=kind, bundle=bundle, name=name, file=os.path.abspath(fp),
                sr=info.sr, ch=info.ch, dur=info.dur, frames=info.frames, group=grp,
                split=split_of_atom(f"{bundle}/{name}", ratios, salt),
                split_group=split_of(grp, ratios, salt),
                voice_index=meta.get("voiceIndex"),
                voice_title=meta.get("voiceTitle"),
                place_type=meta.get("placeType"),
                lang=meta.get("lang"),
            ))

    # 统计
    sp = {"train": 0, "val": 0, "test": 0}
    for a in atoms:
        sp[a.split] += 1
    stats = {
        "n": len(atoms),
        "by_kind": by_kind,
        "by_sr": {str(k): v for k, v in sorted(sr_hist.items())},
        "by_ch": {str(k): v for k, v in sorted(ch_hist.items())},
        "by_split": sp,
        "n_groups": len({a.group for a in atoms}),
        "groups_by_split": {s: len({a.group for a in atoms if a.split == s}) for s in sp},
        "n_missing": len(missing),
        "n_bad": len(bad),
        "dup_names": sum(1 for v in seen_names.values() if v > 1),
        "total_dur": round(sum(a.dur for a in atoms), 2),
    }
    man = Manifest(atoms=atoms, stats=stats,
                   source=os.path.abspath(atom_root), created=time.time())
    man.save(out_path)

    if verbose:
        print(f"[manifest] 原子 {stats['n']}  组 {stats['n_groups']}  -> {out_path}")
        print(f"[manifest] kind={by_kind} sr={stats['by_sr']} ch={stats['by_ch']}")
        print(f"[manifest] split={sp}  groups={stats['groups_by_split']}")
        print(f"[manifest] 缺失={len(missing)} 异常={len(bad)} 重名={stats['dup_names']} "
              f"总时长={stats['total_dur']/3600:.2f}h")
        if missing:
            print(f"[manifest] 缺失样例: {missing[:3]}")
        if bad:
            print(f"[manifest] 异常样例: {bad[:3]}")
    return man


def load_or_build(data_root: str, atom_root: str, *,
                  indexes: Iterable[str] = DEFAULT_INDEXES,
                  rebuild: bool = False,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                  salt: str = "ai-v1") -> Manifest:
    """有 manifest.json 就加载, 否则构建。"""
    p = os.path.join(data_root, "manifest.json")
    if os.path.exists(p) and not rebuild:
        return Manifest.load(p)
    os.makedirs(data_root, exist_ok=True)
    return build_manifest(atom_root, p, indexes=indexes, ratios=ratios, salt=salt)


def review_groups(man: Manifest, n: int = 12) -> str:
    """抽样打印分组结果, 用于人工确认划分合理性。"""
    out = []
    for s in ("train", "val", "test"):
        gs = man.groups(s)
        out.append(f"[{s}] {len(gs)} 组, 样例: {gs[:n]}")
    return "\n".join(out)
