"""原子库: 把 22328 条 WAV 打成一个紧凑的二进制池 + 索引 + 统计。

为什么不用一堆小文件 / 一个 1.5GB npy:
  * 训练时要在 GPU 上随机取原子, 每个 epoch 数十万次; 小文件系统调用是瓶颈。
  * 单文件 + offset 表 = 一次 `np.memmap` 随机读, 零解析开销, 且不占内存。
  * 波形按【原始采样率】存 int16: 语音 16k、SFX 44.1k/48k 混存。
    若统一升到 48k, 体积会膨胀 3 倍(约 1.5GB)且语音那部分毫无收益。

磁盘布局 (data/atomlib/):
  wav.i16    所有原子首尾相接的 int16 原始采样
  index.npz  ids, offset, length, sr, ch, shard
  stats.npz  rms_db, peak, active_ratio(非静音样本占比), centroid_hz, bw_hi_ratio
  meta.json  构建参数与校验和
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Sequence

import numpy as np

from .audio import SR, read_wav, resample, trim_silence
from .manifest import Manifest

WAV_FILE = "wav.i16"
IDX_FILE = "index.npz"
STAT_FILE = "stats.npz"
META_FILE = "meta.json"


class AtomLib:
    """只读原子池。线程内安全; 不要跨进程写。"""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, META_FILE), encoding="utf-8") as f:
            self.meta = json.load(f)
        z = np.load(os.path.join(self.root, IDX_FILE))
        self.ids = z["ids"]
        self.offset = z["offset"]
        self.length = z["length"]
        self.sr = z["sr"]
        self.ch = z["ch"]
        self.shard = z["shard"] if "shard" in z else np.zeros_like(self.ids)
        s = np.load(os.path.join(self.root, STAT_FILE), allow_pickle=False)
        self.stats = s["values"]                       # [n, n_stat]
        # 统计名放 meta.json (不用 npz 的 object 数组: 那需要 allow_pickle=True)
        self.stat_names = list(self.meta.get("stat_names", STAT_NAMES))
        assert self.stats.shape[1] == len(self.stat_names), "统计表列数与名称不符"
        self._wav = None

    # ---- 基础属性 ---------------------------------------------------------
    def __len__(self) -> int:
        return int(self.ids.size)

    @property
    def n_stat(self) -> int:
        return len(self.stat_names)

    def stat(self, name: str, ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        k = self.stat_names.index(name)
        v = self.stats[:, k]
        return v if ids is None else v[np.asarray(ids)]

    @property
    def wav(self) -> np.memmap:
        if self._wav is None:
            self._wav = np.memmap(os.path.join(self.root, WAV_FILE), dtype="<i2", mode="r")
        return self._wav

    # ---- 取波形 -----------------------------------------------------------
    def raw(self, i: int) -> tuple[np.ndarray, int]:
        """取第 i 条原子的原始采样率 float32 波形 (单声道)。"""
        o, n = int(self.offset[i]), int(self.length[i])
        x = np.asarray(self.wav[o:o + n], dtype=np.float32) / 32768.0
        return x, int(self.sr[i])

    def get(self, i: int, *, sr_out: int | None = None,
            trim: bool = False, preload: bool = True) -> np.ndarray:
        """取波形并可选重采样到 sr_out。trim=True 时裁掉首尾静音。"""
        x, sr = self.raw(i)
        if trim:
            x = trim_silence(x, sr)
        if sr_out is not None and sr != sr_out:
            x = resample(x, sr, sr_out)
        return np.ascontiguousarray(x, dtype=np.float32) if preload else x

    def get48k(self, i: int, *, trim: bool = False) -> np.ndarray:
        return self.get(i, sr_out=SR, trim=trim)

    def batch48k(self, ids: Sequence[int], *, max_len: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        """批量取 48k 波形 -> (波形 [B, L] 零填充, 长度 [B])。"""
        xs = [self.get48k(int(i)) for i in ids]
        lens = np.array([len(x) for x in xs], dtype=np.int64)
        L = int(lens.max()) if lens.size else 0
        if max_len:
            L = min(L, int(max_len))
        out = np.zeros((len(xs), L), dtype=np.float32)
        for k, x in enumerate(xs):
            m = min(L, x.size)
            out[k, :m] = x[:m]
        return out, np.minimum(lens, L)

    def iter_raw(self, ids: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray, int]]:
        for i in (range(len(self)) if ids is None else ids):
            x, sr = self.raw(int(i))
            yield int(i), x, sr

    def dur(self, i: int) -> float:
        return float(self.length[i]) / float(self.sr[i])

    # ---- 分组索引 (给困难负样本用) ---------------------------------------
    def group_index(self, groups: Sequence[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            out.setdefault(g, []).append(i)
        return out


# --------------------------------------------------------------------- 构建
def _atom_stats(x: np.ndarray, sr: int) -> tuple[float, float, float, float, float]:
    """(rms_db, peak, active_ratio, centroid_hz, hi_ratio)"""
    if x.size == 0:
        return -120.0, 0.0, 0.0, 0.0, 0.0
    xf = x.astype(np.float64)
    r = float(np.sqrt(np.mean(xf * xf)))
    rms_db = 20 * np.log10(max(r, 1e-12))
    pk = float(np.max(np.abs(xf)))
    # 活跃样本占比 (滑窗 RMS > -60dBFS)
    win = max(1, int(sr * 0.01))
    n = x.size // win
    if n >= 2:
        e = np.sqrt((xf[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
        active = float(np.mean(20 * np.log10(e) > -60.0))
    else:
        active = 1.0
    # 谱重心与高频能量比 (8k 以上), 用于"语音 16k 上限"的统计确认
    nfft = 1
    take = xf[: min(xf.size, sr * 4)]
    while nfft < take.size:
        nfft *= 2
    X = np.abs(np.fft.rfft(take * np.hanning(take.size), n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    tot = float(X.sum()) + 1e-12
    centroid = float((X * f).sum() / tot)
    hi = float(X[f > 8000.0].sum() / tot)
    return rms_db, pk, active, centroid, hi


STAT_NAMES = ["rms_db", "peak", "active_ratio", "centroid_hz", "hi8k_ratio"]


def build_atomlib(man: Manifest, out_dir: str, *, sr_out: int = SR,
                  hop_resample: bool = False, verbose: bool = True) -> AtomLib:
    """把清单里的原子写成池。波形按原始 sr 存 int16 (不做重采样)。"""
    assert not hop_resample, "hop_resample 已废弃: 池内保持原始采样率"
    os.makedirs(out_dir, exist_ok=True)
    n = man.n
    wav_path = os.path.join(out_dir, WAV_FILE)
    offset = np.zeros(n, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    srs = np.zeros(n, dtype=np.int32)
    chs = np.zeros(n, dtype=np.int32)
    shards = np.zeros(n, dtype=np.int32)
    stats = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)

    t0 = time.time()
    pos = 0
    with open(wav_path, "wb", buffering=1 << 20) as f:
        for a in man.atoms:
            x = read_wav(a.file, mono=True)          # 原始 sr
            i16 = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
            if i16.size == 0:                        # 极少数空文件兜底
                i16 = np.zeros(1, dtype="<i2")
            f.write(i16.tobytes())
            offset[a.id] = pos
            length[a.id] = int(i16.size)
            srs[a.id] = int(a.sr)
            chs[a.id] = int(a.ch)
            shards[a.id] = int(a.id // 2048)
            stats[a.id] = _atom_stats(x, int(a.sr))
            pos += i16.size
            if verbose and (a.id + 1) % 4000 == 0:
                print(f"[atomlib] {a.id+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    np.savez(os.path.join(out_dir, IDX_FILE),
             ids=np.arange(n, dtype=np.int64), offset=offset, length=length,
             sr=srs, ch=chs, shard=shards)
    np.savez(os.path.join(out_dir, STAT_FILE), values=stats)
    meta = {
        "n": int(n),
        "n_samples": int(pos),
        "bytes": int(pos * 2),
        "total_dur": round(float(sum(a.dur for a in man.atoms)), 2),
        "built": time.time(),
        "manifest_source": man.source,
        "stat_names": STAT_NAMES,
        "sr_hist": {str(k): int(v) for k, v in
                    zip(*np.unique(srs, return_counts=True))},
    }
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"[atomlib] {n} 原子, {meta['bytes']/1e6:.1f} MB int16, "
              f"总时长 {meta['total_dur']/3600:.2f}h, 用时 {time.time()-t0:.1f}s -> {out_dir}")
    return AtomLib(out_dir)


def load_or_build_atomlib(man: Manifest, out_dir: str, *, rebuild: bool = False,
                          verbose: bool = True) -> AtomLib:
    if not rebuild and os.path.exists(os.path.join(out_dir, META_FILE)):
        lib = AtomLib(out_dir)
        if len(lib) == man.n:
            return lib
        if verbose:
            print(f"[atomlib] 清单({man.n})与池({len(lib)})不一致, 重建")
    return build_atomlib(man, out_dir, verbose=verbose)
