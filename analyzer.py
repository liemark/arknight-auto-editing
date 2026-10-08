# analyzer.py —— 模板 / 识别 / 分析 / 导出（matcher + exporter 已并入本文件）
# 其它模块：app_core(配置/事件) gpu_caps(GPU/FFmpeg) pipeline(解码流水线)

from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import time
from dataclasses import dataclass, field
import cv2
import numpy as np
import app_core
import gpu_caps
from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)
import bisect
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any
import pipeline

# matcher.py —— 模板匹配：多后端（CUDA / OpenCL / 批量 CPU / 逐帧 CPU / 进程池）+ 自动测速选路
#
# 掩码 NCC（与 cv2.matchTemplate(TM_CCOEFF_NORMED, mask=) 数学等价）：
#   maskedNCC(I,T,M) = Σ[(T-μT)(I-μI)M] / sqrt(Σ[(T-μT)²M] · Σ[(I-μI)²M])
#   μT/μI 是「只在掩码像素上」统计的均值。
# t' = (T-μT)·M 天然零均值，于是只需 2~3 次互相关就能算完整批：
#   num = corr(I, t')，ΣI²M = corr(I², M)，ΣIM = corr(I, M)
#   den = ||t'|| · sqrt(ΣI²M - (ΣIM)²/n)，分数 = num/den（den<=0 时为 0）
#
# 批量化：把 N 帧的 ROI 竖着叠成一张高图，帧间插 (kh-1) 行 0 填充，
# 每帧的有效结果行恰为 [:roi_h-kh+1]，跨帧窗口落在被丢弃的区间里。



import concurrent.futures
import hashlib
import multiprocessing
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import app_core
import gpu_caps
from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)

# ===============================================================
#  后端标识
# ===============================================================

BACKEND_CV2_DIRECT = "cv2_direct"     # 逐帧 cv2.matchTemplate(mask=)，最保守的基准
BACKEND_CPU_BATCH = "cpu_batch"       # 单进程批量 TM_CCORR（band 堆叠）
BACKEND_CPU_POOL = "cpu_pool"         # 多进程池（兼容旧行为）
BACKEND_OPENCL = "opencl"             # cv2 UMat（T-API），NVIDIA/AMD/Intel 通用
BACKEND_TORCH_CUDA = "torch_cuda"
BACKEND_TORCH_DML = "torch_dml"

BACKEND_LABEL = {
    BACKEND_CV2_DIRECT: "cv2 直算",
    BACKEND_CPU_BATCH: "CPU 批量",
    BACKEND_CPU_POOL: "CPU 多进程池",
    BACKEND_OPENCL: "OpenCL(T-API)",
    BACKEND_TORCH_CUDA: "CUDA(torch)",
    BACKEND_TORCH_DML: "DirectML(torch)",
}

BACKEND_PREFERENCE = [
    BACKEND_TORCH_CUDA, BACKEND_TORCH_DML, BACKEND_OPENCL,
    BACKEND_CPU_BATCH, BACKEND_CV2_DIRECT, BACKEND_CPU_POOL,
]

AUTOTUNE_VERSION = 1

_CATEGORIES = ('pause', 'speed_1x', 'speed_2x', 'speed_0_2x')


# ===============================================================
#  逐帧实现（基准/兜底；analyzer 会 re-export 这两个名字）
# ===============================================================

def get_best_score(gray_frame: np.ndarray, templates: list, proc_res: tuple) -> float:
    max_score = -1.0
    fh, fw = gray_frame.shape
    for t in templates:
        erx, ery, erw, erh = t['cached_roi']
        t_r, m_r = t['cached_t'], t['cached_m']
        erw, erh = min(fw - erx, erw), min(fh - ery, erh)

        if erw <= 0 or erh <= 0:
            continue
        roi = gray_frame[ery:ery + erh, erx:erx + erw]
        if roi.shape[0] < t_r.shape[0] or roi.shape[1] < t_r.shape[1]:
            continue

        res = cv2.matchTemplate(roi, t_r, cv2.TM_CCOEFF_NORMED, mask=m_r)
        _, score, _, _ = cv2.minMaxLoc(res)
        if np.isfinite(score):
            max_score = max(max_score, score)
    return max_score


def classify_gray(gray: np.ndarray, configs: dict,
                  thresholds: dict, proc_res: tuple) -> int:
    if configs['pause'] and get_best_score(gray, configs['pause'], proc_res) >= thresholds['pause']:
        return FRAME_TYPE_PAUSE
    x1s = get_best_score(gray, configs['speed_1x'], proc_res) if configs['speed_1x'] else -1.0
    x2s = get_best_score(gray, configs['speed_2x'], proc_res) if configs['speed_2x'] else -1.0
    if x1s >= thresholds['speed_1x'] and x1s > x2s:
        return FRAME_TYPE_1X
    if x2s >= thresholds['speed_2x'] and x2s > x1s:
        return FRAME_TYPE_2X
    if configs['speed_0_2x']:
        s02 = get_best_score(gray, configs['speed_0_2x'], proc_res)
        # 规则：play 命中、且 1x/2x 都没命中（也不是暂停）→ 0.2 倍速。
        # 1x/2x 上面已经优先判过，这里不再比大小。
        if s02 >= thresholds['speed_0_2x']:
            return FRAME_TYPE_0_2X
    return FRAME_TYPE_NORMAL


def _scores_legacy(gray_batch: np.ndarray, configs: dict, proc_res: tuple) -> dict:
    """逐帧（无短路）算出四类最佳分数，用于与批量后端做数值对照。"""
    out = {c: np.full(len(gray_batch), -1.0, np.float32) for c in _CATEGORIES}
    for i, gray in enumerate(gray_batch):
        for c in _CATEGORIES:
            if configs.get(c):
                out[c][i] = get_best_score(gray, configs[c], proc_res)
    return out


def classify_from_scores(scores: dict, has_cat: dict, thresholds: dict) -> np.ndarray:
    """规则与 classify_gray 一致，只是去掉短路。"""
    n = len(next(iter(scores.values()))) if scores else 0
    out = np.full(n, FRAME_TYPE_NORMAL, np.int8)
    if n == 0:
        return out

    pause_hit = np.zeros(n, bool)
    if has_cat.get('pause') and 'pause' in scores:
        pause_hit = scores['pause'] >= thresholds['pause']
        out[pause_hit] = FRAME_TYPE_PAUSE

    pending = ~pause_hit
    neg = np.full(n, -1.0, np.float32)
    x1s = scores.get('speed_1x', neg) if has_cat.get('speed_1x') else neg
    x2s = scores.get('speed_2x', neg) if has_cat.get('speed_2x') else neg

    m1 = pending & (x1s >= thresholds['speed_1x']) & (x1s > x2s)
    out[m1] = FRAME_TYPE_1X
    m2 = pending & (x2s >= thresholds['speed_2x']) & (x2s > x1s)
    out[m2] = FRAME_TYPE_2X

    if has_cat.get('speed_0_2x') and 'speed_0_2x' in scores:
        # 规则：play 命中、1x/2x 都没命中（也不是暂停）→ 0.2 倍速。
        # 1x/2x 在上一段已经优先判过（m1/m2），这里不比大小。
        s02 = scores['speed_0_2x']
        m02 = pending & ~m1 & ~m2 & (s02 >= thresholds['speed_0_2x'])
        out[m02] = FRAME_TYPE_0_2X
    return out


# ===============================================================
#  模板编译
# ===============================================================

@dataclass
class TplSpec:
    cat: str
    index: int
    roi: tuple[int, int, int, int]     # 已按帧尺寸裁剪的 (erx, ery, erw, erh)
    th: int
    tw: int
    t_f: np.ndarray                    # float32 原模板
    mask_f: np.ndarray                 # float32 0/1 掩码
    tc: np.ndarray                     # 零均值掩码模板 (T-μT)·M
    tn: float                          # ||tc||
    n: float                           # ΣM
    valid: bool
    reason: str = ""


@dataclass
class CompiledTemplates:
    cats: dict = field(default_factory=dict)
    has: dict = field(default_factory=dict)
    proc_res: tuple = (400, 225)
    counts: dict = field(default_factory=dict)

    def signature(self) -> str:
        h = hashlib.sha1()
        h.update(f"{self.proc_res}|".encode())
        for cat in sorted(self.cats):
            for sp in self.cats[cat]:
                h.update(f"{cat}:{sp.roi}:{sp.th}x{sp.tw}:{sp.tn:.6f}:{sp.n}:{sp.valid}".encode())
                h.update(np.ascontiguousarray(sp.t_f).tobytes()[:4096])
        return h.hexdigest()[:16]

    @property
    def total_templates(self) -> int:
        return sum(len(v) for v in self.cats.values())


def compile_templates(configs: dict, proc_res: tuple,
                      frame_wh: tuple | None = None) -> CompiledTemplates:
    """把 load_templates 的配置编译成可批量计算的核。

    frame_wh 给定时按 get_best_score 的裁剪逻辑预判每个模板是否可用，
    保证与逐帧结果一致。
    """
    pw, ph = int(proc_res[0]), int(proc_res[1])
    fw, fh = (int(frame_wh[0]), int(frame_wh[1])) if frame_wh else (pw, ph)

    compiled = CompiledTemplates(proc_res=(pw, ph))
    for cat in _CATEGORIES:
        specs: list[TplSpec] = []
        for idx, t in enumerate(configs.get(cat) or []):
            erx, ery, erw, erh = [int(v) for v in t['cached_roi']]
            t_img = t['cached_t']
            m_img = t['cached_m']
            th, tw = int(t_img.shape[0]), int(t_img.shape[1])
            erw_eff = min(fw - erx, erw)
            erh_eff = min(fh - ery, erh)
            valid = True
            reason = ""
            if erw_eff <= 0 or erh_eff <= 0:
                valid, reason = False, "ROI 超出帧范围"
            elif erh_eff < th or erw_eff < tw:
                valid, reason = False, "ROI 小于模板"
            elif m_img.shape[:2] != (th, tw):
                valid, reason = False, "掩码与模板尺寸不一致"

            mask_f = None
            n = 0.0
            if valid:
                mask_f = (m_img > 0).astype(np.float32)
                n = float(mask_f.sum())
                if n <= 0:
                    valid, reason = False, "掩码为空"
            if valid:
                t_f = t_img.astype(np.float32)
                mu = float((t_f * mask_f).sum() / n)
                tc = (t_f - mu) * mask_f
                tn = float(np.sqrt(float((tc * tc).sum())))
                if tn <= 0:
                    valid, reason = False, "模板方差为 0"
            if not valid:
                blank = np.zeros((1, 1), np.float32)
                specs.append(TplSpec(cat, idx, (erx, ery, max(0, erw_eff), max(0, erh_eff)),
                                     th, tw, blank, blank, blank, 0.0, 0.0, False, reason))
                continue

            specs.append(TplSpec(
                cat=cat, index=idx,
                roi=(erx, ery, int(erw_eff), int(erh_eff)),
                th=th, tw=tw, t_f=t_f, mask_f=mask_f,
                tc=np.ascontiguousarray(tc), tn=tn, n=n, valid=True,
            ))
        compiled.cats[cat] = specs
        compiled.has[cat] = any(sp.valid for sp in specs)
        compiled.counts[cat] = len(specs)
    return compiled


# ===============================================================
#  互相关与批量打分
# ===============================================================

def _corr_cv2(stack: np.ndarray, kernel: np.ndarray, use_umat: bool = False) -> np.ndarray:
    """有效区（valid）互相关：输出 (H-kh+1, W-kw+1)。"""
    if use_umat:
        res = cv2.matchTemplate(cv2.UMat(np.ascontiguousarray(stack)),
                                cv2.UMat(np.ascontiguousarray(kernel)),
                                cv2.TM_CCORR)
        return res.get()
    return cv2.matchTemplate(stack, kernel, cv2.TM_CCORR)


def _scores_cv2_batched(gray_batch: np.ndarray, compiled: CompiledTemplates,
                        use_umat: bool = False) -> dict:
    n_frames = gray_batch.shape[0]
    scores: dict[str, np.ndarray] = {}
    for cat in _CATEGORIES:
        best = np.full(n_frames, -1.0, np.float32)
        for sp in compiled.cats.get(cat, ()):
            if not sp.valid:
                continue
            erx, ery, erw, erh = sp.roi
            kh, kw = sp.th, sp.tw
            oh, ow = erh - kh + 1, erw - kw + 1
            if oh <= 0 or ow <= 0:
                continue
            pitch = erh + kh - 1
            stack = np.zeros((n_frames * pitch, erw), np.float32)
            roi = gray_batch[:, ery:ery + erh, erx:erx + erw].astype(np.float32, copy=False)
            stack.reshape(n_frames, pitch, erw)[:, :erh, :] = roi

            # 输出行号 == 窗口顶行号，故每帧有效行为 [i*pitch, i*pitch + oh)
            row_idx = (np.arange(n_frames)[:, None] * pitch + np.arange(oh)[None, :]).ravel()

            def _take(stack_img, kernel):
                return _corr_cv2(stack_img, kernel, use_umat)[row_idx].reshape(n_frames, oh, ow)

            c1 = _take(stack, sp.tc)
            c2 = _take(stack, sp.mask_f)
            c3 = _take(stack * stack, sp.mask_f)

            with np.errstate(divide="ignore", invalid="ignore"):
                var = c3 - (c2 * c2) / sp.n
                np.maximum(var, 0.0, out=var)
                den = sp.tn * np.sqrt(var)
                sc = np.where(den > 0, c1 / den, 0.0)
            sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
            best = np.maximum(best, sc.max(axis=(1, 2)))
        scores[cat] = best
    return scores


def _resolve_torch_device(device: str):
    import torch
    if device == "dml":
        import torch_directml  # type: ignore
        return torch, torch_directml.device()
    return torch, torch.device(device)


def _torch_kernel_cache(compiled: CompiledTemplates, dev) -> list:
    """把每个模板的卷积核常驻显存（不在每批重复上传小核）。"""
    cache = getattr(compiled, "_torch_dev_cache", None)
    if cache is None:
        cache = {}
        setattr(compiled, "_torch_dev_cache", cache)
    key = str(dev)
    entry = cache.get(key)
    if entry is not None:
        return entry

    torch, _ = _resolve_torch_device("cuda" if "cuda" in key else "dml")
    entry = []
    for cat in _CATEGORIES:
        for sp in compiled.cats.get(cat, ()):
            if not sp.valid:
                continue
            # 一次 conv 同时得到 c1=corr(I,t') 与 c2=ΣI·M（输出通道堆叠）
            wk = torch.from_numpy(
                np.ascontiguousarray(np.stack([sp.tc, sp.mask_f]))
            ).to(dev).view(2, 1, sp.th, sp.tw)
            km = torch.from_numpy(np.ascontiguousarray(sp.mask_f)).to(dev).view(1, 1, sp.th, sp.tw)
            entry.append((cat, sp, wk, km, 1.0 / sp.n))
    cache[key] = entry
    return entry


def _scores_torch(gray_batch: np.ndarray, compiled: CompiledTemplates,
                  device: str = "cuda") -> dict:
    """torch 批量化。

    PyTorch 的 conv2d 是互相关（不翻转核），与公式同向；
    逐元素后处理放回 numpy，且每批只在最后同步一次。
    """
    import torch
    import torch.nn.functional as F

    _torch_mod, dev = _resolve_torch_device(device)
    # 只上传 uint8，且只对 ROI 做 uint8→float：整帧转 float32 代价高得多
    xu = torch.from_numpy(np.ascontiguousarray(gray_batch)).to(dev)
    n_frames = int(xu.shape[0])

    best = {c: np.full(n_frames, -1.0, np.float32) for c in _CATEGORIES}
    pending: list[tuple[str, TplSpec, float, object]] = []
    for cat, sp, wk, km, inv_n in _torch_kernel_cache(compiled, dev):
        erx, ery, erw, erh = sp.roi
        roi = xu[:, ery:ery + erh, erx:erx + erw].unsqueeze(1).float()
        roi2 = roi * roi
        a = F.conv2d(roi, wk, stride=1)
        c3 = F.conv2d(roi2, km, stride=1)
        pending.append((cat, sp, inv_n, torch.cat((a, c3), dim=1)))

    if pending and getattr(dev, "type", "") == "cuda":
        torch.cuda.synchronize()

    for cat, sp, inv_n, packed_t in pending:
        arr = packed_t.cpu().numpy()
        # 转成 [3,N,oh,ow] 连续内存：跨步视图会让 numpy 退化成逐元素慢路径
        arr = np.ascontiguousarray(arr.transpose(1, 0, 2, 3))
        c1, c2, var = arr[0], arr[1], arr[2]
        var -= c2 * c2 * inv_n
        np.maximum(var, 0.0, out=var)
        np.sqrt(var, out=var)
        var *= sp.tn
        sc = np.divide(c1, var, out=np.zeros_like(c1), where=var > 0)
        np.maximum(best[cat], sc.max(axis=(1, 2)), out=best[cat])

    return best


# ===============================================================
#  后端可用性与调度
# ===============================================================

def available_backends(profile=None) -> list[str]:
    torch_st = (profile.torch if profile is not None else None) or gpu_caps.torch_status()
    out: list[str] = []
    if torch_st.get("cuda_available"):
        out.append(BACKEND_TORCH_CUDA)
    if torch_st.get("torch_dml"):
        out.append(BACKEND_TORCH_DML)
    if gpu_caps.opencl_available():
        out.append(BACKEND_OPENCL)
    out += [BACKEND_CPU_BATCH, BACKEND_CV2_DIRECT, BACKEND_CPU_POOL]
    return [b for b in BACKEND_PREFERENCE if b in out]


def backend_label(backend: str) -> str:
    return BACKEND_LABEL.get(backend, backend)


def resolve_backend(requested: str | None, profile=None) -> str:
    """auto/空 → 偏好顺序里第一个可用的；选了不可用的则回退并留事件。"""
    avail = available_backends(profile)
    req = (requested or "auto").strip()
    if req in ("", "auto"):
        return avail[0] if avail else BACKEND_CV2_DIRECT
    if req in avail:
        return req
    fallback = avail[0] if avail else BACKEND_CV2_DIRECT
    app_core.warn(f"匹配后端 {backend_label(req)} 不可用，已回退到 {backend_label(fallback)}",
                  "matcher")
    return fallback


def scores_batch(gray_batch: np.ndarray, compiled: CompiledTemplates, backend: str,
                 configs: dict | None = None, proc_res: tuple | None = None,
                 n_threads: int = 4, device: str = "cuda") -> dict:
    if backend in (BACKEND_TORCH_CUDA, BACKEND_TORCH_DML):
        return _scores_torch(gray_batch, compiled, device=device)
    if backend == BACKEND_OPENCL:
        return _scores_cv2_batched(gray_batch, compiled, use_umat=True)
    if backend == BACKEND_CPU_BATCH:
        return _scores_cv2_batched(gray_batch, compiled, use_umat=False)
    if backend == BACKEND_CV2_DIRECT:
        return _scores_legacy(gray_batch, configs or {}, proc_res or compiled.proc_res)
    raise ValueError(f"unsupported backend for scores_batch: {backend}")


# ===============================================================
#  Matcher
# ===============================================================

class Matcher:
    """模板 + 后端 + 静止帧跳过 + 统计。

    classify() 输入一批灰度帧，输出 np.int8 状态数组，
    语义与逐帧 classify_gray 完全一致。
    """

    def __init__(self, compiled: CompiledTemplates, configs: dict, thresholds: dict,
                 proc_res: tuple, backend: str = BACKEND_CV2_DIRECT,
                 skip_identical: bool = True, n_threads: int = 4):
        self.compiled = compiled
        self.configs = configs
        self.thresholds = thresholds
        self.proc_res = tuple(proc_res)
        self.backend = backend
        self.skip_identical = bool(skip_identical)
        self.n_threads = int(n_threads)

        self._prev_gray: np.ndarray | None = None
        self._prev_state: int = FRAME_TYPE_NORMAL
        self.stats = {"frames": 0, "matched": 0, "skipped": 0, "match_seconds": 0.0}

    def classify_one(self, gray: np.ndarray) -> int:
        return int(self.classify(gray[None, ...])[0])

    def classify(self, gray_batch: np.ndarray) -> np.ndarray:
        n = int(gray_batch.shape[0])
        if n == 0:
            return np.zeros(0, np.int8)

        same = np.zeros(n, bool)
        if self.skip_identical and self._prev_gray is not None:
            prev = self._prev_gray
            same[0] = (gray_batch[0].shape == prev.shape) and bool(
                np.array_equal(gray_batch[0], prev))
            if n > 1:
                same[1:] = np.all(gray_batch[1:] == gray_batch[:-1], axis=(1, 2))

        idx = np.flatnonzero(~same)
        out = np.empty(n, np.int8)
        matched = int(idx.size)
        t0 = time.perf_counter()
        if matched:
            sub = np.ascontiguousarray(gray_batch[idx])
            if self.backend == BACKEND_CPU_POOL:
                out[idx] = self._classify_pool(sub)
            else:
                sc = scores_batch(sub, self.compiled, self.backend,
                                  configs=self.configs, proc_res=self.proc_res,
                                  n_threads=self.n_threads)
                out[idx] = classify_from_scores(sc, self.compiled.has, self.thresholds)
        dt = time.perf_counter() - t0

        if self.skip_identical:
            last = self._prev_state
            for i in range(n):
                if same[i]:
                    out[i] = last
                else:
                    last = int(out[i])

        self._prev_gray = np.ascontiguousarray(gray_batch[-1])
        self._prev_state = int(out[-1])
        self.stats["frames"] += n
        self.stats["matched"] += matched
        self.stats["skipped"] += (n - matched)
        self.stats["match_seconds"] += dt
        return out

    def _classify_pool(self, gray_batch: np.ndarray) -> np.ndarray:
        import analyzer
        n_workers = max(1, min(self.n_threads, multiprocessing.cpu_count()))
        chunk = max(1, len(gray_batch) // (n_workers * 2))
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, initializer=analyzer._worker_init,
                initargs=(self.configs, self.thresholds, self.proc_res)) as ex:
            res = list(ex.map(analyzer._worker_classify_gray, gray_batch, chunksize=chunk))
        return np.asarray(res, dtype=np.int8)

    @property
    def ms_per_frame(self) -> float:
        m = self.stats["matched"]
        return (self.stats["match_seconds"] * 1000.0 / m) if m else 0.0

    def skip_ratio(self) -> float:
        f = self.stats["frames"]
        return (self.stats["skipped"] / f) if f else 0.0

    def describe(self) -> str:
        return (f"匹配 {backend_label(self.backend)} · {self.ms_per_frame:.3f} ms/帧 · "
                f"静止帧跳过 {self.skip_ratio() * 100:.0f}%")


# ===============================================================
#  自动测速与一致性对照
# ===============================================================

@dataclass
class BenchResult:
    backend: str
    ok: bool
    ms_per_frame: float = 0.0
    max_delta: float = 0.0
    flips: int = 0
    frames: int = 0
    near_threshold: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {"backend": self.backend, "ok": self.ok,
                "ms_per_frame": round(self.ms_per_frame, 5),
                "max_delta": round(self.max_delta, 8),
                "flips": int(self.flips), "frames": int(self.frames),
                "near_threshold": int(self.near_threshold), "error": self.error}


def sample_gray_frames(video_path: str, proc_res: tuple, count: int = 48,
                       start_frame: int = 0, decode_backend=None,
                       ffmpeg_path: str | None = None) -> np.ndarray:
    """抽 count 帧灰度图（优先硬件解码，失败回退 OpenCV）。"""
    import pipeline
    pw, ph = int(proc_res[0]), int(proc_res[1])
    frames: list[np.ndarray] = []
    if decode_backend:
        try:
            for gray in pipeline.iter_gray_frames(
                    video_path, proc_res, decode_backend=decode_backend,
                    start_frame=start_frame, max_frames=count,
                    ffmpeg_path=ffmpeg_path):
                frames.append(gray)
                if len(frames) >= count:
                    break
            if frames:
                return np.stack(frames)
        except Exception as exc:
            app_core.warn(f"抽帧走 {decode_backend} 失败，回退 OpenCV: {exc}", "matcher")

    cap = cv2.VideoCapture(video_path)
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        while len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(
                cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return np.stack(frames) if frames else np.zeros((0, ph, pw), np.uint8)


def _flip_stats(ref_states: np.ndarray, got_states: np.ndarray, ref_scores: dict,
                thresholds: dict) -> tuple[int, int]:
    """返回 (不一致帧数, 其中贴近阈值 2e-3 的帧数)。"""
    diff = ref_states != got_states
    flips = int(diff.sum())
    if flips == 0:
        return 0, 0
    near = np.zeros(len(ref_states), bool)
    for cat in _CATEGORIES:
        if cat in ref_scores and cat in thresholds:
            near |= np.abs(ref_scores[cat] - thresholds[cat]) <= 2e-3
    return flips, int((diff & near).sum())


def autotune(compiled: CompiledTemplates, configs: dict, thresholds: dict,
             proc_res: tuple, sample: np.ndarray, profile=None,
             candidates: list[str] | None = None) -> tuple[list[BenchResult], str | None]:
    """实测各后端的速度与一致性，返回 (结果, 选中的后端)。

    一致性门槛：分数 max|Δ| ≤ 1e-3，且分类不一致的帧必须全部落在
    「距离阈值 2e-3 以内」的模糊带上；不达标的后端即使最快也不选用。
    """
    if sample is None or len(sample) == 0:
        return [], None

    n = len(sample)
    ref_scores = _scores_legacy(sample, configs, proc_res)
    # 基准本身必须可信：cv2 的带掩码 TM_CCOEFF_NORMED 在个别环境下会给出非有限值，
    # get_best_score 会因此跳过所有模板（分数全 -1）→ 漏判暂停；此时它就是错的，
    # 不能拿它去否定其它后端（否则正确后端会被全部判掉，最后选中一个坏基准）。
    ref_np = scores_batch(sample, compiled, BACKEND_CPU_BATCH, configs, proc_res)
    ref_deltas = [float(np.max(np.abs(ref_scores[c] - ref_np[c]))) for c in _CATEGORIES
                  if ref_scores.get(c) is not None and ref_np.get(c) is not None]
    cv2_ref_delta = max(ref_deltas) if ref_deltas else 0.0
    cv2_trusted = cv2_ref_delta <= 1e-3
    if not cv2_trusted:
        app_core.warn(
            f"cv2 直算与内置 NCC 基准不一致（max|Δ|={cv2_ref_delta:.2e}），"
            f"改用内置 NCC 作基准，并排除 cv2 直算（否则会漏判暂停）", "matcher")
        ref_scores = ref_np
    ref_states = classify_from_scores(ref_scores, compiled.has, thresholds)

    pool = candidates if candidates is not None else available_backends(profile)
    results: list[BenchResult] = []
    for backend in pool:
        r = BenchResult(backend=backend, ok=False, frames=n)
        if backend == BACKEND_CPU_POOL:
            r.error = "不参与自动测速（兼容保留）"
            results.append(r)
            continue
        if not cv2_trusted and backend == BACKEND_CV2_DIRECT:
            r.error = "cv2 实现与本程序内置 NCC 不一致，已排除（避免漏判暂停）"
            results.append(r)
            continue
        try:
            # 预热必须用与计时相同的批大小，否则会把 cuDNN 选核的冷启动算进去
            _ = scores_batch(sample, compiled, backend, configs, proc_res)
            best_dt = None
            got_scores = None
            for _ in range(2):
                t0 = time.perf_counter()
                got_scores = scores_batch(sample, compiled, backend, configs, proc_res)
                dt = time.perf_counter() - t0
                best_dt = dt if best_dt is None else min(best_dt, dt)
            r.ms_per_frame = (best_dt or 0.0) * 1000.0 / n

            deltas = [float(np.max(np.abs(ref_scores[c] - got_scores[c])))
                      for c in _CATEGORIES
                      if ref_scores.get(c) is not None and got_scores.get(c) is not None]
            r.max_delta = max(deltas) if deltas else 0.0

            got_states = classify_from_scores(got_scores, compiled.has, thresholds)
            r.flips, r.near_threshold = _flip_stats(ref_states, got_states, ref_scores, thresholds)

            if r.max_delta > 1e-3:
                r.error = f"分数偏差过大 (max|Δ|={r.max_delta:.2e} > 1e-3)"
            elif r.flips != r.near_threshold or r.flips > max(1, int(0.001 * n)):
                r.error = f"分类不一致 {r.flips} 帧（其中 {r.near_threshold} 帧贴近阈值）"
            else:
                r.ok = True
            app_core.debug(
                f"测速 {backend_label(backend)}：{r.ms_per_frame:.4f} ms/帧 · "
                f"max|Δ|={r.max_delta:.2e} · 不一致 {r.flips}/{n}"
                + (f" · {r.error}" if r.error else " · 通过"), "matcher")
        except Exception as exc:
            r.error = f"{type(exc).__name__}: {exc}"
            app_core.debug(f"测速 {backend} 异常：{r.error}", "matcher")
        results.append(r)

    passing = [r for r in results if r.ok]
    if passing:
        chosen = min(passing, key=lambda r: r.ms_per_frame).backend
    else:
        app_core.warn("没有任何匹配后端通过一致性门槛，使用 cv2 直算", "matcher")
        chosen = BACKEND_CV2_DIRECT
    return results, chosen


def bench_cache_key(compiled: CompiledTemplates, profile, proc_res: tuple) -> str:
    prof_sig = gpu_caps.profile_signature(profile) if profile is not None else ""
    return (f"v{AUTOTUNE_VERSION}|{compiled.signature()}|{prof_sig}|"
            f"{int(proc_res[0])}x{int(proc_res[1])}")


def cached_choice(key: str) -> dict | None:
    entry = app_core.load_bench().get(key)
    return entry if isinstance(entry, dict) else None


def store_choice(key: str, chosen: str, results: list[BenchResult]) -> None:
    bench = app_core.load_bench()
    bench[key] = {"chosen": chosen, "when": time.time(),
                  "results": [r.to_dict() for r in results]}
    if len(bench) > 20:
        for k in sorted(bench, key=lambda x: bench[x].get("when", 0))[:-20]:
            bench.pop(k, None)
    app_core.save_bench(bench)


def format_results(results: list[BenchResult], chosen: str | None = None) -> str:
    if not results:
        return "尚未测速"
    rows = []
    for r in sorted(results, key=lambda x: (not x.ok, x.ms_per_frame)):
        mark = "✔" if r.ok else "✘"
        if r.ok:
            rows.append(f"{mark} {backend_label(r.backend)} {r.ms_per_frame:.3f} ms/帧"
                        f"（Δ{r.max_delta:.1e}）")
        else:
            rows.append(f"{mark} {backend_label(r.backend)}：{r.error or '未通过'}")
    if chosen:
        rows.append(f"→ 已选：{backend_label(chosen)}")
    return " ｜ ".join(rows)


# exporter.py —— 导出：阶梯式回退 + 编码器 profile + 关键帧无损直通 + 真实进度 + 可取消
#
# 阶梯（逐级尝试，失败自动降级）：
#   1) 关键帧无损直通：段起点都落在关键帧 → 逐段 -c copy + concat，零重编码；
#   2) 分块并行重编码：段数够多时按块并行 + concat；
#   3) 单遍滤镜编码（硬件或 libx264）；
#   4) 逐帧管道写入器（硬件 → CPU）；
#   5) imageio / cv2.VideoWriter。



import bisect
import concurrent.futures
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import app_core
import gpu_caps

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# 单次 ffmpeg 滤镜图里最多放多少个区间：区间数一多，滤镜图初始化会变得极慢
# 而且期间 ffmpeg 不输出 -progress，界面上看就是「卡住」
_MAX_RANGES_PER_CHUNK = 200
_MAX_GOP_FRAMES = 900          # 关键帧间隔上限（15s@60fps），超过就认为索引不可信
_KEYFRAME_CACHE: dict[str, tuple[float, list[int]]] = {}
_KEYFRAME_LOCKS: dict[str, threading.Lock] = {}
_AUDIO_CACHE: dict[str, bool] = {}
_CACHE_LOCK = threading.Lock()


class ExportCancelled(Exception):
    """用户取消导出。"""


@dataclass
class _FFmpegPipe:
    process: subprocess.Popen
    stderr_file: Any

    @property
    def stdin(self):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin 不可用")
        return self.process.stdin


# ===============================================================
#  基础工具
# ===============================================================

def _kept_frame_ranges(to_del: np.ndarray) -> list[tuple[int, int]]:
    """删除掩码 → 左闭右开的保留帧区间。"""
    ranges = []
    i = 0
    total = len(to_del)
    while i < total:
        if to_del[i]:
            i += 1
            continue
        start = i
        while i < total and not to_del[i]:
            i += 1
        ranges.append((start, i))
    return ranges


def _ffprobe_path(ffmpeg: str | None) -> str | None:
    if ffmpeg:
        cand = os.path.join(os.path.dirname(ffmpeg),
                            "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffprobe")


def _has_audio_stream(video_path: str, ffmpeg_path: str | None = None) -> bool:
    try:
        key = (f"{os.path.normcase(os.path.abspath(video_path))}|"
               f"{os.path.getmtime(video_path):.0f}")
    except OSError:
        key = os.path.normcase(video_path)
    with _CACHE_LOCK:
        if key in _AUDIO_CACHE:
            return _AUDIO_CACHE[key]

    result = False
    probe = _ffprobe_path(None)
    if probe:
        try:
            res = subprocess.run(
                [probe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
                check=False, capture_output=True, text=True, timeout=20,
                creationflags=_NO_WINDOW)
            result = bool((res.stdout or "").strip())
        except Exception:
            result = False
    if not result:
        try:
            ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
            res = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video_path,
                 "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-"],
                check=False, capture_output=True, timeout=30,
                creationflags=_NO_WINDOW)
            result = res.returncode == 0
        except Exception:
            result = False
    with _CACHE_LOCK:
        _AUDIO_CACHE[key] = result
    return result


def _video_frame_count(video_path: str, ffmpeg_path: str | None = None) -> int:
    """精确帧数（全文件解码计数，长片很贵；只在时长校验存疑时调用）。"""
    probe = _ffprobe_path(gpu_caps.resolve_ffmpeg_path(ffmpeg_path) if ffmpeg_path else None)
    if not probe:
        return -1
    try:
        res = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", video_path],
            check=False, capture_output=True, text=True, timeout=900,
            creationflags=_NO_WINDOW)
        text = (res.stdout or "").strip().splitlines()
        return int(text[0]) if text and text[0].strip().isdigit() else -1
    except Exception:
        return -1


def media_duration(video_path: str, ffmpeg_path: str | None = None) -> float:
    """容器时长（O(1)）。"""
    probe = _ffprobe_path(gpu_caps.resolve_ffmpeg_path(ffmpeg_path) if ffmpeg_path else None)
    if not probe:
        return -1.0
    try:
        res = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            check=False, capture_output=True, text=True, timeout=60,
            creationflags=_NO_WINDOW)
        line = (res.stdout or "").strip().splitlines()
        return float(line[0]) if line else -1.0
    except Exception:
        return -1.0


def _duration_ok(path: str, expected_frames: int, fps: float,
                 tolerance_frames: float = 4.0) -> tuple[bool, str]:
    if fps <= 0 or expected_frames <= 0:
        return True, ""
    got = media_duration(path)
    if got <= 0:
        return True, ""
    want = expected_frames / fps
    tol = max(0.5, tolerance_frames / fps)
    if abs(got - want) > tol:
        return False, f"时长校验不符（{got:.3f}s vs 期望 {want:.3f}s，容差 {tol:.3f}s）"
    return True, ""


def _count_tolerance(n_ranges: int) -> int:
    """帧数校验容差。

    区间多时也不能放宽到几十帧，否则「只出了 8 帧」这种截断成品会被当成
    「存疑但正确」放行。
    """
    return max(2, min(16, n_ranges))


def _mp4_keyframes(path: str) -> list[int] | None:
    """从 MP4/MOV 容器的 stss 索引直接读关键帧帧号（0 基）。

    只读 moov 里的小索引表，不开解码器、不扫全片，长片也是毫秒级；
    ffprobe 的 -skip_frame nokey 需要读完整个文件的包，既慢又可能中途出错
    只拿到前半段（实测 56 分钟素材只扫到前 17 分钟，导致靠后的分块拿不到
    锚点、只能从第 0 帧开始解码）。读不到就返回 None，调用方回退 ffprobe。
    """
    if not path.lower().endswith(('.mp4', '.mov', '.m4v', '.m4a')):
        return None
    import struct
    containers = {b'moov', b'trak', b'mdia', b'minf', b'stbl', b'edts', b'dinf',
                  b'udta', b'meta', b'ilst'}

    def find(fh, start: int, end: int, want: bytes, depth: int = 0) -> list:
        out = []
        fh.seek(start)
        while fh.tell() + 8 <= end:
            pos = fh.tell()
            head = fh.read(8)
            if len(head) < 8:
                break
            size, typ = struct.unpack('>I4s', head)
            if size == 1:
                size = struct.unpack('>Q', fh.read(8))[0]
            if size == 0:
                size = end - pos
            if size < 8 or pos + size > end:
                break
            if typ == want:
                out.append((pos, size))
            if typ in containers and depth < 8:
                inner = pos + (12 if typ == b'meta' else 8)
                out.extend(find(fh, inner, pos + size, want, depth + 1))
            fh.seek(pos + size)
        return out

    try:
        with open(path, 'rb') as fh:
            total = os.fstat(fh.fileno()).st_size
            moovs = find(fh, 0, total, b'moov')
            for moov_at, moov_size in moovs:
                for trak_at, trak_size in find(fh, moov_at + 8, moov_at + moov_size,
                                               b'trak'):
                    hdlrs = find(fh, trak_at + 8, trak_at + trak_size, b'hdlr')
                    is_video = False
                    for hp, _hs in hdlrs:
                        fh.seek(hp + 16)
                        if fh.read(4) == b'vide':
                            is_video = True
                            break
                    if not is_video:
                        continue
                    stss = find(fh, trak_at + 8, trak_at + trak_size, b'stss')
                    if not stss:
                        continue
                    fh.seek(stss[0][0] + 12)          # box: size+type+version/flags+count
                    data = fh.read(4)
                    if len(data) < 4:
                        continue
                    count = struct.unpack('>I', data)[0]
                    if count <= 0 or count > 5_000_000:
                        continue
                    nums = struct.unpack('>%dI' % count, fh.read(4 * count))
                    return [int(x) - 1 for x in nums]
    except Exception as exc:
        app_core.warn(f"读取 MP4 关键帧索引失败，回退 ffprobe: {exc}", "exporter")
    return None


def _video_frame_total(path: str) -> int:
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return total
    except Exception:
        return 0


def _index_partial(frames, video_path: str, fps: float = 0.0,
                   ratio: float = 0.95) -> bool:
    """索引是否只覆盖了前一段（扫全片中途出错时会只拿到前半段）。

    帧数拿不到时用「时长×fps」兜底，避免因为取不到总帧数就把截断索引当成完整的。
    """
    if not frames:
        return True
    total = _video_frame_total(video_path)
    if not total and fps > 0:
        dur = media_duration(video_path)
        total = int(dur * fps) if dur > 0 else 0
    return bool(total) and int(frames[-1]) < total * ratio


def keyframe_frames(video_path: str, fps: float,
                    ffmpeg_path: str | None = None,
                    status_cb=None) -> list[int]:
    """关键帧帧号列表（时间戳×fps 折算，CFR 下准确）。

    扫描要读完整个文件的包索引，长片很慢，因此结果按「路径+mtime+大小」
    同时缓存在内存和磁盘（重启程序后不必重扫）。
    """
    try:
        stat = os.stat(video_path)
        key = (f"{os.path.normcase(os.path.abspath(video_path))}|"
               f"{stat.st_mtime:.0f}|{stat.st_size}")
    except OSError:
        key = os.path.normcase(video_path)
    with _CACHE_LOCK:
        cached = _KEYFRAME_CACHE.get(key)
    if cached is not None:
        return list(cached[1])

    disk = app_core.load_keyframes(key)
    if disk is not None and not _index_partial(disk, video_path, fps):
        with _CACHE_LOCK:
            _KEYFRAME_CACHE[key] = (time.time(), list(disk))
        return list(disk)

    # 单飞：并发导出（分段导出有多个 worker）时只让一个线程去扫
    with _CACHE_LOCK:
        scan_lock = _KEYFRAME_LOCKS.setdefault(key, threading.Lock())
    with scan_lock:
        with _CACHE_LOCK:
            cached = _KEYFRAME_CACHE.get(key)
        if cached is not None:
            return list(cached[1])
        disk = app_core.load_keyframes(key)
        if disk is not None and not _index_partial(disk, video_path, fps):
            with _CACHE_LOCK:
                _KEYFRAME_CACHE[key] = (time.time(), list(disk))
            return list(disk)

        if status_cb:
            status_cb("扫描关键帧（长片较慢，只做一次并会记住）…")
        frames: list[int] = _mp4_keyframes(video_path) or []
        probe = None
        if not frames:
            probe = _ffprobe_path(
                gpu_caps.resolve_ffmpeg_path(ffmpeg_path) if ffmpeg_path else None)
        if probe and fps > 0:
            try:
                res = subprocess.run(
                    [probe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                     "-show_entries", "frame=best_effort_timestamp_time",
                     "-of", "csv=p=0", video_path],
                    check=False, capture_output=True, text=True, timeout=3600,
                    creationflags=_NO_WINDOW)
                for line in (res.stdout or "").splitlines():
                    line = line.strip().rstrip(",")
                    if not line:
                        continue
                    try:
                        frames.append(int(round(float(line) * fps)))
                    except ValueError:
                        continue
            except Exception as exc:
                app_core.warn(f"关键帧探测失败，无损直通不可用: {exc}", "exporter")
                frames = []
        if _index_partial(frames, video_path):
            app_core.warn(
                f"关键帧索引只覆盖到第 {frames[-1] if frames else 0} 帧"
                f"（全片约 {_video_frame_total(video_path)} 帧），已丢弃，"
                f"靠后的片段将无法定位锚点", "exporter")
            frames = []
        with _CACHE_LOCK:
            _KEYFRAME_CACHE[key] = (time.time(), list(frames))
        if frames:
            app_core.save_keyframes(key, frames)
        app_core.debug(
            f"关键帧索引 {len(frames)} 个"
            + ("（MP4 stss 直读）" if probe is None else "（ffprobe 扫描）")
            + (f"，覆盖 {frames[0]}~{frames[-1]}" if frames else "，不可用"), "exporter")
        return frames


def _anchor_for(index: int, fps: float, keyframes) -> int:
    """返回 ≤ index 的最近关键帧帧号，用于把绝对帧号换成相对帧号。

    以关键帧为锚点做输入 seek 才是精确的：seek 到某个关键帧的时间戳后，
    解码出的第一帧就是该关键帧本身。锚点缺失或跨度异常时返回 0（从头解码）。
    """
    if index <= 0 or not keyframes:
        return 0
    ks = sorted(keyframes) if isinstance(keyframes, (set, frozenset)) else keyframes
    i = bisect.bisect_right(ks, index) - 1
    if i < 0:
        return 0
    k = int(ks[i])
    if k < 0 or index - k > _MAX_GOP_FRAMES:
        return 0
    return k


# ===============================================================
#  带进度/取消的 ffmpeg 调用
# ===============================================================

def _run_ffmpeg_progress(cmd: list[str], total_frames: int,
                         progress_cb=None, cancel_event=None,
                         ratio_base: float = 0.0, ratio_span: float = 1.0,
                         done_offset: int = 0,
                         timeout: float = 7200.0) -> tuple[bool, str]:
    """跑一个 ffmpeg，解析 `-progress pipe:1`，支持取消。

    输出用独立线程读进队列：ffmpeg 在滤镜图初始化阶段完全不输出，
    直接阻塞读 stdout 会让「取消」和「超时」都失效。
    """
    stderr_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=stderr_file, text=True, encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW)
    except Exception as exc:
        stderr_file.close()
        return False, f"{type(exc).__name__}: {exc}"

    lines: queue.Queue = queue.Queue()

    def _pump():
        try:
            if proc.stdout is not None:
                for raw in proc.stdout:
                    lines.put(raw)
        except Exception:
            pass
        finally:
            lines.put(None)

    threading.Thread(target=_pump, daemon=True, name="ffmpeg-progress").start()

    deadline = time.perf_counter() + timeout
    cancelled = False
    rc = -1
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            try:
                raw = lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None and lines.empty():
                    break
                if time.perf_counter() > deadline:
                    proc.kill()
                    return False, f"ffmpeg 超时（{timeout:.0f}s）"
                continue
            if raw is None:
                break
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if progress_cb and key == "frame" and total_frames > 0:
                try:
                    done = int(value)
                except ValueError:
                    continue
                progress_cb(ratio_base + min(1.0, done / total_frames) * ratio_span,
                            done_offset + done)
        if cancelled:
            proc.kill()
            proc.wait(timeout=15)
            raise ExportCancelled("已取消导出")
        rc = proc.wait(timeout=30)
    except ExportCancelled:
        raise
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass

    try:
        stderr_file.seek(0)
        err = stderr_file.read().decode("utf-8", errors="replace").strip()
    except Exception:
        err = ""
    finally:
        stderr_file.close()
    if rc != 0:
        return False, err[-800:] if err else f"ffmpeg 退出码 {rc}"
    return True, err


def _hwaccel_input_args(profile: gpu_caps.GpuProfile | None) -> list[str]:
    """导出用的硬解输入参数（只取 -hwaccel，不指定 hwaccel_output_format，
    让 trim/concat 在内存帧上跑）。

    优先用实测选中的解码变体；若 profile 未做解码实测（例如导出时为了省时间
    用 probe_decoders=False 拿的缓存），就按 hwaccels 里可用的接口直接选一个，
    否则导出会悄悄退化成软件解码。
    """
    if profile is None:
        return []
    if profile.decode_variant:
        for v in gpu_caps.build_decode_variants(profile.hwaccels):
            if v.key == profile.decode_variant and v.hwaccel:
                return v.hwaccel[:2]
    have = {h.lower() for h in (profile.hwaccels or [])}
    for name in ("cuda", "qsv", "d3d11va", "dxva2"):
        if name in have:
            return ["-hwaccel", name]
    return []


# ===============================================================
#  1) 关键帧无损直通
# ===============================================================

def _copy_one_range(video_path: str, out_path: str, start: int, end: int,
                    fps: float, ffmpeg: str, timeout: float = 600.0) -> bool:
    """无损复制一段。

    用 -frames:v 而不是 -t 时长：-t 会因时间戳取整多带 1 帧。
    """
    n = max(0, int(end) - int(start))
    if n <= 0:
        return False
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{start / fps:.6f}", "-i", video_path,
           "-frames:v", str(n),
           "-c", "copy", "-avoid_negative_ts", "make_zero", "-an", out_path]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout,
                             creationflags=_NO_WINDOW)
    except Exception:
        return False
    return res.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _concat_parts(parts: list[str], out_path: str, ffmpeg: str,
                  tmpdir: str, copy_streams: bool = True) -> bool:
    list_file = os.path.join(tmpdir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "concat", "-safe", "0", "-i", list_file]
    cmd += ["-c", "copy"] if copy_streams else []
    cmd.append(out_path)
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, timeout=3600,
                             creationflags=_NO_WINDOW)
    except Exception:
        return False
    return res.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _try_copy_export(video_path: str, output_path: str,
                     ranges: list[tuple[int, int]], fps: float,
                     profile, ffmpeg: str, progress_cb=None,
                     cancel_event=None, workers: int = 4,
                     keyframes=None, status_cb=None, include_audio: bool = True) -> bool:
    expected = sum(e - s for s, e in ranges)
    if expected <= 0:
        return False
    keys = set(keyframes) if keyframes else set(keyframe_frames(
        video_path, fps, ffmpeg_path=ffmpeg, status_cb=status_cb))
    if not keys:
        return False

    # 关键帧探测有 ±1 帧误差
    def _is_key(idx: int) -> bool:
        return idx in keys or (idx - 1) in keys or (idx + 1) in keys

    bad = [s for s, _ in ranges if not _is_key(s)]
    if bad:
        app_core.info(
            f"无损直通未命中：{len(bad)}/{len(ranges)} 段起点不是关键帧，改用重编码",
            "exporter")
        return False

    with tempfile.TemporaryDirectory(prefix="aae_copy_") as tmpdir:
        parts = [os.path.join(tmpdir, f"p{i:05d}.mp4") for i in range(len(ranges))]
        done = 0
        lock = threading.Lock()

        def _one(i: int) -> bool:
            nonlocal done
            if cancel_event is not None and cancel_event.is_set():
                return False
            s, e = ranges[i]
            ok = _copy_one_range(video_path, parts[i], s, e, fps, ffmpeg)
            with lock:
                done += 1
                if progress_cb:
                    progress_cb(0.02 + 0.86 * done / len(ranges), 0)
            return ok

        max_w = max(1, min(int(workers), 8, len(ranges)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as ex:
            results = list(ex.map(_one, range(len(ranges))))
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled("已取消导出")
        if not all(results):
            app_core.warn("逐段无损复制失败，改用重编码路径", "exporter")
            return False

        video_only = os.path.join(tmpdir, "video_only.mp4")
        if not _concat_parts(parts, video_only, ffmpeg, tmpdir):
            app_core.warn("无损直通 concat 失败，改用重编码路径", "exporter")
            return False

        # 先做 O(1) 的时长校验，只有存疑时才全解码计数
        ok_dur, why = _duration_ok(video_only, expected, fps)
        if not ok_dur:
            got = _video_frame_count(video_only)
            if got > 0 and abs(got - expected) <= _count_tolerance(len(ranges)):
                app_core.info(f"时长校验存疑但帧数正确（{got} vs {expected}），继续", "exporter")
            else:
                app_core.warn(f"无损直通{why}，改用重编码路径", "exporter")
                return False

        if include_audio and _has_audio_stream(video_path, ffmpeg_path=ffmpeg):
            _mux_audio_for_ranges(video_path, video_only, output_path, ranges, fps,
                                  ffmpeg_path=ffmpeg, cancel_event=cancel_event,
                                  status_cb=status_cb, keyframes=keys)
        else:
            shutil.move(video_only, output_path)
    if progress_cb:
        progress_cb(1.0, expected)
    return True


# ===============================================================
#  2) 分块并行重编码
# ===============================================================

def _write_filter_script(path: str, ranges: list[tuple[int, int]], fps: float,
                         has_audio: bool, audio_idx: int = 0,
                         audio_sink: bool = False) -> None:
    lines = []
    labels = []
    for idx, (start, end) in enumerate(ranges):
        lines.append(f"[0:v]trim=start_frame={start}:end_frame={end},"
                     f"setpts=PTS-STARTPTS[v{idx}]")
        labels.append(f"[v{idx}]")
        if has_audio:
            lines.append(f"[{audio_idx}:a]atrim=start={start / fps:.9f}:end={end / fps:.9f},"
                         f"asetpts=PTS-STARTPTS[a{idx}]")
            labels.append(f"[a{idx}]")
    lines.append("".join(labels) + f"concat=n={len(ranges)}:v=1:a={1 if has_audio else 0}"
                 + ("[outv][outa]" if has_audio else "[outv]"))
    if has_audio and audio_sink:
        # 不要音频也要留在滤镜图里（见 _build_filter_cmd 注释），用 anullsink 吃掉，
        # 否则 ffmpeg 会报 "Error binding filtergraph inputs/outputs"
        lines.append("[outa]anullsink")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";\n".join(lines))


def _build_filter_cmd(ffmpeg: str, video_path: str, out_path: str, script_path: str,
                      base: int, rel: list[tuple[int, int]], fps: float, quality: int,
                      use_gpu: bool, gpu_encoder: str, preset: str | None, profile,
                      graph_audio: bool, map_audio: bool, silent_audio: bool) -> list[str]:
    """组装 trim+concat 滤镜命令。

    graph_audio 必须为 True：concat 滤镜在 a=0 且分支较多时只输出极少数帧
    （实测 100 个单帧分支只出 2 帧）。源没有音频时用 anullsrc 补一路静音，
    音频是否写进成品由 map_audio 决定。
    """
    audio_idx = 1 if silent_audio else 0
    _write_filter_script(script_path, rel, fps, graph_audio, audio_idx,
                         audio_sink=not map_audio)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           *_hwaccel_input_args(profile)]
    if base > 0:
        cmd += ["-ss", f"{base / fps:.6f}"]
    cmd += ["-i", video_path]
    if silent_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += ["-filter_complex_script", script_path, "-map", "[outv]"]
    if map_audio:
        cmd += ["-map", "[outa]"]
    cmd += _encode_args(quality, use_gpu, gpu_encoder, ffmpeg, preset)
    cmd += ["-c:a", "aac"] if map_audio else ["-an"]
    cmd += ["-progress", "pipe:1", out_path]
    return cmd


def _encode_args(quality: int, use_gpu: bool, gpu_encoder: str,
                 ffmpeg: str, preset: str | None) -> list[str]:
    return gpu_caps.video_encoder_args(quality, use_gpu, gpu_encoder,
                                       ffmpeg_path=ffmpeg, preset=preset)


def _run_filter_export(ffmpeg: str, video_path: str, output_path: str, base: int,
                       rel_ranges: list[tuple[int, int]], fps: float, quality: int,
                       use_gpu: bool, gpu_encoder: str, has_audio: bool,
                       progress_cb=None, preset: str | None = None,
                       cancel_event=None, profile=None,
                       ratio_base: float = 0.02, ratio_span: float = 0.96) -> bool:
    """跑一次 trim+concat 滤镜导出；base>0 时先 seek 到该帧再解。

    has_audio 只表示「音频要不要写进成品」，滤镜图始终带音频分支。
    """
    total_frames = sum(e - s for s, e in rel_ranges)
    src_audio = _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filter_file = os.path.join(tmpdir, "filter.txt")
        cmd = _build_filter_cmd(ffmpeg, video_path, output_path, filter_file, base,
                                rel_ranges, fps, quality, use_gpu, gpu_encoder,
                                preset, profile, graph_audio=True,
                                map_audio=bool(has_audio) and src_audio,
                                silent_audio=not src_audio)

        ok, err = _run_ffmpeg_progress(
            cmd, total_frames, progress_cb=progress_cb, cancel_event=cancel_event,
            ratio_base=ratio_base, ratio_span=ratio_span, timeout=14400.0)
    if not ok:
        enc = gpu_caps.resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg) if use_gpu else None
        app_core.warn(f"滤镜导出失败（编码器 {enc or 'libx264'}）：{err[:300]}", "exporter")
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 0


def _export_ranges_with_ffmpeg_filters(
        video_path: str, output_path: str, ranges: list[tuple[int, int]],
        fps: float, quality: int, use_gpu: bool, gpu_encoder: str,
        include_audio: bool, progress_cb=None,
        ffmpeg_path: str | None = None, preset: str | None = None,
        cancel_event=None, profile=None,
        ratio_base: float = 0.02, ratio_span: float = 0.96,
        keyframes=None, status_cb=None) -> bool:
    """滤镜 trim+concat 导出（左闭右开区间）。

    区间起点靠后时以最近关键帧为锚点做输入 seek，只解码自己那一段；
    不对齐就从头解。锚点跑到偏差时会自动退回从头解的版本。
    """
    if not ranges:
        return False
    try:
        ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        return False

    has_audio = include_audio and _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    total_frames = sum(e - s for s, e in ranges)
    anchor = _anchor_for(ranges[0][0], fps, keyframes)

    for base in ([anchor, 0] if anchor else [0]):
        if status_cb:
            status_cb("滤镜编码中…" if base == 0
                      else f"滤镜编码中（从第 {base} 帧起解）…")
        rel = [(s - base, e - base) for s, e in ranges]
        if not _run_filter_export(ffmpeg, video_path, output_path, base, rel, fps,
                                 quality, use_gpu, gpu_encoder, has_audio,
                                 progress_cb, preset, cancel_event, profile,
                                 ratio_base, ratio_span):
            return False
        if base == 0:
            return True
        ok_dur, why = _duration_ok(output_path, total_frames, fps)
        if ok_dur:
            return True
        app_core.warn(f"定位导出{why}，改用从头解码重试", "exporter")
        try:
            os.remove(output_path)
        except OSError:
            pass
    return False


# ===============================================================
#  3) 音频混流（视频 copy，仅重编码音频）
# ===============================================================

def _mux_audio_for_ranges(video_path: str, video_only_path: str,
                          output_path: str, ranges: list[tuple[int, int]],
                          fps: float, ffmpeg_path: str | None = None,
                          cancel_event=None, status_cb=None,
                          keyframes=None) -> None:
    """把原片对应区间的音频剪出来，与 video_only_path 合成 output_path。

    区间多时必须分组：音频滤镜里每个 atrim 分支都会过一遍所有音频帧，
    几千个区间塞进一张图会慢到几十分钟（而且没有任何输出）。
    """
    if not _has_audio_stream(video_path, ffmpeg_path=ffmpeg_path):
        os.replace(video_only_path, output_path)
        return

    ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    n_groups = max(1, -(-len(ranges) // _MAX_RANGES_PER_CHUNK))
    per = -(-len(ranges) // n_groups)

    with tempfile.TemporaryDirectory() as tmpdir:
        parts: list[str] = []
        for gi in range(n_groups):
            group = ranges[gi * per:(gi + 1) * per]
            if not group:
                continue
            if cancel_event is not None and cancel_event.is_set():
                raise ExportCancelled("已取消导出")
            base = _anchor_for(group[0][0], fps, keyframes) if keyframes else 0
            rel = [(s - base, e - base) for s, e in group]
            script = os.path.join(tmpdir, f"a{gi:03d}.txt")
            lines, labels = [], []
            for i, (s, e) in enumerate(rel):
                lines.append(f"[0:a]atrim=start={s / fps:.9f}:end={e / fps:.9f},"
                             f"asetpts=PTS-STARTPTS[a{i}]")
                labels.append(f"[a{i}]")
            lines.append("".join(labels) + f"concat=n={len(rel)}:v=0:a=1[outa]")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(";\n".join(lines))

            part = os.path.join(tmpdir, f"a{gi:03d}.m4a")
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                   "-nostdin"]
            if base > 0:
                cmd += ["-ss", f"{base / fps:.6f}"]
            cmd += ["-i", video_path, "-filter_complex_script", script,
                    "-map", "[outa]", "-c:a", "aac", "-b:a", "192k", part]
            ok, err = _run_ffmpeg_progress(cmd, 0, cancel_event=cancel_event,
                                           timeout=7200.0)
            if not ok:
                raise RuntimeError(f"音频切片失败: {err[:300]}")
            parts.append(part)
            if status_cb:
                status_cb(f"混流音频 {gi + 1}/{n_groups}…")

        if len(parts) == 1:
            audio_all = parts[0]
        else:
            list_file = os.path.join(tmpdir, "alist.txt")
            with open(list_file, "w", encoding="utf-8") as fh:
                for p in parts:
                    fh.write("file '" + p.replace("'", "'\\''") + "'\n")
            audio_all = os.path.join(tmpdir, "audio.m4a")
            ok, err = _run_ffmpeg_progress(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                 "-nostdin", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c", "copy", audio_all],
                0, cancel_event=cancel_event, timeout=3600.0)
            if not ok:
                raise RuntimeError(f"音频拼接失败: {err[:300]}")

        # 不能用 -shortest：音频按 AAC 包粒度会比视频略短，会把尾部视频帧裁掉
        ok, err = _run_ffmpeg_progress(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
             "-i", video_only_path, "-i", audio_all,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
             output_path],
            0, cancel_event=cancel_event, timeout=3600.0)
        if not ok:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"音频混流失败: {err[:300]}")


# ===============================================================
#  4) 逐帧管道写入器（兜底）
# ===============================================================

def _open_ffmpeg_pipe_writer(output_path: str, fps: float, w: int, h: int, quality: int,
                             use_gpu: bool, gpu_encoder: str = "",
                             ffmpeg_path: str | None = None,
                             preset: str | None = None):
    ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    base_cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
    ]
    cmd = base_cmd + _encode_args(quality, use_gpu, gpu_encoder, ffmpeg, preset) + \
        ["-pix_fmt", "yuv420p", output_path]
    return _spawn_ffmpeg_pipe(cmd)


def _spawn_ffmpeg_pipe(cmd: list[str]):
    stderr_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=stderr_file,
            creationflags=_NO_WINDOW)
    except Exception:
        stderr_file.close()
        raise
    return _FFmpegPipe(process, stderr_file)


def _close_video_writer(writer_kind, writer, ffmpeg_proc):
    if writer_kind == "ffmpeg" and ffmpeg_proc:
        try:
            ffmpeg_proc.stdin.close()
        except OSError:
            pass
        stderr_file = ffmpeg_proc.stderr_file
        timed_out = False
        try:
            returncode = ffmpeg_proc.process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.process.kill()
            try:
                returncode = ffmpeg_proc.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                returncode = -1
            timed_out = True
        try:
            stderr_file.seek(0)
            err = stderr_file.read()
        finally:
            stderr_file.close()
        if timed_out:
            raise RuntimeError("ffmpeg 编码超时")
        if returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {err.decode('utf-8', errors='ignore')}")
    elif writer_kind == "imageio" and writer:
        writer.close()
    elif writer_kind == "cv2" and writer:
        writer.release()


def _export_per_frame(video_path: str, output_path: str, to_del: np.ndarray,
                      fps: float, quality: int, progress_cb=None,
                      use_gpu: bool = False, gpu_encoder: str = "",
                      ffmpeg_path: str | None = None, preset: str | None = None,
                      avoid_seek: bool = False, include_audio: bool = True):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ranges = _kept_frame_ranges(to_del)
    try:
        ffmpeg_bin = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        ffmpeg_bin = None

    ret, sample = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        cap.release()
        raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]

    use_ffmpeg = bool(ffmpeg_bin)
    video_only_path = output_path + ".video-only.tmp.mp4" if use_ffmpeg else output_path
    writer_kind = None
    ffmpeg_proc = None
    writer = None

    if use_ffmpeg:
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            video_only_path, fps, w, h, quality,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_bin,
            preset=preset)
        writer_kind = "ffmpeg"
    else:
        try:
            import imageio
            writer = imageio.get_writer(video_only_path, fps=fps, codec='libx264',
                                        quality=quality, pixelformat='yuv420p')
            writer_kind = "imageio"
        except ImportError:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_only_path, fourcc, fps, (w, h))
            writer_kind = "cv2"

    written = 0
    try:
        idx = 0
        while idx < total:
            if to_del[idx]:
                next_keep = idx + 1
                while next_keep < total and to_del[next_keep]:
                    next_keep += 1
                gap = next_keep - idx
                if gap > 30 and not avoid_seek:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_keep)
                else:
                    # grab() 只解码不转 BGR，比 read() 省一次色彩转换
                    for _ in range(gap):
                        if not cap.grab():
                            break
                idx = next_keep
                continue

            ret, frame = cap.read()
            if not ret:
                break
            if writer_kind == "ffmpeg":
                if ffmpeg_proc is None:
                    raise RuntimeError("FFmpeg 写入器未初始化")
                ffmpeg_proc.stdin.write(frame.tobytes())
            elif writer_kind == "imageio":
                if writer is None:
                    raise RuntimeError("imageio 写入器未初始化")
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if writer is None:
                    raise RuntimeError("OpenCV 写入器未初始化")
                writer.write(frame)
            written += 1
            idx += 1
            if progress_cb and written % 60 == 0:
                progress_cb(idx / max(1, total), written)
    finally:
        cap.release()
        try:
            _close_video_writer(writer_kind, writer, ffmpeg_proc)
        except Exception:
            if use_ffmpeg and video_only_path != output_path and os.path.isfile(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
            raise

    if use_ffmpeg:
        try:
            if include_audio:
                _mux_audio_for_ranges(video_path, video_only_path, output_path,
                                      ranges, fps, ffmpeg_path=ffmpeg_bin)
            else:
                os.replace(video_only_path, output_path)
        finally:
            if video_only_path != output_path and os.path.isfile(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
    return written, total


# ===============================================================
#  对外入口
# ===============================================================

def export_video(video_path: str, output_path: str, to_del,
                 fps: float, quality: int, progress_cb=None,
                 use_gpu: bool = False, gpu_encoder: str = "",
                 ffmpeg_path: str | None = None,
                 export_preset: str | None = None,
                 export_workers: int | None = None,
                 keyframe_copy: bool = True,
                 cancel_event: threading.Event | None = None,
                 profile=None, status_cb=None, export_audio: bool = True):
    """整段剪辑导出。返回 (written, total)。export_audio=False 则不处理音频。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if isinstance(to_del, set):
        mask = np.zeros(total, dtype=bool)
        for idx in to_del:
            if 0 <= idx < total:
                mask[idx] = True
        to_del = mask

    ranges = _kept_frame_ranges(to_del)
    if not ranges:
        raise RuntimeError("没有可导出的帧")
    written_fast = sum(end - start for start, end in ranges)

    if profile is None:
        try:
            profile = gpu_caps.detect(ffmpeg_path, probe_decoders=False)
        except Exception:
            profile = None
    workers = int(export_workers or (profile.export_workers if profile else 1) or 1)

    try:
        ffmpeg_bin = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        ffmpeg_bin = None

    if ffmpeg_bin:
        keys = None
        if keyframe_copy:
            try:
                keys = keyframe_frames(video_path, fps, ffmpeg_path=ffmpeg_bin,
                                       status_cb=status_cb)
            except Exception as exc:
                app_core.warn(f"关键帧索引不可用，跳过无损直通: {exc}", "exporter")

        app_core.debug(
            f"导出开始：{len(ranges)} 段 / {written_fast} 帧 / 并发 {workers} / "
            f"编码器 {gpu_encoder or '自动'} / 音频 {'开' if export_audio else '关'} / "
            f"预设 {export_preset or '默认'}", "exporter")

        if keys:
            try:
                if _try_copy_export(video_path, output_path, ranges, fps, profile,
                                    ffmpeg_bin, progress_cb, cancel_event,
                                    workers=max(2, workers), keyframes=keys,
                                    status_cb=status_cb, include_audio=export_audio):
                    app_core.debug("导出路径：无损直通（-c copy + concat）", "exporter")
                    return written_fast, total
            except ExportCancelled:
                raise
            except Exception as exc:
                app_core.warn(f"无损直通异常，回退重编码: {exc}", "exporter")

        if workers > 1 and len(ranges) > _MAX_RANGES_PER_CHUNK:
            try:
                if _export_chunks_parallel(video_path, output_path, ranges, fps,
                                           quality, use_gpu, gpu_encoder, ffmpeg_bin,
                                           workers, export_preset, progress_cb,
                                           cancel_event, profile, keyframes=keys,
                                           status_cb=status_cb,
                                           include_audio=export_audio):
                    app_core.debug("导出路径：分块并行重编码", "exporter")
                    return written_fast, total
            except ExportCancelled:
                raise
            except Exception as exc:
                app_core.warn(f"分块并行导出异常，回退单遍滤镜: {exc}", "exporter")

        if _export_ranges_with_ffmpeg_filters(
                video_path, output_path, ranges, fps, quality, use_gpu,
                gpu_encoder, export_audio, progress_cb, ffmpeg_path=ffmpeg_bin,
                preset=export_preset, cancel_event=cancel_event, profile=profile,
                keyframes=keys, status_cb=status_cb):
            app_core.debug("导出路径：单遍滤镜", "exporter")
            return written_fast, total

    app_core.debug("导出路径：逐帧兜底", "exporter")
    written, tot = _export_per_frame(
        video_path, output_path, to_del, fps, quality, progress_cb,
        use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_path,
        preset=export_preset, include_audio=export_audio)
    return written, tot


def _export_chunks_parallel(video_path: str, output_path: str,
                            ranges: list[tuple[int, int]], fps: float,
                            quality: int, use_gpu: bool, gpu_encoder: str,
                            ffmpeg: str, workers: int, preset: str | None,
                            progress_cb, cancel_event, profile,
                            keyframes=None, status_cb=None,
                            include_audio: bool = True) -> bool:
    """按块并行重编码再 concat。

    块大小按「最多 _MAX_RANGES_PER_CHUNK 个区间」切，而不是按并发数切：
    区间一多滤镜图初始化就极慢且期间没有任何进度输出。每块以关键帧为锚点
    做输入 seek，块内 trim 用相对帧号，因此只解码自己那一段。
    """
    expected = sum(e - s for s, e in ranges)
    # 块数同时受两头约束：至少要够喂满并发（每块固定开销不小），
    # 每块区间数又不能太多（每个 trim 分支都要过一遍解码流）
    n_chunks = max(workers, -(-len(ranges) // _MAX_RANGES_PER_CHUNK))
    per = -(-len(ranges) // n_chunks)
    chunks = [ranges[i:i + per] for i in range(0, len(ranges), per)]
    n_chunks = len(chunks)
    if n_chunks < 2:
        return False

    src_audio = _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    has_audio = include_audio and src_audio
    enc = gpu_caps.resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg) if use_gpu else None
    concurrency = max(1, min(workers, n_chunks))
    app_core.info(f"分块并行导出：{n_chunks} 块 / 并发 {concurrency} / "
                  f"编码器 {enc or 'libx264'}", "exporter")
    if status_cb:
        status_cb(f"分块编码 0/{n_chunks}（并发 {concurrency}）")
    chunk_t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="aae_chunk_") as tmpdir:
        parts = [os.path.join(tmpdir, f"c{i:03d}.mp4") for i in range(n_chunks)]
        progress = [0.0] * n_chunks
        finished = [0]
        lock = threading.Lock()

        def run_chunk(ci: int) -> bool:
            chunk = chunks[ci]
            base = _anchor_for(chunk[0][0], fps, keyframes)
            rel = [(s - base, e - base) for s, e in chunk]
            filter_file = os.path.join(tmpdir, f"f{ci:03d}.txt")
            cmd = _build_filter_cmd(ffmpeg, video_path, parts[ci], filter_file, base,
                                    rel, fps, quality, use_gpu, gpu_encoder, preset,
                                    profile, graph_audio=True, map_audio=has_audio,
                                    silent_audio=not src_audio)

            chunk_total = sum(e - s for s, e in rel)

            def _prog(frac, done, _ci=ci):
                with lock:
                    progress[_ci] = frac
                    if progress_cb:
                        progress_cb(0.02 + 0.88 * (sum(progress) / len(progress)), done)

            ok, err = _run_ffmpeg_progress(
                cmd, chunk_total, progress_cb=_prog, cancel_event=cancel_event,
                ratio_base=0.0, ratio_span=1.0, timeout=14400.0)
            with lock:
                finished[0] += 1
                done_n = finished[0]
            if not ok:
                app_core.warn(f"块 {ci + 1}/{n_chunks} 编码失败：{err[:200]}", "exporter")
            elif status_cb:
                used = time.perf_counter() - chunk_t0
                status_cb(f"分块编码 {done_n}/{n_chunks}（并发 {concurrency}，"
                          f"已用 {int(used // 60)}:{int(used % 60):02d}）")
            return ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(run_chunk, range(n_chunks)))
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled("已取消导出")
        if not all(results):
            return False

        video_only = os.path.join(tmpdir, "merged.mp4")
        if not _concat_parts(parts, video_only, ffmpeg, tmpdir):
            app_core.warn("分块 concat 失败，回退单遍滤镜", "exporter")
            return False
        ok_dur, why = _duration_ok(video_only, expected, fps, tolerance_frames=1.5)
        if not ok_dur:
            got = _video_frame_count(video_only)
            if got > 0 and abs(got - expected) <= _count_tolerance(len(ranges)):
                app_core.info(f"时长校验存疑但帧数正确（{got} vs {expected}），继续", "exporter")
            else:
                app_core.warn(f"分块导出{why}，回退单遍滤镜", "exporter")
                return False

        # 每块已经带了音频，合好的文件就是成品；这里绝不能再用整段音频滤镜
        # 重新剪音频：那会让每个音频帧都过一遍全部区间分支（几千个 atrim）。
        if status_cb:
            status_cb("写入文件…")
        shutil.move(video_only, output_path)
    if progress_cb:
        progress_cb(1.0, expected)
    return True


def export_ranges(video_path: str, output_path: str, ranges: list,
                  fps: float, quality: int, progress_cb=None,
                  use_gpu: bool = False, gpu_encoder: str = "",
                  ffmpeg_path: str | None = None,
                  export_preset: str | None = None,
                  keyframe_copy: bool = True,
                  cancel_event: threading.Event | None = None,
                  profile=None, status_cb=None, export_audio: bool = True):
    """导出一组区间为一个文件（ranges 为左闭右闭的 (start, end)）。

    分段导出走这里：起点对齐关键帧时用无损 copy；否则滤镜编码会先 seek 到
    该段之前的关键帧，只解码这一段，而不是从视频第 0 帧解到段尾。
    """
    if not ranges:
        return 0, 0

    exclusive = [(int(s), int(e) + 1) for s, e in ranges]
    total_frames_to_export = sum(e - s for s, e in exclusive)
    try:
        ffmpeg_bin = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        ffmpeg_bin = None

    if profile is None:
        try:
            profile = gpu_caps.detect(ffmpeg_path, probe_decoders=False)
        except Exception:
            profile = None

    keys = None
    if ffmpeg_bin and keyframe_copy:
        try:
            keys = keyframe_frames(video_path, fps, ffmpeg_path=ffmpeg_bin,
                                   status_cb=status_cb)
        except Exception as exc:
            app_core.warn(f"关键帧索引不可用: {exc}", "exporter")

    if ffmpeg_bin and keys:
        try:
            if _try_copy_export(video_path, output_path, exclusive, fps, profile,
                                ffmpeg_bin, progress_cb, cancel_event, workers=2,
                                keyframes=keys, status_cb=status_cb,
                                include_audio=export_audio):
                return total_frames_to_export, total_frames_to_export
        except ExportCancelled:
            raise
        except Exception as exc:
            app_core.warn(f"分段无损直通异常，回退滤镜编码: {exc}", "exporter")

    if ffmpeg_bin and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, exclusive, fps, quality,
            use_gpu, gpu_encoder, export_audio, progress_cb, ffmpeg_path=ffmpeg_bin,
            preset=export_preset, cancel_event=cancel_event, profile=profile,
            ratio_base=0.02, ratio_span=0.96, keyframes=keys, status_cb=status_cb):
        return total_frames_to_export, total_frames_to_export

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    mask = np.zeros(max(total, exclusive[-1][1]), dtype=bool)
    mask[:] = True
    for s, e in exclusive:
        mask[s:min(e, len(mask))] = False
    written, _ = _export_per_frame(
        video_path, output_path, mask, fps, quality, progress_cb,
        use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_path,
        preset=export_preset)
    return written, total_frames_to_export


# analyzer.py —— 模板加载 + 段落提取 + 删除掩码
#
# 匹配（原 matcher.py）与导出（原 exporter.py）已并入本文件；matcher.py /
# exporter.py 只剩惰性转发。解码流水线仍在 pipeline.py，能力探测在 gpu_caps.py，
# 配置与运行事件在 app_core.py。



import os
import sys

import cv2
import numpy as np

import app_core

import gpu_caps

import pipeline
from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)

# ===============================================================
#  模板加载
# ===============================================================

TEMPLATE_DIRS = {
    'pause': {'ref_dir': 'templates_pause', 'source_dir': 'source_images_pause'},
    'speed_1x': {'ref_dir': 'templates_1x', 'source_dir': 'source_images_1x'},
    'speed_2x': {'ref_dir': 'templates_2x', 'source_dir': 'source_images_2x'},
    'speed_0_2x': {'ref_dir': 'templates_play', 'source_dir': 'source_images_play'},
}

IMG_EXTS = ('.png', '.jpg', '.bmp', '.jpeg')

# 匹配时的搜索余量（处理尺度像素）：UI 相对模板源截图可能整体平移，
# 16 对应原生 1920 宽下的约 77 像素，够覆盖窗口标题栏/窗口位置差。
_MATCH_PAD = 16


def imread_gray(path: str):
    """灰度读图，支持非 ASCII 路径。

    cv2.imread 在 Windows 上用 ANSI 代码页解释路径，路径含中文时会读不到文件，
    因此走 np.fromfile + cv2.imdecode。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    except OSError:
        pass
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def asset_dir(name: str) -> str | None:
    """素材目录解析：程序目录 → onefile 解包目录 → 当前工作目录。

    打包后不能只依赖工作目录（快捷方式启动时 CWD 未必是 exe 目录）。
    """
    cands = [os.path.join(app_core.base_dir(), name)]
    bundle = app_core.bundle_dir()
    if bundle != app_core.base_dir():
        cands.append(os.path.join(bundle, name))
    cands.append(name)
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def load_templates(proc_res: tuple = (400, 225),
                   drop_indistinct: bool = True) -> tuple[dict, int]:
    configs: dict[str, list] = {k: [] for k in TEMPLATE_DIRS}
    total = 0

    for ctype, dirs in TEMPLATE_DIRS.items():
        src_dir = asset_dir(dirs['source_dir'])
        ref_dir = asset_dir(dirs['ref_dir'])
        if not src_dir or not ref_dir:
            continue

        src_files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(IMG_EXTS)]
        if not src_files or not ref_files:
            continue

        src_img = imread_gray(os.path.join(src_dir, src_files[0]))
        if src_img is None:
            continue
        sh, sw = src_img.shape

        # 整帧缩放后的源图：模板要从这里裁，才能和「分析时整帧缩放」的像素一致。
        # 若改成「原图裁块再单独缩放」，插值与取整会和帧不同，模板自匹配分数
        # 会掉到 0.7 上下，阈值就没有余量（实测 2x 模板 0.737、1x 0.862）。
        scaled_src = None
        if (sw, sh) != (int(proc_res[0]), int(proc_res[1])):
            scaled_src = cv2.resize(src_img, (int(proc_res[0]), int(proc_res[1])),
                                    interpolation=cv2.INTER_AREA)

        for rf in ref_files:
            ref_img = imread_gray(os.path.join(ref_dir, rf))
            if ref_img is None:
                continue
            rh, rw = ref_img.shape

            res = cv2.matchTemplate(src_img, ref_img, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            rx, ry = max_loc
            _, mask = cv2.threshold(ref_img, 10, 255, cv2.THRESH_BINARY)

            scale_x, scale_y = proc_res[0] / sw, proc_res[1] / sh
            tw, th = max(1, int(round(rw * scale_x))), max(1, int(round(rh * scale_y)))

            cached_t = None
            if scaled_src is not None:
                tx, ty = int(round(rx * scale_x)), int(round(ry * scale_y))
                patch = scaled_src[ty:ty + th, tx:tx + tw]
                if patch.shape == (th, tw):
                    cached_t = patch.copy()
            if cached_t is None:
                cached_t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

            # 搜索窗口用「固定像素余量」而不是模板尺寸的倍数：带窗口标题栏/窗口偏移的
            # 录制里，UI 相对模板源截图可能整体平移几十个像素（实测 dx≈-65 dy≈+42
            # 原生像素）。按 2 倍模板算出来的窗口只有 ±半个模板宽，直接漏检。
            erx = max(0, int(round(rx * scale_x)) - _MATCH_PAD)
            ery = max(0, int(round(ry * scale_y)) - _MATCH_PAD)
            configs[ctype].append({
                'roi_orig': (rx, ry, rw, rh),
                'source_res': (sw, sh),
                'cached_proc_res': proc_res,
                'cached_roi': (erx, ery, tw + 2 * _MATCH_PAD, th + 2 * _MATCH_PAD),
                'cached_t': cached_t,
                'cached_m': cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST),
            })
            total += 1

    # 所有入口统一在这里剔除「区分不了类别」的模板（例如把常驻 UI 元素
    # 当成 0.2 倍速标志），否则界面、自检、分析三条路的分类结果会不一致。
    if drop_indistinct:
        configs, notes = filter_indistinct_templates(configs, proc_res)
        for note in notes:
            app_core.warn(note, "matcher")
        total = sum(len(v) for v in configs.values())
    return configs, total


def _source_frames(proc_res: tuple) -> dict:
    """每类取一张「整帧缩放后」的源图，返回 {类别: (内容指纹, 帧)}。

    指纹用于识别重复素材：source_images_play 与 source_images_2x 可能放的
    是同一张截图，不能拿它当另一类的负例（否则会把正确的模板误判成不可区分）。
    """
    import hashlib

    out: dict = {}
    for ctype, dirs in TEMPLATE_DIRS.items():
        src_dir = asset_dir(dirs['source_dir'])
        if not src_dir:
            continue
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        if not files:
            continue
        path = os.path.join(src_dir, files[0])
        img = imread_gray(path)
        if img is None:
            continue
        try:
            with open(path, 'rb') as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
        except OSError:
            digest = os.path.normcase(path)
        sh, sw = img.shape
        if (sw, sh) != (int(proc_res[0]), int(proc_res[1])):
            img = cv2.resize(img, (int(proc_res[0]), int(proc_res[1])),
                             interpolation=cv2.INTER_AREA)
        out[ctype] = (digest, img)
    return out


def template_discrimination(configs: dict, proc_res: tuple = (400, 225)) -> dict:
    """每类模板的区分度：在自己源图上的分数 vs 在「别类且不同源图」上的最高分。

    比值接近 1 说明这类模板在别的类别画面上也一样高分——它区分不了类别。
    典型例子：把常驻 UI 元素（速度条上的暂停图标等）当成「0.2 倍速」标志，
    结果任何播放中的画面都会被判成 0.2 倍速。
    """
    frames = _source_frames(proc_res)

    out: dict = {}
    for ctype, tpls in configs.items():
        if not tpls or ctype not in frames:
            continue
        own_digest, own_frame = frames[ctype]
        own = max(float(get_best_score(own_frame, [t], proc_res)) for t in tpls)
        cross = 0.0
        for other, (odigest, oframe) in frames.items():
            if other == ctype or odigest == own_digest:
                continue
            for t in tpls:
                cross = max(cross, float(get_best_score(oframe, [t], proc_res)))
        out[ctype] = {'self': own, 'cross': cross,
                      'ratio': (cross / own) if own > 1e-6 else 99.0}
    return out


def filter_indistinct_templates(configs: dict, proc_res: tuple = (400, 225),
                                ratio_limit: float = 0.95) -> tuple[dict, list[str]]:
    """剔除「区分不了类别」的模板（在别类源图上同样高分），返回 (configs, 说明)。"""
    disc = template_discrimination(configs, proc_res)
    notes: list[str] = []
    kept = {k: list(v) for k, v in configs.items()}
    for ctype, d in disc.items():
        # speed_0_2x 用「正在播放」图标当兜底判据（1x/2x 都没命中才算），
        # 它在别的画面上也高分是正常的，不能因此停用。
        if ctype == 'speed_0_2x':
            continue
        if d['ratio'] >= ratio_limit and d['self'] > 0:
            kept[ctype] = []
            notes.append(f"{ctype} 已停用：模板在别类画面上也能得 {d['cross']:.3f}"
                         f"（自己 {d['self']:.3f}），无法区分，请换一张真正的标志图")
    return kept, notes


def template_self_scores(configs: dict, proc_res: tuple = (400, 225)) -> dict:
    """每个模板在它自己的源截图上的匹配分数（理想接近 1）。

    这是阈值是否留有余量的直接指标：自匹配只有 0.7x 时，真实视频上很容易
    跌破阈值，从而漏判或串类（例如 2 倍速被判成 0.2 倍速）。
    """
    out: dict[str, float] = {}
    for ctype, dirs in TEMPLATE_DIRS.items():
        src_dir = asset_dir(dirs['source_dir'])
        ref_dir = asset_dir(dirs['ref_dir'])
        if not src_dir or not ref_dir:
            continue
        src_files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        if not src_files:
            continue
        src_img = imread_gray(os.path.join(src_dir, src_files[0]))
        if src_img is None:
            continue
        sh, sw = src_img.shape
        frame = src_img
        if (sw, sh) != (int(proc_res[0]), int(proc_res[1])):
            frame = cv2.resize(src_img, (int(proc_res[0]), int(proc_res[1])),
                               interpolation=cv2.INTER_AREA)
        best = -1.0
        for t in configs.get(ctype, []):
            score = get_best_score(frame, [t], proc_res)
            best = max(best, float(score))
        if best > -1.0:
            out[ctype] = best
    return out


# ===============================================================
#  单帧匹配（进程池后端使用）
# ===============================================================

_get_best_score = get_best_score
_classify_gray = classify_gray

_worker_configs: dict = {}
_worker_thresholds: dict = {}
_worker_proc_res: tuple = (400, 225)


def _worker_init(configs: dict, thresholds: dict, proc_res: tuple):
    global _worker_configs, _worker_thresholds, _worker_proc_res
    _worker_configs = configs
    _worker_thresholds = thresholds
    _worker_proc_res = proc_res


def _worker_classify_gray(gray: np.ndarray) -> int:
    return _classify_gray(gray, _worker_configs, _worker_thresholds, _worker_proc_res)


# ===============================================================
#  解码后端 / 能力探测 / 分析入口（re-export）
# ===============================================================

DECODE_BACKEND_OPENCV = pipeline.DECODE_BACKEND_OPENCV
DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH = pipeline.DECODE_BACKEND_FFMPEG_SW
DECODE_BACKEND_FFMPEG_SW = pipeline.DECODE_BACKEND_FFMPEG_SW

resolve_ffmpeg_path = gpu_caps.resolve_ffmpeg_path
normalize_decode_backend = pipeline.normalize_decode_backend
resolve_decode_backend = pipeline.resolve_decode_backend
decode_backend_label = pipeline.decode_backend_label

ANALYSIS_CONTEXT_VERSION = pipeline.ANALYSIS_CONTEXT_VERSION
_BoundaryTracker = pipeline._BoundaryTracker
_make_analysis_context = pipeline._make_analysis_context
context_records_for_pauses = pipeline.context_records_for_pauses
analysis_context_skips_second_scan = pipeline.analysis_context_skips_second_scan
_finalize_analysis_arrays = pipeline._finalize_analysis_arrays

analyze_video = pipeline.analyze_video
analyze_video_with_context = pipeline.analyze_video_with_context


# ===============================================================
#  段落提取
# ===============================================================

def _analyze_pause_mask(s_i: int, e_i: int, diffs: np.ndarray,
                        still_frames: int, motion_thresh: float):
    """暂停段内部删除掩码（游程统计用 np.diff 向量化）。"""
    seg_len = e_i - s_i + 1
    if seg_len <= 0:
        return np.zeros(0, dtype=np.uint8), 'all'

    # active[k] = (diffs[s_i+k] > thr) 或 (diffs[s_i+k+1] > thr)
    active_mask = np.zeros(seg_len, dtype=bool)
    if seg_len > 1:
        hot = diffs[s_i + 1: s_i + seg_len] > motion_thresh
        active_mask[1:] = hot
        active_mask[:-1] |= hot

    del_mask = np.zeros(seg_len, dtype=np.uint8)

    change = np.flatnonzero(np.diff(active_mask.view(np.int8))) + 1
    run_starts = np.concatenate(([0], change))
    run_ends = np.concatenate((change - 1, [seg_len - 1]))
    run_vals = active_mask[run_starts]

    if not bool(run_vals.any()):
        if seg_len > 2 * still_frames:
            del_mask[still_frames: seg_len - still_frames] = 1
        return del_mask, 'auto'

    for val, s, e in zip(run_vals, run_starts, run_ends):
        if val:
            continue
        run_len = int(e) - int(s) + 1
        if run_len > still_frames:
            if s == 0:
                keep_start = int(e) - still_frames + 1
                del_mask[0:keep_start] = 1
            elif e == seg_len - 1:
                keep_end = int(s) + still_frames - 1
                del_mask[keep_end + 1:int(e) + 1] = 1
            else:
                half = still_frames // 2
                other_half = still_frames - half
                del_mask[int(s) + half: int(e) - other_half + 1] = 1

    return del_mask, 'auto'


def build_segments(states: np.ndarray, diffs: np.ndarray, video_path: str,
                   proc_res: tuple, compare_cfg: dict, fps: float,
                   progress_cb=None, *, analysis_context=None) -> tuple[list, list]:
    total = len(states)
    pauses = []
    speeds = []

    still_time = compare_cfg.get('still_time_thresh', 0.1)
    motion_thresh = compare_cfg.get('motion_thresh', 2.0)
    boundary_thresh = compare_cfg.get('boundary_thresh', 5.0)
    still_frames = max(2, int(fps * still_time))

    i = 0
    while i < total:
        curr = int(states[i])
        s_i = i
        while i < total and int(states[i]) == curr:
            i += 1
        e_i = i - 1

        if curr == FRAME_TYPE_PAUSE:
            del_mask, mode = _analyze_pause_mask(s_i, e_i, diffs, still_frames, motion_thresh)
            pauses.append({
                'id': len(pauses),
                'start': s_i,
                'end': e_i,
                'mode': mode,
                'local_del_mask': del_mask,
                'boundary_diff': 0.0
            })
            if progress_cb:
                progress_cb(1.0)

        elif curr in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
            speeds.append({'type': curr, 'start': s_i, 'end': e_i})

    # 暂停段的边界差分只用第一遍分析留下的上下文。
    # 拿不到上下文、或者某段算出来一帧都不该删 → 这一类暂停段不做任何处理：
    # 帧状态改成 others(正常) 原样直出，既不剪也不参与后续速度处理。
    records = None
    if pauses and analysis_context is not None:
        records = context_records_for_pauses(analysis_context, pauses, total)

    if records is None:
        if pauses:
            app_core.info(f"{len(pauses)} 个暂停段没有边界上下文，"
                          f"按 others 原样输出（不剪）", "analyzer")
            for p in pauses:
                states[p['start']:p['end'] + 1] = FRAME_TYPE_NORMAL
            pauses = []
    else:
        kept = []
        for p, rec in zip(pauses, records):
            diff = float(rec['diff'])
            p['boundary_diff'] = diff
            if diff < boundary_thresh:
                p['mode'] = 'all'
            elif not np.any(np.asarray(p.get('local_del_mask', []))):
                # 掩码一帧都不删 → 同样按 others 直出
                states[p['start']:p['end'] + 1] = FRAME_TYPE_NORMAL
                continue
            kept.append(p)
        if len(kept) != len(pauses):
            app_core.info(f"{len(pauses) - len(kept)} 个暂停段掩码为空，"
                          f"按 others 原样输出（不剪）", "analyzer")
        pauses = kept

    if progress_cb:
        progress_cb(1.0)

    return pauses, speeds


# ===============================================================
#  删除掩码
# ===============================================================

def _speedup_mask(states: np.ndarray, frame_type: int, factor: int,
                  exclude_mask: np.ndarray) -> np.ndarray:
    """变速跳帧掩码。

    用 np.maximum.accumulate 求「当前位置所属段起点」，整体 O(总帧数)；
    逐段回填 offsets 的写法是 O(段数 × 总帧数)。
    """
    total = len(states)
    type_mask = (states == frame_type) & ~exclude_mask

    if not type_mask.any():
        return np.zeros(total, dtype=bool)

    cumsum = np.cumsum(type_mask)
    shifted = np.empty(total, dtype=bool)
    shifted[0] = False
    shifted[1:] = type_mask[:-1]
    seg_flags = type_mask & ~shifted

    idx = np.arange(total, dtype=np.int64)
    last_start = np.maximum.accumulate(np.where(seg_flags, idx, -1))
    base = np.where(last_start > 0, cumsum[np.maximum(last_start - 1, 0)], 0)
    local_cnt = np.where(type_mask, cumsum - base, 0)

    if factor == 2:
        return type_mask & (local_cnt % 2 == 0)
    return type_mask & (local_cnt % factor != 1)


def build_delete_set(total: int, states: np.ndarray,
                     pause_segments: list, speed_segments: list,
                     clip_segments: list,
                     speedup_1x: bool, speedup_02: bool,
                     speedup_02_factor: int) -> np.ndarray:
    # 容器元数据帧数与实际分析帧数可能差一两帧（末帧索引/时间戳问题）。
    # 一律按「实际分析长度」对齐，否则 del_mask 与 _speedup_mask 长度不同会直接
    # 报 operands could not be broadcast together。
    n = int(total)
    if states is not None and len(states) != n:
        n = min(n, len(states)) if len(states) > 0 else n
    n = max(0, n)
    del_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return del_mask

    for seg in pause_segments:
        s, e = int(seg['start']), int(seg['end'])
        if s >= n:
            continue
        e = min(e, n - 1)
        m = seg.get('local_del_mask')
        if seg.get('mode') == 'all' or m is None or len(m) != e - s + 1:
            # 边界是渐变（boundary_diff 小）→ 整段删掉，这是最干脆的剪法
            del_mask[s:e + 1] = True
        else:
            # 硬切边界 → 按掩码删：<3 帧的抖动算噪声照删，≥3 帧的动画面保留
            del_mask[s:e + 1] = (m == 1) | (m == 2)

    for seg in clip_segments:
        s, e = int(seg['start']), int(seg['end'])
        if s >= n:
            continue
        e = min(e, n - 1)
        ki, ko = seg['keep_in'], seg['keep_out']
        if ki > ko:
            del_mask[s:e + 1] = True
        else:
            if ki > s:
                del_mask[s:min(ki, n)] = True
            if ko < e:
                del_mask[max(ko + 1, 0):e + 1] = True

    if speedup_1x:
        del_mask |= _speedup_mask(states[:n], FRAME_TYPE_1X, 2, del_mask)

    if speedup_02 and speedup_02_factor > 1:
        del_mask |= _speedup_mask(states[:n], FRAME_TYPE_0_2X, speedup_02_factor, del_mask)

    return del_mask


# ===============================================================
#  导出（re-export）
# ===============================================================

