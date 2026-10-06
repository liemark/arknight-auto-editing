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
    if configs['speed_0_2x'] and get_best_score(gray, configs['speed_0_2x'],
                                               proc_res) >= thresholds['speed_0_2x']:
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
        m02 = pending & ~m1 & ~m2 & (scores['speed_0_2x'] >= thresholds['speed_0_2x'])
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
    ref_states = classify_from_scores(ref_scores, compiled.has, thresholds)

    pool = candidates if candidates is not None else available_backends(profile)
    results: list[BenchResult] = []
    for backend in pool:
        r = BenchResult(backend=backend, ok=False, frames=n)
        if backend == BACKEND_CPU_POOL:
            r.error = "不参与自动测速（兼容保留）"
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
        except Exception as exc:
            r.error = f"{type(exc).__name__}: {exc}"
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
