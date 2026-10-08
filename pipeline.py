# pipeline.py —— 解码后端（含硬件解码）+ 读帧/批处理流水线 + 帧差向量化
#
# 解码后端：
#   opencv                 cv2.VideoCapture（硬件加速优先）+ 线程池并行 resize/转灰
#   ffmpeg_sw_passthrough  FFmpeg 软件解码（scale=area,format=gray）
#   <硬件变体 key>          NVDEC/QSV/D3D11VA 硬解 + GPU/软件缩放 + 转灰
# 硬件变体由 gpu_caps 实测选出。
#
# 读帧线程持续预读，主线程只做批匹配，两者重叠。

from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import cv2
import numpy as np

import app_core
import gpu_caps
from frame_types import FRAME_TYPE_PAUSE

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# ===============================================================
#  后端常量与解析
# ===============================================================

DECODE_BACKEND_OPENCV = "opencv"
DECODE_BACKEND_FFMPEG_SW = "ffmpeg_sw_passthrough"
DECODE_BACKEND_AUTO = "auto"

_DECODE_ALIASES = {
    "opencv": DECODE_BACKEND_OPENCV,
    "cv2": DECODE_BACKEND_OPENCV,
    "default": DECODE_BACKEND_OPENCV,
    "ffmpeg_sw_passthrough": DECODE_BACKEND_FFMPEG_SW,
    "ffmpeg_sw_gray_passthrough": DECODE_BACKEND_FFMPEG_SW,
    "a_pt": DECODE_BACKEND_FFMPEG_SW,
    "auto": DECODE_BACKEND_AUTO,
}

_HARDWARE_VARIANTS = {
    "nv_scale_cuda", "nv_scale_npp", "nv_hwdec_swscale",
    "intel_scale_qsv", "intel_vpp_qsv", "intel_hwdec_swscale",
    "d3d11va", "dxva2",
}


def normalize_decode_backend(decode_backend: str | None) -> str:
    key = (decode_backend or DECODE_BACKEND_OPENCV).strip().lower()
    if key in _DECODE_ALIASES:
        return _DECODE_ALIASES[key]
    if key in _HARDWARE_VARIANTS:
        return key
    raise ValueError(
        f"unknown decode_backend={decode_backend!r}; allowed="
        f"{sorted(set(_DECODE_ALIASES.values())) + sorted(_HARDWARE_VARIANTS)}"
    )


def resolve_decode_backend(requested: str | None, profile: gpu_caps.GpuProfile | None = None,
                           ffmpeg_path: str | None = None) -> str:
    """auto：优先实测通过的硬件变体；否则退回 OpenCV（等价于改造前行为）。"""
    key = normalize_decode_backend(requested)
    if key != DECODE_BACKEND_AUTO:
        return key
    try:
        prof = profile or gpu_caps.detect(ffmpeg_path)
    except Exception as exc:
        app_core.warn(f"自动选择解码后端失败，回退 OpenCV: {exc}", "pipeline")
        return DECODE_BACKEND_OPENCV
    return prof.decode_variant or DECODE_BACKEND_OPENCV


def decode_backend_label(key: str) -> str:
    if key == DECODE_BACKEND_OPENCV:
        return "OpenCV"
    if key == DECODE_BACKEND_FFMPEG_SW:
        return "FFmpeg 软件"
    for v in gpu_caps.build_decode_variants(["cuda", "qsv", "d3d11va", "dxva2"]):
        if v.key == key:
            return "硬件解码 " + v.label
    return key


# ===============================================================
#  帧源
# ===============================================================

class GrayFrameSource:
    """read_batch(n) → list[np.ndarray]，空列表代表结束。"""

    def __init__(self, video_path: str, proc_res: tuple):
        self.video_path = video_path
        self.proc_res = (int(proc_res[0]), int(proc_res[1]))
        self.total = 0

    def read_batch(self, n: int) -> list[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class OpenCvGraySource(GrayFrameSource):
    """cv2.VideoCapture 解码 + 线程池并行 resize/cvtColor。"""

    def __init__(self, video_path: str, proc_res: tuple, n_threads: int = 4):
        super().__init__(video_path, proc_res)
        self.cap = gpu_caps.open_video_capture(video_path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_threads = max(1, int(n_threads))
        self._eof = False
        self._pending: list = []
        self._pool: ThreadPoolExecutor | None = None

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        pw, ph = self.proc_res
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def read_batch(self, n: int) -> list[np.ndarray]:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.n_threads,
                                            thread_name_prefix="prep")
        out: list[np.ndarray] = []
        window = max(2, self.n_threads * 2)
        while len(out) < n:
            while not self._eof and len(self._pending) < window:
                ret, frame = self.cap.read()
                if not ret:
                    self._eof = True
                    break
                self._pending.append(self._pool.submit(self._prep, frame))
            if not self._pending:
                break
            out.append(self._pending.pop(0).result())
        return out

    def close(self) -> None:
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class FfmpegGraySource(GrayFrameSource):
    """FFmpeg 管道解码（软件或硬件）→ 灰度 → rawvideo，读帧线程预读。"""

    def __init__(self, video_path: str, proc_res: tuple, backend_key: str,
                 ffmpeg_path: str | None = None, readahead: int = 24,
                 total: int | None = None):
        super().__init__(video_path, proc_res)
        self.ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
        self.backend_key = backend_key
        self.bpf = self.proc_res[0] * self.proc_res[1]

        if total is None:
            cap = gpu_caps.open_video_capture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        self.total = int(total or 0)

        info = gpu_caps.ffmpeg_info(self.ffmpeg)
        variant = None
        if backend_key == DECODE_BACKEND_FFMPEG_SW:
            variant = gpu_caps.DecodeVariant("sw", "none", "软件解码 + 软件缩放", [],
                                             "scale={w}:{h}:flags=area,format=gray")
        else:
            for v in gpu_caps.build_decode_variants(info.get("hwaccels") or []):
                if v.key == backend_key:
                    variant = v
                    break
        if variant is None:
            raise RuntimeError(f"未知的 ffmpeg 解码后端: {backend_key}")
        self.variant = variant

        frames_arg = self.total if self.total > 0 else 0
        self.cmd = gpu_caps.build_decode_cmd(
            self.ffmpeg, video_path, frames_arg, self.proc_res[0], self.proc_res[1], variant)
        self.timeout_s = max(180.0, 120.0 + float(max(self.total, 0)) * 0.06)

        self._q: queue.Queue = queue.Queue(maxsize=max(2, readahead))
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str = ""
        self._eof = False
        self.decoded = 0

    def start(self) -> None:
        self._stderr_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            creationflags=_NO_WINDOW if sys.platform == "win32" else 0)
        self._thread = threading.Thread(target=self._reader, daemon=True, name="ffmpeg-reader")
        self._thread.start()

    def _stderr_tail(self) -> str:
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - 4096), os.SEEK_SET)
            return self._stderr_file.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _reader(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        bpf = self.bpf
        try:
            while not self._stop.is_set():
                buf = bytearray()
                while len(buf) < bpf:
                    chunk = proc.stdout.read(bpf - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) == 0:
                    break
                if len(buf) != bpf:
                    self._error = (f"FFmpeg 在帧 {self.decoded} 处输出不完整帧 "
                                   f"({len(buf)}/{bpf} 字节) stderr={self._stderr_tail()[:300]}")
                    break
                arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                    (self.proc_res[1], self.proc_res[0])).copy()
                while not self._stop.is_set():
                    try:
                        self._q.put(arr, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                self.decoded += 1
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            # 哨兵必须送达，否则主线程会在最后一轮空等到超时
            while True:
                try:
                    self._q.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if self._stop.is_set():
                        break

    def read_batch(self, n: int) -> list[np.ndarray]:
        if self._proc is None:
            self.start()
        if self._eof:
            return []
        out: list[np.ndarray] = []
        deadline = time.perf_counter() + self.timeout_s
        while len(out) < n:
            if self._error:
                if out:
                    break
                raise RuntimeError(f"FFmpeg 解码失败: {self._error}")
            timeout = max(0.05, deadline - time.perf_counter())
            if timeout <= 0.05:
                raise TimeoutError(
                    f"FFmpeg 解码超时（已解 {self.decoded}/{self.total} 帧，"
                    f"后端 {self.backend_key}）")
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._eof = True
                break
            out.append(item)
        return out

    def close(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass


def make_source(video_path: str, proc_res: tuple, backend_key: str,
                n_threads: int = 4, ffmpeg_path: str | None = None,
                total: int | None = None) -> GrayFrameSource:
    if backend_key == DECODE_BACKEND_OPENCV:
        return OpenCvGraySource(video_path, proc_res, n_threads=n_threads)
    return FfmpegGraySource(video_path, proc_res, backend_key,
                            ffmpeg_path=ffmpeg_path, total=total)


def iter_gray_frames(video_path: str, proc_res: tuple, decode_backend=None,
                     start_frame: int = 0, max_frames: int | None = None,
                     ffmpeg_path: str | None = None) -> Iterator[np.ndarray]:
    """简易灰度帧迭代器（抽帧/测速用），不做流水线。"""
    key = resolve_decode_backend(decode_backend, ffmpeg_path=ffmpeg_path)
    if key == DECODE_BACKEND_OPENCV or start_frame > 0:
        cap = gpu_caps.open_video_capture(video_path)
        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            pw, ph = int(proc_res[0]), int(proc_res[1])
            produced = 0
            while max_frames is None or produced < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                yield cv2.cvtColor(cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2GRAY)
                produced += 1
        finally:
            cap.release()
        return

    src = FfmpegGraySource(video_path, proc_res, key, ffmpeg_path=ffmpeg_path)
    try:
        produced = 0
        while max_frames is None or produced < max_frames:
            batch = src.read_batch(1)
            if not batch:
                break
            yield batch[0]
            produced += 1
    finally:
        src.close()


# ===============================================================
#  帧差（向量化）
# ===============================================================

def batch_diffs(batch: list[np.ndarray] | np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """与逐帧 cv2.mean(cv2.absdiff(g, prev_g)) 等价。

    第 i 帧的 diff 是它与前一个解码帧之差；第 0 帧的前一帧是 prev（跨批边界），
    prev 为空时记 0.0。
    """
    arr = batch if isinstance(batch, np.ndarray) else np.stack(batch)
    n = arr.shape[0]
    out = np.zeros(n, np.float64)
    if n == 0:
        return out
    if n > 1:
        d = np.abs(arr[1:].astype(np.int16) - arr[:-1].astype(np.int16))
        out[1:] = d.mean(axis=(1, 2), dtype=np.float64)
    if prev is not None:
        d0 = np.abs(arr[0].astype(np.int16) - prev.astype(np.int16))
        out[0] = float(d0.mean(dtype=np.float64))
    return out


# ===============================================================
#  暂停边界上下文
# ===============================================================

ANALYSIS_CONTEXT_VERSION = 1


class _BoundaryTracker:
    """按顺序记录每个暂停段的边界差分。

    只保留最多两张小灰度图，内存不随视频时长增长。
    逻辑长度一律取「实际解码帧数 L」，不是容器元数据。
    """

    def __init__(self):
        self.records: list[dict] = []
        self._open_start: int | None = None
        self._before_gray: np.ndarray | None = None
        self._skipped_records = 0

    def observe(self, idx: int, gray: np.ndarray, state: int,
                prev_gray: np.ndarray | None) -> None:
        is_pause = int(state) == FRAME_TYPE_PAUSE
        if self._open_start is None:
            if is_pause:
                self._open_start = idx
                src = gray if idx == 0 else prev_gray
                self._before_gray = None if src is None else np.ascontiguousarray(src).copy()
        elif not is_pause:
            self._close_run(end=idx - 1, after_idx=idx, after_gray=gray)

    def finish_with_last(self, decoded: int, last_gray: np.ndarray | None) -> None:
        if self._open_start is None:
            return
        if decoded <= 0:
            self._open_start = None
            self._before_gray = None
            self._skipped_records += 1
            return
        end = int(decoded) - 1
        after_idx = min(int(decoded) - 1, end + 1)
        if after_idx <= end and last_gray is not None:
            self._close_run(end=end, after_idx=after_idx, after_gray=last_gray)
            return
        self._open_start = None
        self._before_gray = None
        self._skipped_records += 1

    def _close_run(self, end: int, after_idx: int, after_gray: np.ndarray) -> None:
        start = int(self._open_start)
        before_idx = max(0, start - 1)
        if self._before_gray is None:
            self._open_start = None
            self._before_gray = None
            self._skipped_records += 1
            return
        diff = float(cv2.mean(cv2.absdiff(self._before_gray, after_gray))[0])
        self.records.append({"start": start, "end": int(end),
                             "before_index": int(before_idx),
                             "after_index": int(after_idx), "diff": diff})
        self._open_start = None
        self._before_gray = None

    @property
    def skipped(self) -> int:
        return self._skipped_records


def _make_analysis_context(logical_len: int, records: list[dict], complete: bool) -> dict:
    L = int(logical_len)
    return {"version": ANALYSIS_CONTEXT_VERSION, "complete": bool(complete),
            "frame_count": L, "decoded_frame_count": L,
            "pause_boundary_diffs": list(records)}


# 容器元数据帧数与实际解码帧数常差 1~2 帧（末帧时间戳/索引问题）。
# 这个量级直接放行：不打印提示，也不因此丢掉第一遍上下文去重扫。
_FRAME_COUNT_TOLERANCE = 3


def context_records_for_pauses(analysis_context, pauses: list, total: int):
    """整体可用才返回 records，否则 None（不混用缓存与重扫结果）。"""
    if not isinstance(analysis_context, dict):
        return None
    try:
        if analysis_context.get("version") != ANALYSIS_CONTEXT_VERSION:
            return None
        if analysis_context.get("complete") is not True:
            return None
        L = int(analysis_context.get("frame_count", -1))
        if L <= 0 or abs(L - int(total)) >= _FRAME_COUNT_TOLERANCE:
            return None
        if abs(int(analysis_context.get("decoded_frame_count", -1)) - L) \
                >= _FRAME_COUNT_TOLERANCE:
            return None
        records = analysis_context.get("pause_boundary_diffs")
        if not isinstance(records, list) or len(records) != len(pauses):
            return None
        for rec, p in zip(records, pauses):
            if not isinstance(rec, dict):
                return None
            start, end = int(rec["start"]), int(rec["end"])
            before, after = int(rec["before_index"]), int(rec["after_index"])
            diff = float(rec["diff"])
            if start != int(p["start"]) or end != int(p["end"]):
                return None
            if before != max(0, start - 1) or after != min(L - 1, end + 1):
                return None
            if not np.isfinite(diff):
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return records


def analysis_context_skips_second_scan(analysis_context, pauses: list, total: int) -> bool:
    if not pauses:
        return True
    return context_records_for_pauses(analysis_context, pauses, total) is not None


def _finalize_analysis_arrays(states: np.ndarray, diffs: np.ndarray,
                              allocated_total: int, decoded: int,
                              tracker: _BoundaryTracker | None,
                              last_gray: np.ndarray | None, silent: bool = False):
    if decoded < len(states):
        states = states[:decoded]
        diffs = diffs[:decoded]
    L = int(decoded)
    gap = int(allocated_total) - L if allocated_total > 0 else 0
    if not silent and gap >= _FRAME_COUNT_TOLERANCE:
        app_core.info(f"实际解码 {L} 帧 < 容器元数据 {int(allocated_total)} 帧，"
                      f"按实际长度计算上下文完整性", "pipeline")
    if tracker is not None:
        tracker.finish_with_last(L, last_gray)
        complete = tracker.skipped == 0 and L > 0
        context = _make_analysis_context(L, tracker.records, complete)
    else:
        context = _make_analysis_context(L, [], False)
    return states, diffs, context


# ===============================================================
#  主分析流程
# ===============================================================

def analyze_video_with_context(
    video_path: str,
    configs: dict,
    thresholds: dict,
    proc_res: tuple,
    batch_size: int,
    n_threads: int,
    progress_cb=None,
    decode_backend: str = DECODE_BACKEND_OPENCV,
    ffmpeg_path: str | None = None,
    match_backend: str | None = None,
    skip_identical: bool = True,
    on_stats=None,
    profile=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """整段分析：返回 (states, diffs, analysis_context)。"""
    backend_key = resolve_decode_backend(decode_backend, profile=profile,
                                        ffmpeg_path=ffmpeg_path)
    batch_size = max(1, int(batch_size))
    # matcher 的实现已并入 analyzer，模块级导入会形成循环依赖（analyzer 在本模块
    # 初始化期间就取 DECODE_BACKEND_OPENCV），因此改成调用时再导入。
    import matcher as matcher_mod
    compiled = matcher_mod.compile_templates(configs, proc_res)
    chosen = matcher_mod.resolve_backend(match_backend, profile=profile)
    matcher = matcher_mod.Matcher(compiled, configs, thresholds, proc_res,
                                 backend=chosen, skip_identical=skip_identical,
                                 n_threads=n_threads)

    src = make_source(video_path, proc_res, backend_key, n_threads=n_threads,
                      ffmpeg_path=ffmpeg_path)
    app_core.debug(f"分析参数：解码={backend_key} 匹配={chosen} 批={batch_size} "
                   f"线程={n_threads} 处理分辨率={proc_res[0]}x{proc_res[1]} "
                   f"跳过相同帧={skip_identical}", "pipeline")
    total_alloc = max(0, int(src.total or 0))
    use_dynamic = total_alloc <= 0
    states = np.zeros(total_alloc, dtype=np.int8)
    diffs = np.zeros(total_alloc, dtype=np.float32)
    states_list: list[int] = []
    diffs_list: list[float] = []
    tracker = _BoundaryTracker()

    idx = 0
    prev_gray: np.ndarray | None = None
    last_gray: np.ndarray | None = None
    t_start = time.perf_counter()
    try:
        while True:
            batch = src.read_batch(batch_size)
            if not batch:
                break
            batch_arr = np.stack(batch) if len(batch) > 1 else batch[0][None, ...]
            d = batch_diffs(batch_arr, prev_gray)
            st = matcher.classify(batch_arr)

            for i in range(len(batch)):
                if use_dynamic:
                    states_list.append(int(st[i]))
                    diffs_list.append(float(d[i]))
                elif idx + i < len(states):
                    states[idx + i] = int(st[i])
                    diffs[idx + i] = float(d[i])
                tracker.observe(idx + i, batch[i], int(st[i]),
                                prev_gray if i == 0 else batch[i - 1])
            prev_gray = batch[-1]
            last_gray = batch[-1]
            idx += len(batch)

            if progress_cb:
                # 二次扫片已不需要，解码+匹配就是全部工作量，进度直接走到 100%
                denom = max(1, total_alloc if total_alloc > 0 else idx)
                progress_cb(min(1.0, idx / denom))
    finally:
        src.close()

    elapsed = max(1e-6, time.perf_counter() - t_start)
    app_core.debug(
        f"分析完成：{idx} 帧 / {elapsed:.2f}s / {elapsed * 1000.0 / max(1, idx):.3f} ms/帧"
        f" · 匹配 {matcher.ms_per_frame:.3f} ms/帧"
        f" · 跳过相同帧 {matcher.skip_ratio() * 100:.1f}%"
        f" · 实际解码 {idx} / 容器元数据 {total_alloc}", "pipeline")
    if on_stats:
        try:
            on_stats({
                "decode_backend": backend_key,
                "match_backend": chosen,
                "frames": idx,
                "seconds": elapsed,
                "ms_per_frame": elapsed * 1000.0 / max(1, idx),
                "match_ms_per_frame": matcher.ms_per_frame,
                "skip_ratio": matcher.skip_ratio(),
                "matcher_stats": dict(matcher.stats),
            })
        except Exception:
            pass

    if use_dynamic:
        states = np.asarray(states_list, dtype=np.int8)
        diffs = np.asarray(diffs_list, dtype=np.float32)
        allocated_total = idx
    else:
        allocated_total = total_alloc
    states, diffs, context = _finalize_analysis_arrays(
        states, diffs, max(allocated_total, idx), idx, tracker, last_gray)
    return states, diffs, context


def analyze_video(video_path: str, configs: dict, thresholds: dict,
                  proc_res: tuple, batch_size: int, n_threads: int,
                  progress_cb=None,
                  decode_backend: str = DECODE_BACKEND_OPENCV,
                  ffmpeg_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    states, diffs, _ = analyze_video_with_context(
        video_path, configs, thresholds, proc_res, batch_size, n_threads,
        progress_cb, decode_backend=decode_backend, ffmpeg_path=ffmpeg_path)
    return states, diffs
