# analyzer.py —— 模板加载 + 帧状态识别（无 GUI 依赖）

import cv2
import numpy as np
import os
import shutil
import subprocess
import concurrent.futures
import multiprocessing
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any
from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)

_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0
_GPU_PROBE_CACHE: dict[str, bool] = {}
_GPU_PROBE_LOCK = threading.Lock()


@dataclass
class _FFmpegPipe:
    process: subprocess.Popen
    stderr_file: Any

    @property
    def stdin(self):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin 不可用")
        return self.process.stdin

# ---------------------------------------------------------------
#  模板加载（带预缩放缓存）
# ---------------------------------------------------------------

TEMPLATE_DIRS = {
    'pause': {'ref_dir': 'templates_pause', 'source_dir': 'source_images_pause'},
    'speed_1x': {'ref_dir': 'templates_1x', 'source_dir': 'source_images_1x'},
    'speed_2x': {'ref_dir': 'templates_2x', 'source_dir': 'source_images_2x'},
    'speed_0_2x': {'ref_dir': 'templates_play', 'source_dir': 'source_images_play'},
}

IMG_EXTS = ('.png', '.jpg', '.bmp', '.jpeg')


def load_templates(proc_res: tuple = (400, 225)) -> tuple[dict, int]:
    configs: dict[str, list] = {k: [] for k in TEMPLATE_DIRS}
    total = 0

    for ctype, dirs in TEMPLATE_DIRS.items():
        src_dir, ref_dir = dirs['source_dir'], dirs['ref_dir']
        if not os.path.exists(src_dir) or not os.path.exists(ref_dir): continue

        src_files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(IMG_EXTS)]
        if not src_files or not ref_files: continue

        src_img = cv2.imread(os.path.join(src_dir, src_files[0]), cv2.IMREAD_GRAYSCALE)
        if src_img is None: continue
        sh, sw = src_img.shape

        for rf in ref_files:
            ref_img = cv2.imread(os.path.join(ref_dir, rf), cv2.IMREAD_GRAYSCALE)
            if ref_img is None: continue
            rh, rw = ref_img.shape

            res = cv2.matchTemplate(src_img, ref_img, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            rx, ry = max_loc
            _, mask = cv2.threshold(ref_img, 10, 255, cv2.THRESH_BINARY)

            scale_x, scale_y = proc_res[0] / sw, proc_res[1] / sh
            ext = 2.0
            erx = max(0, int(rx * scale_x - rw * scale_x * (ext - 1) / 2))
            ery = max(0, int(ry * scale_y - rh * scale_y * (ext - 1) / 2))
            tw, th = max(1, int(rw * scale_x)), max(1, int(rh * scale_y))

            configs[ctype].append({
                'roi_orig': (rx, ry, rw, rh),
                'source_res': (sw, sh),
                'cached_proc_res': proc_res,
                'cached_roi': (erx, ery, int(rw * scale_x * ext), int(rh * scale_y * ext)),
                'cached_t': cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA),
                'cached_m': cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST),
            })
            total += 1
    return configs, total


# ---------------------------------------------------------------
#  单帧匹配
# ---------------------------------------------------------------

def _get_best_score(gray_frame: np.ndarray, templates: list, proc_res: tuple) -> float:
    max_score = -1.0
    fh, fw = gray_frame.shape
    for t in templates:
        erx, ery, erw, erh = t['cached_roi']
        t_r, m_r = t['cached_t'], t['cached_m']
        erw, erh = min(fw - erx, erw), min(fh - ery, erh)

        if erw <= 0 or erh <= 0: continue
        roi = gray_frame[ery:ery + erh, erx:erx + erw]
        if roi.shape[0] < t_r.shape[0] or roi.shape[1] < t_r.shape[1]: continue

        res = cv2.matchTemplate(roi, t_r, cv2.TM_CCOEFF_NORMED, mask=m_r)
        _, score, _, _ = cv2.minMaxLoc(res)
        if np.isfinite(score): max_score = max(max_score, score)
    return max_score


def _classify_gray(gray: np.ndarray, configs: dict,
                   thresholds: dict, proc_res: tuple) -> int:
    if configs['pause'] and _get_best_score(gray, configs['pause'], proc_res) >= thresholds['pause']:
        return FRAME_TYPE_PAUSE
    x1s = _get_best_score(gray, configs['speed_1x'], proc_res) if configs['speed_1x'] else -1.0
    x2s = _get_best_score(gray, configs['speed_2x'], proc_res) if configs['speed_2x'] else -1.0
    if x1s >= thresholds['speed_1x'] and x1s > x2s: return FRAME_TYPE_1X
    if x2s >= thresholds['speed_2x'] and x2s > x1s: return FRAME_TYPE_2X
    if configs['speed_0_2x'] and _get_best_score(gray, configs['speed_0_2x'], proc_res) >= thresholds['speed_0_2x']:
        return FRAME_TYPE_0_2X
    return FRAME_TYPE_NORMAL


# ---------------------------------------------------------------
#  子进程全局状态
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
#  First-pass pause-boundary context (skip second VideoCapture scan)
# ---------------------------------------------------------------

ANALYSIS_CONTEXT_VERSION = 1


class _BoundaryTracker:
    """Collect one scalar boundary record per pause run during ordered commit.

    Keeps at most two small gray frames: prev_gray (also used for diffs) and
    the open pause run's before-gray. Never grows with video duration.
    """

    def __init__(self, total: int):
        self._total = int(total)
        self.records: list[dict] = []
        self._open_start: int | None = None
        self._before_gray: np.ndarray | None = None
        self._skipped_records = 0

    def observe(
        self,
        idx: int,
        gray: np.ndarray,
        state: int,
        prev_gray: np.ndarray | None,
    ) -> None:
        is_pause = int(state) == FRAME_TYPE_PAUSE
        if self._open_start is None:
            if is_pause:
                self._open_start = idx
                # before = max(0, start-1): previous gray, or this frame if start==0
                self._before_gray = gray if idx == 0 else prev_gray
        elif not is_pause:
            self._close_run(end=idx - 1, after_idx=idx, after_gray=gray)

    def finish_with_last(self, decoded: int, last_gray: np.ndarray | None) -> None:
        if self._open_start is None:
            return
        end = decoded - 1
        after_idx = min(self._total - 1, end + 1)
        if after_idx <= end:
            if last_gray is not None:
                self._close_run(end=end, after_idx=after_idx, after_gray=last_gray)
                return
        # Early EOF / missing after frame → context incomplete
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
        self.records.append(
            {
                "start": start,
                "end": int(end),
                "before_index": int(before_idx),
                "after_index": int(after_idx),
                "diff": diff,
            }
        )
        self._open_start = None
        self._before_gray = None

    @property
    def skipped(self) -> int:
        return self._skipped_records


def _make_analysis_context(
    total: int, decoded: int, records: list[dict], complete: bool
) -> dict:
    return {
        "version": ANALYSIS_CONTEXT_VERSION,
        "complete": bool(complete),
        "frame_count": int(total),
        "decoded_frame_count": int(decoded),
        "pause_boundary_diffs": list(records),
    }


def context_records_for_pauses(analysis_context, pauses: list, total: int):
    """Return records if the whole context is usable; else None.

    All-or-nothing: never mix cached and rescanned boundaries.
    """
    if not isinstance(analysis_context, dict):
        return None
    try:
        if analysis_context.get("version") != ANALYSIS_CONTEXT_VERSION:
            return None
        if analysis_context.get("complete") is not True:
            return None
        if int(analysis_context.get("frame_count", -1)) != int(total):
            return None
        if int(analysis_context.get("decoded_frame_count", -1)) != int(total):
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
            if before != max(0, start - 1) or after != min(int(total) - 1, end + 1):
                return None
            if not np.isfinite(diff):
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return records


def _finalize_analysis_arrays(
    states: np.ndarray,
    diffs: np.ndarray,
    allocated_total: int,
    decoded: int,
    tracker: _BoundaryTracker | None,
    last_gray: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if decoded < allocated_total:
        states = states[:decoded]
        diffs = diffs[:decoded]
    if tracker is not None:
        tracker.finish_with_last(decoded, last_gray)
        complete = decoded == allocated_total and tracker.skipped == 0
        # frame_count for consumers is the states length after trim
        context = _make_analysis_context(
            len(states), decoded, tracker.records, complete
        )
    else:
        context = _make_analysis_context(len(states), decoded, [], False)
    return states, diffs, context


# ---------------------------------------------------------------
#  分析解码后端（生产可选 A_PT）
# ---------------------------------------------------------------

# Default remains OpenCV for compatibility. Optional:
#   ffmpeg_sw_passthrough == verified A_PT path
#   (software decode + scale=area + gray + -fps_mode passthrough)
DECODE_BACKEND_OPENCV = "opencv"
DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH = "ffmpeg_sw_passthrough"
_DECODE_BACKEND_ALIASES = {
    "opencv": DECODE_BACKEND_OPENCV,
    "cv2": DECODE_BACKEND_OPENCV,
    "default": DECODE_BACKEND_OPENCV,
    "ffmpeg_sw_passthrough": DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH,
    "ffmpeg_sw_gray_passthrough": DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH,
    "a_pt": DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH,
}


def normalize_decode_backend(decode_backend: str | None) -> str:
    key = (decode_backend or DECODE_BACKEND_OPENCV).strip().lower()
    if key not in _DECODE_BACKEND_ALIASES:
        raise ValueError(
            f"unknown decode_backend={decode_backend!r}; "
            f"allowed={sorted(set(_DECODE_BACKEND_ALIASES.values()))}"
        )
    return _DECODE_BACKEND_ALIASES[key]


def resolve_ffmpeg_path(ffmpeg_path: str | None = None) -> str:
    """Resolve FFmpeg binary: explicit path → PATH → imageio_ffmpeg (if installed)."""
    if ffmpeg_path:
        p = os.path.expanduser(str(ffmpeg_path).strip())
        if p and os.path.isfile(p):
            return os.path.abspath(p)
        found = shutil.which(p) if p else None
        if found:
            return found
        raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    raise FileNotFoundError(
        "FFmpeg not found on PATH and imageio_ffmpeg is unavailable; "
        "set an explicit ffmpeg path in settings."
    )


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _ffmpeg_sw_passthrough_cmd(
    ffmpeg: str, video_path: str, frames: int, proc_res: tuple[int, int]
) -> list[str]:
    """Verified A_PT command graph (do not add extra timestamp/sync flags)."""
    pw, ph = int(proc_res[0]), int(proc_res[1])
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        video_path,
        "-an",
        "-frames:v",
        str(int(frames)),
        "-vf",
        f"scale={pw}:{ph}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-fps_mode",
        "passthrough",
        "pipe:1",
    ]


def _analyze_video_opencv(
    video_path: str,
    configs: dict,
    thresholds: dict,
    proc_res: tuple,
    batch_size: int,
    n_threads: int,
    progress_cb=None,
    want_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    states = np.zeros(max(0, total), dtype=np.int8)
    diffs = np.zeros(max(0, total), dtype=np.float32)
    n_workers = min(n_threads, multiprocessing.cpu_count())
    pw, ph = proc_res
    tracker = _BoundaryTracker(total if total > 0 else 1) if want_context else None
    # If total unknown/0, still allow reading until EOF with growable lists.
    use_dynamic = total <= 0
    if use_dynamic:
        states_list: list[int] = []
        diffs_list: list[float] = []
        tracker = _BoundaryTracker(10**9) if want_context else None

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(configs, thresholds, proc_res)) as ex:

        idx = 0
        prev_gray = None
        last_gray = None
        allocated_total = total

        while True:
            batch_grays = []
            batch_indices = []
            batch_prev = []

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(
                    cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
                batch_grays.append(gray)
                batch_indices.append(idx)
                batch_prev.append(prev_gray)

                if not use_dynamic:
                    if prev_gray is not None and idx < len(diffs):
                        diffs[idx] = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                else:
                    diffs_list.append(
                        0.0
                        if prev_gray is None
                        else float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                    )
                prev_gray = gray
                last_gray = gray
                idx += 1

            if not batch_grays:
                break

            chunk = max(4, len(batch_grays) // (n_workers * 2))
            results = list(ex.map(_worker_classify_gray, batch_grays, chunksize=chunk))

            for i, s, g, pg in zip(batch_indices, results, batch_grays, batch_prev):
                if use_dynamic:
                    states_list.append(int(s))
                else:
                    states[i] = s
                if tracker is not None:
                    tracker.observe(i, g, int(s), pg)

            if progress_cb:
                denom = max(1, allocated_total if allocated_total > 0 else idx)
                progress_cb((idx / denom) * 0.5)

    cap.release()
    if use_dynamic:
        states = np.asarray(states_list, dtype=np.int8)
        diffs = np.asarray(diffs_list, dtype=np.float32)
        allocated_total = idx
        decoded = idx
    else:
        decoded = idx
        # allocated_total from metadata; decoded may be smaller
    if want_context:
        states, diffs, context = _finalize_analysis_arrays(
            states, diffs, max(allocated_total, decoded), decoded, tracker, last_gray
        )
        return states, diffs, context
    if decoded < len(states):
        states = states[:decoded]
        diffs = diffs[:decoded]
    return states, diffs, None


def _analyze_video_ffmpeg_sw_passthrough(
    video_path: str,
    configs: dict,
    thresholds: dict,
    proc_res: tuple,
    batch_size: int,
    n_threads: int,
    progress_cb=None,
    ffmpeg_path: str | None = None,
    want_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Production A_PT: FFmpeg software gray@proc_res with output fps_mode=passthrough."""
    ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    pw, ph = int(proc_res[0]), int(proc_res[1])
    if pw <= 0 or ph <= 0:
        raise ValueError(f"invalid proc_res={proc_res}")

    # Frame count oracle matches existing OpenCV path (same CAP_PROP_FRAME_COUNT).
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        raise RuntimeError(f"cannot determine frame count for {video_path}")

    bpf = pw * ph
    cmd = _ffmpeg_sw_passthrough_cmd(ffmpeg, video_path, total, (pw, ph))
    # Wall budget scales with length; 180s is only enough for short clips.
    timeout_s = max(180.0, 120.0 + float(total) * 0.12)

    states = np.zeros(total, dtype=np.int8)
    diffs = np.zeros(total, dtype=np.float32)
    n_workers = min(n_threads, multiprocessing.cpu_count())
    tracker = _BoundaryTracker(total) if want_context else None

    stderr_file = tempfile.TemporaryFile()
    proc: subprocess.Popen | None = None
    got = 0
    prev_gray = None
    last_gray = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            creationflags=_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if proc.stdout is None:
            raise RuntimeError("FFmpeg stdout pipe unavailable")

        deadline = time.perf_counter() + timeout_s

        def _stderr_tail() -> str:
            try:
                stderr_file.flush()
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 4096), os.SEEK_SET)
                return stderr_file.read().decode("utf-8", errors="replace")
            except Exception:
                return ""

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(configs, thresholds, proc_res),
        ) as ex:
            while got < total:
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"FFmpeg A_PT timed out after {timeout_s:.1f}s at frame {got}/{total}"
                    )

                batch_n = min(int(batch_size), total - got)
                batch_grays = []
                batch_indices = []
                batch_prev = []
                eof = False
                for _ in range(batch_n):
                    raw = _read_exact(proc.stdout, bpf)
                    if len(raw) == 0:
                        # Clean EOF: CAP_PROP_FRAME_COUNT is often slightly high.
                        eof = True
                        break
                    if len(raw) != bpf:
                        raise RuntimeError(
                            f"FFmpeg A_PT partial frame at index {got}/{total}; "
                            f"got_bytes={len(raw)} expected={bpf}; "
                            f"stderr={_stderr_tail()[:500]}"
                        )
                    gray = np.frombuffer(raw, dtype=np.uint8).reshape((ph, pw)).copy()
                    batch_grays.append(gray)
                    batch_indices.append(got)
                    batch_prev.append(prev_gray)
                    if prev_gray is not None:
                        diffs[got] = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                    prev_gray = gray
                    last_gray = gray
                    got += 1

                if batch_grays:
                    chunk = max(4, len(batch_grays) // (n_workers * 2))
                    results = list(
                        ex.map(_worker_classify_gray, batch_grays, chunksize=chunk)
                    )
                    for i, s, g, pg in zip(
                        batch_indices, results, batch_grays, batch_prev
                    ):
                        states[i] = s
                        if tracker is not None:
                            tracker.observe(i, g, int(s), pg)
                    if progress_cb:
                        progress_cb((got / total) * 0.5)

                if eof:
                    break

        # Drain/wait FFmpeg
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            rc = proc.wait(timeout=max(30.0, min(120.0, timeout_s * 0.1)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError("FFmpeg A_PT did not exit after stdout closed")
        if got == 0:
            raise RuntimeError(
                f"FFmpeg A_PT produced 0 frames; exit={rc}; stderr={_stderr_tail()[:500]}"
            )
        if got < total:
            # Metadata overstated frame count (common). Accept actual stream length.
            print(
                f"[analyze] A_PT decoded {got}/{total} frames "
                f"(container metadata may overstate FRAME_COUNT); exit={rc}",
                flush=True,
            )
        elif rc not in (0, None) and got != total:
            raise RuntimeError(
                f"FFmpeg A_PT exit={rc} after {got}/{total} frames; stderr={_stderr_tail()[:500]}"
            )
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
        try:
            stderr_file.close()
        except Exception:
            pass

    if want_context:
        states, diffs, context = _finalize_analysis_arrays(
            states, diffs, total, got, tracker, last_gray
        )
        return states, diffs, context
    if got != total:
        states = states[:got]
        diffs = diffs[:got]
    return states, diffs, None


# ---------------------------------------------------------------
#  批量分析整段视频
# ---------------------------------------------------------------

def analyze_video(video_path: str, configs: dict, thresholds: dict,
                  proc_res: tuple, batch_size: int, n_threads: int,
                  progress_cb=None,
                  decode_backend: str = DECODE_BACKEND_OPENCV,
                  ffmpeg_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Analyze full video into states/diffs (compatible two-item API).

    decode_backend:
      - "opencv" (default): production baseline BGR→INTER_AREA→GRAY
      - "ffmpeg_sw_passthrough" / "a_pt": verified A_PT software gray path

    For skipping the second boundary VideoCapture scan, prefer
    analyze_video_with_context(...) and pass the context to build_segments.
    """
    states, diffs, _ = analyze_video_with_context(
        video_path,
        configs,
        thresholds,
        proc_res,
        batch_size,
        n_threads,
        progress_cb,
        decode_backend=decode_backend,
        ffmpeg_path=ffmpeg_path,
    )
    return states, diffs


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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Three-item API: (states, diffs, analysis_context).

    analysis_context is JSON-safe (no frame pixels). When complete, pass it to
    build_segments(..., analysis_context=context) to skip the second boundary
    VideoCapture scan. Incomplete/mismatched context is rejected wholesale.
    """
    backend = normalize_decode_backend(decode_backend)
    if backend == DECODE_BACKEND_OPENCV:
        states, diffs, context = _analyze_video_opencv(
            video_path,
            configs,
            thresholds,
            proc_res,
            batch_size,
            n_threads,
            progress_cb,
            want_context=True,
        )
        return states, diffs, context or _make_analysis_context(len(states), len(states), [], False)
    if backend == DECODE_BACKEND_FFMPEG_SW_PASSTHROUGH:
        states, diffs, context = _analyze_video_ffmpeg_sw_passthrough(
            video_path,
            configs,
            thresholds,
            proc_res,
            batch_size,
            n_threads,
            progress_cb,
            ffmpeg_path=ffmpeg_path,
            want_context=True,
        )
        return states, diffs, context or _make_analysis_context(len(states), len(states), [], False)
    raise ValueError(f"unsupported decode_backend={backend!r}")


# ---------------------------------------------------------------
#  段落提取 + 内部操作细粒度帧差分析 + 外部边界差分
# ---------------------------------------------------------------

def _analyze_pause_mask(s_i: int, e_i: int, diffs: np.ndarray, still_frames: int, motion_thresh: float):
    seg_len = e_i - s_i + 1
    if seg_len <= 0:
        return np.zeros(0, dtype=np.uint8), 'all'

    active_mask = np.zeros(seg_len, dtype=bool)
    active_mask[0] = False

    for k in range(1, seg_len):
        idx = s_i + k
        if diffs[idx] > motion_thresh:
            active_mask[k] = True
            active_mask[k - 1] = True

    del_mask = np.zeros(seg_len, dtype=np.uint8)

    if seg_len > 0:
        runs = []
        curr_val = active_mask[0]
        start = 0
        for i in range(1, seg_len):
            if active_mask[i] != curr_val:
                runs.append((curr_val, start, i - 1))
                curr_val = active_mask[i]
                start = i
        runs.append((curr_val, start, seg_len - 1))

        has_active = any(val for val, s, e in runs)

        if not has_active:
            if seg_len > 2 * still_frames:
                del_mask[still_frames: seg_len - still_frames] = 1
            return del_mask, 'auto'

        for val, s, e in runs:
            if not val:
                run_len = e - s + 1
                if run_len > still_frames:
                    if s == 0:
                        keep_start = e - still_frames + 1
                        del_mask[s:keep_start] = 1
                    elif e == seg_len - 1:
                        keep_end = s + still_frames - 1
                        del_mask[keep_end + 1:e + 1] = 1
                    else:
                        half = still_frames // 2
                        other_half = still_frames - half
                        del_mask[s + half: e - other_half + 1] = 1

    return del_mask, 'auto'


def build_segments(states: np.ndarray, diffs: np.ndarray, video_path: str, proc_res: tuple,
                   compare_cfg: dict, fps: float, progress_cb=None, *,
                   analysis_context=None) -> tuple[list, list]:
    total = len(states)
    pauses = []
    speeds = []

    still_time = compare_cfg.get('still_time_thresh', 0.1)
    motion_thresh = compare_cfg.get('motion_thresh', 2.0)
    boundary_thresh = compare_cfg.get('boundary_thresh', 5.0)
    still_frames = max(2, int(fps * still_time))

    # 1. 基础分段
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
                'boundary_diff': 0.0  # 预占位，稍后计算
            })
            if progress_cb: progress_cb(0.5 + (e_i / max(1, total)) * 0.25)

        elif curr in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
            speeds.append({'type': curr, 'start': s_i, 'end': e_i})

    # 2. 暂停边界差分
    #    完整的第一遍 analysis_context 可直接提供每个暂停段的 boundary_diff，
    #    此时不再打开第二个 VideoCapture。上下文不可用时整体拒绝并回退到
    #    原始全量顺序 grab 扫描；绝不混用缓存与重扫结果。
    context_records = None
    if pauses and analysis_context is not None:
        context_records = context_records_for_pauses(analysis_context, pauses, total)

    if pauses and context_records is not None:
        for p, rec in zip(pauses, context_records):
            diff = float(rec['diff'])
            p['boundary_diff'] = diff
            # 严格小于：等于阈值不强制全删
            if diff < boundary_thresh:
                p['mode'] = 'all'
        if progress_cb:
            progress_cb(1.0)
        return pauses, speeds

    if pauses:
        cap = cv2.VideoCapture(video_path)
        # 获取所有目标帧索引，去重并排序
        target_indices = sorted(list(set([max(0, p['start'] - 1) for p in pauses] +
                                         [min(total - 1, p['end'] + 1) for p in pauses])))
        target_frames = {}
        curr_idx = 0
        for target in target_indices:
            # 顺序 grab 直到目标帧，这是最稳定精准读取特定帧的方法
            while curr_idx < target:
                cap.grab()
                curr_idx += 1
            ret, frame = cap.read()
            if ret:
                target_frames[target] = cv2.cvtColor(cv2.resize(frame, proc_res, interpolation=cv2.INTER_AREA),
                                                     cv2.COLOR_BGR2GRAY)
            curr_idx += 1
        cap.release()

        # 根据边界差分改写判定
        for p in pauses:
            b_idx = max(0, p['start'] - 1)
            a_idx = min(total - 1, p['end'] + 1)
            if b_idx in target_frames and a_idx in target_frames:
                diff = float(cv2.mean(cv2.absdiff(target_frames[b_idx], target_frames[a_idx]))[0])
                p['boundary_diff'] = diff
                # 核心机制：一旦前后差距过小，不管之前算出来动作多大，一律强制“全删”
                if diff < boundary_thresh:
                    p['mode'] = 'all'
        if progress_cb:
            progress_cb(1.0)

    return pauses, speeds


# ---------------------------------------------------------------
#  导出辅助
# ---------------------------------------------------------------

def _speedup_mask(states: np.ndarray, frame_type: int, factor: int,
                  exclude_mask: np.ndarray) -> np.ndarray:
    total = len(states)
    type_mask = (states == frame_type) & ~exclude_mask

    if not type_mask.any(): return np.zeros(total, dtype=bool)

    cumsum = np.cumsum(type_mask)
    shifted = np.empty(total, dtype=bool)
    shifted[0] = False
    shifted[1:] = type_mask[:-1]
    seg_starts = np.where(type_mask & ~shifted)[0]

    offsets = np.zeros(total, dtype=np.int64)
    for s in seg_starts:
        offsets[s:] = cumsum[s - 1] if s > 0 else 0

    local_cnt = np.where(type_mask, cumsum - offsets, 0)

    if factor == 2:
        return type_mask & (local_cnt % 2 == 0)
    return type_mask & (local_cnt % factor != 1)


def build_delete_set(total: int, states: np.ndarray,
                     pause_segments: list, speed_segments: list,
                     clip_segments: list,
                     speedup_1x: bool, speedup_02: bool,
                     speedup_02_factor: int) -> np.ndarray:
    del_mask = np.zeros(total, dtype=bool)

    for seg in pause_segments:
        s, e = seg['start'], seg['end']
        mode = seg.get('mode', 'auto')
        if mode == 'all':
            del_mask[s:e + 1] = True
        elif mode == 'auto' and 'local_del_mask' in seg:
            m = seg['local_del_mask']
            # 1 为自动删除，2 为人工强制删除
            del_mask[s:e + 1] = (m == 1) | (m == 2)

    for seg in clip_segments:
        s, e = seg['start'], seg['end']
        ki, ko = seg['keep_in'], seg['keep_out']
        if ki > ko:
            del_mask[s:e + 1] = True
        else:
            if ki > s:   del_mask[s:ki] = True
            if ko < e:   del_mask[ko + 1:e + 1] = True

    if speedup_1x:
        del_mask |= _speedup_mask(states, FRAME_TYPE_1X, 2, del_mask)

    if speedup_02 and speedup_02_factor > 1:
        del_mask |= _speedup_mask(states, FRAME_TYPE_0_2X, speedup_02_factor, del_mask)

    return del_mask


def _kept_frame_ranges(to_del: np.ndarray) -> list[tuple[int, int]]:
    """将删除掩码转换为左闭右开的保留帧区间。"""
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


def _has_audio_stream(video_path: str, ffmpeg_path: str | None = None) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
                check=False, capture_output=True, text=True, timeout=15,
                creationflags=_NO_WINDOW)
            return bool(result.stdout.strip())
        except Exception:
            pass

    try:
        ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video_path,
             "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-"],
            check=False, capture_output=True, timeout=15,
            creationflags=_NO_WINDOW)
        return result.returncode == 0
    except Exception:
        return False


def _gpu_encoder_probe_args(enc: str) -> list[str]:
    q = "24"
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", "p4", "-cq", q]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-global_quality", q]
    if enc == "h264_amf":
        return ["-c:v", enc, "-usage", "transcoding", "-quality", "speed",
                "-rc", "cqp", "-qp_i", q, "-qp_p", q]
    return ["-c:v", enc]


def _gpu_encoder_works(enc: str, timeout: float = 5, ffmpeg_path: str | None = None) -> bool:
    # 单飞探测：探测全程持锁，避免多线程并发启动多个 ffmpeg 探测同一编码器，
    # 进而在 GPU 资源紧张时因并发抢占产生假阴性。
    with _GPU_PROBE_LOCK:
        cache_key = f"{enc}|{ffmpeg_path or ''}"
        cached = _GPU_PROBE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
        except FileNotFoundError:
            _GPU_PROBE_CACHE[cache_key] = False
            return False

        try:
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=0.1",
                 *_gpu_encoder_probe_args(enc), "-f", "null", "-"],
                check=True, capture_output=True, timeout=timeout,
                creationflags=_NO_WINDOW)
        except FileNotFoundError:
            # FFmpeg 不在 PATH：确定不可用，缓存 False。
            _GPU_PROBE_CACHE[cache_key] = False
            return False
        except subprocess.CalledProcessError:
            # 编码器确实初始化失败（ffmpeg 正常退出且给出错误）：缓存 False。
            # 但超时（TimeoutExpired）等瞬时失败不在此分支，不缓存。
            _GPU_PROBE_CACHE[cache_key] = False
            return False
        except Exception:
            # 瞬时失败（超时、GPU 被占用、驱动初始化等）：不缓存，下次重新探测，
            # 避免一次偶发失败永久禁用该编码器。
            return False

        _GPU_PROBE_CACHE[cache_key] = True
        return True


def list_ffmpeg_gpu_encoders(ffmpeg_path: str | None = None) -> list[str]:
    try:
        ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
        output = subprocess.check_output(
            [ffmpeg, "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
            creationflags=_NO_WINDOW)
    except Exception:
        return []

    # 顺序即优先级：nvenc（NVIDIA）通常最快最稳，其次是 qsv（Intel）、amf（AMD）。
    # 纯 AMD 机型误选不可用 nvenc 的问题（issue #3）已由 _gpu_encoder_works 的实测探测解决，
    # 不依赖把 amf 提前；探测通过的编码器按此序取首个即可。
    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"]
    return [enc for enc in candidates if enc in output]


def list_working_gpu_encoders(ffmpeg_path: str | None = None) -> list[str]:
    return [
        enc
        for enc in list_ffmpeg_gpu_encoders(ffmpeg_path)
        if _gpu_encoder_works(enc, ffmpeg_path=ffmpeg_path)
    ]


def _pick_gpu_encoder(ffmpeg_path: str | None = None) -> str | None:
    working = list_working_gpu_encoders(ffmpeg_path)
    return working[0] if working else None


def _resolve_gpu_encoder(
    gpu_encoder: str | None, ffmpeg_path: str | None = None
) -> str | None:
    selected = (gpu_encoder or "").strip()
    if selected:
        return selected if _gpu_encoder_works(selected, ffmpeg_path=ffmpeg_path) else None
    return _pick_gpu_encoder(ffmpeg_path)


def _encoder_cmd_args(enc: str, quality: int) -> list[str]:
    q = max(0, min(10, int(quality)))
    qp = str(18 + (10 - q))
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", "p4", "-cq", qp]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-global_quality", qp]
    if enc == "h264_amf":
        return ["-c:v", enc, "-usage", "transcoding", "-quality", "speed",
                "-rc", "cqp", "-qp_i", qp, "-qp_p", qp]
    return ["-c:v", enc, "-q:v", qp]


def _video_encoder_args(
    quality: int,
    use_gpu: bool,
    gpu_encoder: str,
    ffmpeg_path: str | None = None,
) -> list[str]:
    q = max(0, min(10, int(quality)))
    if use_gpu:
        enc = _resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg_path)
        if enc:
            return _encoder_cmd_args(enc, q) + ["-pix_fmt", "yuv420p", "-threads", "0"]
    crf = int(round(28 - q))
    return ["-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", "0"]


def _export_ranges_with_ffmpeg_filters(
        video_path: str, output_path: str, ranges: list[tuple[int, int]],
        fps: float, quality: int, use_gpu: bool, gpu_encoder: str,
        include_audio: bool, progress_cb=None,
        ffmpeg_path: str | None = None) -> bool:
    """使用 FFmpeg trim/concat 快速导出左闭右开的帧区间。

    滤镜图通过 -filter_complex_script 写入文件（不走命令行），ffmpeg concat
    可处理上千段，因此不再对段数设上限；空 ranges 时直接回退逐帧路径。
    """
    if not ranges:
        return False

    try:
        ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        return False

    has_audio = include_audio and _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filter_file = os.path.join(tmpdir, "filter.txt")
        lines = []
        concat_inputs = []
        for idx, (start, end) in enumerate(ranges):
            lines.append(
                f"[0:v]trim=start_frame={start}:end_frame={end},"
                f"setpts=PTS-STARTPTS[v{idx}]")
            concat_inputs.append(f"[v{idx}]")
            if has_audio:
                lines.append(
                    f"[0:a]atrim=start={start / fps:.9f}:end={end / fps:.9f},"
                    f"asetpts=PTS-STARTPTS[a{idx}]")
                concat_inputs.append(f"[a{idx}]")

        if has_audio:
            lines.append(
                "".join(concat_inputs)
                + f"concat=n={len(ranges)}:v=1:a=1[outv][outa]")
        else:
            lines.append(
                "".join(concat_inputs)
                + f"concat=n={len(ranges)}:v=1:a=0[outv]")

        with open(filter_file, "w", encoding="utf-8") as handle:
            handle.write(";\n".join(lines))

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", video_path, "-filter_complex_script", filter_file,
            "-map", "[outv]",
        ]
        if has_audio:
            cmd += ["-map", "[outa]"]
        cmd += _video_encoder_args(quality, use_gpu, gpu_encoder, ffmpeg_path=ffmpeg)
        cmd += ["-c:a", "aac"] if has_audio else ["-an"]
        cmd.append(output_path)

        try:
            if progress_cb:
                progress_cb(0.01, 0)
            subprocess.run(cmd, check=True, capture_output=True,
                           timeout=1800, creationflags=_NO_WINDOW)
            if progress_cb:
                progress_cb(1.0, sum(end - start for start, end in ranges))
            return True
        except subprocess.CalledProcessError as exc:
            # 快速滤镜路径失败：删除 ffmpeg 写了一半的输出，回退到逐帧路径。
            # 同时打印 ffmpeg 的真实 stderr，避免“静默变慢 + 无诊断”。
            err = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
            if err:
                print(f"[analyzer] 快速滤镜导出失败，回退逐帧路径。ffmpeg stderr: {err}")
            if os.path.isfile(output_path):
                os.remove(output_path)
            return False
        except Exception as exc:
            # 超时/其它异常同样回退，但打印原因，不再完全静默。
            print(f"[analyzer] 快速滤镜导出异常，回退逐帧路径: {exc}")
            if os.path.isfile(output_path):
                os.remove(output_path)
            return False


def _mux_audio_for_ranges(video_path: str, video_only_path: str,
                          output_path: str, ranges: list[tuple[int, int]],
                          fps: float, ffmpeg_path: str | None = None):
    if not _has_audio_stream(video_path, ffmpeg_path=ffmpeg_path):
        os.replace(video_only_path, output_path)
        return

    ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        filter_file = os.path.join(tmpdir, "audio-filter.txt")
        lines = []
        labels = []
        for idx, (start, end) in enumerate(ranges):
            lines.append(
                f"[0:a]atrim=start={start / fps:.9f}:end={end / fps:.9f},"
                f"asetpts=PTS-STARTPTS[a{idx}]")
            labels.append(f"[a{idx}]")
        lines.append("".join(labels) + f"concat=n={len(ranges)}:v=0:a=1[outa]")
        with open(filter_file, "w", encoding="utf-8") as handle:
            handle.write(";\n".join(lines))

        try:
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                 "-i", video_path, "-i", video_only_path,
                 "-filter_complex_script", filter_file,
                 "-map", "1:v:0", "-map", "[outa]",
                 "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
                check=True, capture_output=True, timeout=1800, creationflags=_NO_WINDOW)
        except Exception as exc:
            # 混流失败：删除可能被 ffmpeg 截断/写半的损坏输出，避免残留假成品。
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            stderr = b""
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = exc.stderr or b""
            if stderr:
                raise RuntimeError(
                    f"音频混流失败: {stderr.decode('utf-8', errors='ignore')}"
                ) from exc
            raise


def export_video(video_path: str, output_path: str, to_del,
                 fps: float, quality: int, progress_cb=None,
                 use_gpu: bool = False, gpu_encoder: str = "",
                 ffmpeg_path: str | None = None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if isinstance(to_del, set):
        mask = np.zeros(total, dtype=bool)
        for idx in to_del:
            if 0 <= idx < total:
                mask[idx] = True
        to_del = mask

    ranges = _kept_frame_ranges(to_del)
    written_fast = sum(end - start for start, end in ranges)
    if not ranges:
        cap.release()
        raise RuntimeError("没有可导出的帧")

    try:
        ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        ffmpeg_bin = None

    if ffmpeg_bin and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, ranges, fps, quality,
            use_gpu, gpu_encoder, True, progress_cb, ffmpeg_path=ffmpeg_bin):
        cap.release()
        return written_fast, total

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
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_bin)
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
                if gap > 30:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_keep)
                else:
                    for _ in range(gap):
                        cap.read()
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
                progress_cb(idx / total, written)
    finally:
        cap.release()
        try:
            _close_video_writer(writer_kind, writer, ffmpeg_proc)
        except Exception:
            # 编码器关闭失败（ffmpeg 非零退出/超时）：仍需清理临时文件再向上抛错。
            if use_ffmpeg and video_only_path != output_path and os.path.isfile(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
            raise

    if use_ffmpeg:
        try:
            _mux_audio_for_ranges(
                video_path, video_only_path, output_path, ranges, fps,
                ffmpeg_path=ffmpeg_bin)
        finally:
            # 混流结束（无论成功失败）都要回收 video-only 临时文件。
            if video_only_path != output_path and os.path.isfile(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
    return written, total


def export_ranges(video_path: str, output_path: str, ranges: list,
                  fps: float, quality: int, progress_cb=None,
                  use_gpu: bool = False, gpu_encoder: str = "",
                  ffmpeg_path: str | None = None):
    if not ranges:
        return 0, 0

    exclusive_ranges = [(start, end + 1) for start, end in ranges]
    total_frames_to_export = sum(end - start for start, end in exclusive_ranges)
    try:
        ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        ffmpeg_bin = None
    if ffmpeg_bin and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, exclusive_ranges, fps, quality,
            use_gpu, gpu_encoder, False, progress_cb, ffmpeg_path=ffmpeg_bin):
        return total_frames_to_export, total_frames_to_export

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ranges[0][0])
    ret, sample = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, ranges[0][0])

    writer_kind = None
    ffmpeg_proc = None
    writer = None
    if ffmpeg_bin:
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            output_path, fps, w, h, quality,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_bin)
        writer_kind = "ffmpeg"
    else:
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                        quality=quality, pixelformat='yuv420p')
            writer_kind = "imageio"
        except ImportError:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            writer_kind = "cv2"

    written = 0
    try:
        for start, end in ranges:
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(start, end + 1):
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
                if progress_cb and written % 30 == 0:
                    progress_cb(written / total_frames_to_export, written)
    finally:
        cap.release()
        _close_video_writer(writer_kind, writer, ffmpeg_proc)
    return written, total_frames_to_export


def _open_ffmpeg_pipe_writer(output_path: str, fps: float, w: int, h: int, quality: int, use_gpu: bool,
                             gpu_encoder: str = "", ffmpeg_path: str | None = None):
    q = max(0, min(10, int(quality)))
    crf = int(round(28 - q))
    ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    base_cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
    ]

    if use_gpu:
        enc = _resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg)
        if enc:
            cmd = base_cmd + _encoder_cmd_args(enc, q) + ["-pix_fmt", "yuv420p", output_path]
            return _spawn_ffmpeg_pipe(cmd)

    cmd = base_cmd + ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", output_path]
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
            # 超时：发 kill 后再限时回收，避免 kill 仍不返回时无限阻塞。
            ffmpeg_proc.process.kill()
            try:
                returncode = ffmpeg_proc.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # 进程拒绝退出：不再死等，按超时处理并尽量回收 stderr。
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
