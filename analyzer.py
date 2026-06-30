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
#  批量分析整段视频
# ---------------------------------------------------------------

def analyze_video(video_path: str, configs: dict, thresholds: dict,
                  proc_res: tuple, batch_size: int, n_threads: int,
                  progress_cb=None) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    states = np.zeros(total, dtype=np.int8)
    diffs = np.zeros(total, dtype=np.float32)
    n_workers = min(n_threads, multiprocessing.cpu_count())
    pw, ph = proc_res

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(configs, thresholds, proc_res)) as ex:

        idx = 0
        prev_gray = None

        while True:
            batch_grays = []
            batch_indices = []

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break

                gray = cv2.cvtColor(cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                batch_grays.append(gray)
                batch_indices.append(idx)

                if prev_gray is not None:
                    diffs[idx] = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                prev_gray = gray
                idx += 1

            if not batch_grays:
                break

            chunk = max(4, len(batch_grays) // (n_workers * 2))
            results = list(ex.map(_worker_classify_gray, batch_grays, chunksize=chunk))

            for i, s in zip(batch_indices, results):
                states[i] = s

            if progress_cb:
                progress_cb((idx / total) * 0.5)

    cap.release()
    return states, diffs


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
                   compare_cfg: dict, fps: float, progress_cb=None) -> tuple[list, list]:
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
            if progress_cb: progress_cb(0.5 + (e_i / total) * 0.25)

        elif curr in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
            speeds.append({'type': curr, 'start': s_i, 'end': e_i})

    # 2. 批量极速比对暂停边界差异
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


def _has_audio_stream(video_path: str) -> bool:
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

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
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


def _gpu_encoder_works(enc: str, timeout: float = 5) -> bool:
    # 单飞探测：探测全程持锁，避免多线程并发启动多个 ffmpeg 探测同一编码器，
    # 进而在 GPU 资源紧张时因并发抢占产生假阴性并污染缓存。
    with _GPU_PROBE_LOCK:
        cached = _GPU_PROBE_CACHE.get(enc)
        if cached is not None:
            return cached

        works = False
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=0.1",
                 *_gpu_encoder_probe_args(enc), "-f", "null", "-"],
                check=True, capture_output=True, timeout=timeout,
                creationflags=_NO_WINDOW)
            works = True
        except FileNotFoundError:
            # FFmpeg 不在 PATH：不缓存，避免污染整个进程生命周期。
            return False
        except Exception:
            works = False

        _GPU_PROBE_CACHE[enc] = works
        return works


def list_ffmpeg_gpu_encoders() -> list[str]:
    try:
        output = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
            creationflags=_NO_WINDOW)
    except Exception:
        return []

    candidates = ["h264_amf", "h264_nvenc", "h264_qsv", "h264_videotoolbox"]
    return [enc for enc in candidates if enc in output]


def list_working_gpu_encoders() -> list[str]:
    return [enc for enc in list_ffmpeg_gpu_encoders() if _gpu_encoder_works(enc)]


def _pick_gpu_encoder() -> str | None:
    working = list_working_gpu_encoders()
    return working[0] if working else None


def _resolve_gpu_encoder(gpu_encoder: str | None) -> str | None:
    selected = (gpu_encoder or "").strip()
    if selected:
        return selected if _gpu_encoder_works(selected) else None
    return _pick_gpu_encoder()


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


def _video_encoder_args(quality: int, use_gpu: bool, gpu_encoder: str) -> list[str]:
    q = max(0, min(10, int(quality)))
    if use_gpu:
        enc = _resolve_gpu_encoder(gpu_encoder)
        if enc:
            return _encoder_cmd_args(enc, q) + ["-pix_fmt", "yuv420p", "-threads", "0"]
    crf = int(round(28 - q))
    return ["-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", "0"]


def _export_ranges_with_ffmpeg_filters(
        video_path: str, output_path: str, ranges: list[tuple[int, int]],
        fps: float, quality: int, use_gpu: bool, gpu_encoder: str,
        include_audio: bool, progress_cb=None) -> bool:
    """使用 FFmpeg trim/concat 快速导出左闭右开的帧区间。"""
    if not ranges or len(ranges) > 120:
        return False

    has_audio = include_audio and _has_audio_stream(video_path)
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
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", video_path, "-filter_complex_script", filter_file,
            "-map", "[outv]",
        ]
        if has_audio:
            cmd += ["-map", "[outa]"]
        cmd += _video_encoder_args(quality, use_gpu, gpu_encoder)
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
        except Exception:
            if os.path.isfile(output_path):
                os.remove(output_path)
            return False


def _mux_audio_for_ranges(video_path: str, video_only_path: str,
                          output_path: str, ranges: list[tuple[int, int]],
                          fps: float):
    if not _has_audio_stream(video_path):
        os.replace(video_only_path, output_path)
        return

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

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
             "-i", video_path, "-i", video_only_path,
             "-filter_complex_script", filter_file,
             "-map", "1:v:0", "-map", "[outa]",
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            check=True, capture_output=True, timeout=1800, creationflags=_NO_WINDOW)


def export_video(video_path: str, output_path: str, to_del,
                 fps: float, quality: int, progress_cb=None,
                 use_gpu: bool = False, gpu_encoder: str = ""):
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

    if shutil.which("ffmpeg") and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, ranges, fps, quality,
            use_gpu, gpu_encoder, True, progress_cb):
        cap.release()
        return written_fast, total

    ret, sample = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        cap.release()
        raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]

    use_ffmpeg = bool(shutil.which("ffmpeg"))
    video_only_path = output_path + ".video-only.tmp.mp4" if use_ffmpeg else output_path
    writer_kind = None
    ffmpeg_proc = None
    writer = None

    if use_ffmpeg:
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            video_only_path, fps, w, h, quality,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder)
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
        _close_video_writer(writer_kind, writer, ffmpeg_proc)

    if use_ffmpeg:
        try:
            _mux_audio_for_ranges(video_path, video_only_path, output_path, ranges, fps)
        finally:
            if os.path.isfile(video_only_path):
                os.remove(video_only_path)
    return written, total


def export_ranges(video_path: str, output_path: str, ranges: list,
                  fps: float, quality: int, progress_cb=None,
                  use_gpu: bool = False, gpu_encoder: str = ""):
    if not ranges:
        return 0, 0

    exclusive_ranges = [(start, end + 1) for start, end in ranges]
    total_frames_to_export = sum(end - start for start, end in exclusive_ranges)
    if shutil.which("ffmpeg") and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, exclusive_ranges, fps, quality,
            use_gpu, gpu_encoder, False, progress_cb):
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
    if shutil.which("ffmpeg"):
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            output_path, fps, w, h, quality,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder)
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
                             gpu_encoder: str = ""):
    q = max(0, min(10, int(quality)))
    crf = int(round(28 - q))
    base_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
    ]

    if use_gpu:
        enc = _resolve_gpu_encoder(gpu_encoder)
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
            ffmpeg_proc.process.kill()
            returncode = ffmpeg_proc.process.wait()
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
