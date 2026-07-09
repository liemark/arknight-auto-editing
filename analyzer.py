# analyzer.py —— 模板加载 + 帧状态识别（无 GUI 依赖）

import cv2
import numpy as np
import os
import shutil
import subprocess
import concurrent.futures
import multiprocessing
from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)

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


def export_video(video_path: str, output_path: str, to_del,
                 fps: float, quality: int, progress_cb=None, use_gpu: bool = False):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if isinstance(to_del, set):
        mask = np.zeros(total, dtype=bool)
        for idx in to_del:
            if 0 <= idx < total: mask[idx] = True
        to_del = mask

    ret, sample = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret: raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]

    writer_kind = None
    ffmpeg_proc = None
    writer = None

    if shutil.which("ffmpeg"):
        ffmpeg_proc = _open_ffmpeg_pipe_writer(output_path, fps, w, h, quality, use_gpu=use_gpu)
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
            if not ret: break

            if writer_kind == "ffmpeg":
                ffmpeg_proc.stdin.write(frame.tobytes())
            elif writer_kind == "imageio":
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(frame)
            written += 1
            idx += 1

            if progress_cb and written % 60 == 0:
                progress_cb(idx / total, written)
    finally:
        _close_video_writer(writer_kind, writer, ffmpeg_proc)

    cap.release()
    return written, total


def export_ranges(video_path: str, output_path: str, ranges: list,
                  fps: float, quality: int, progress_cb=None, use_gpu: bool = False):
    if not ranges:
        return 0, 0

    if len(ranges) == 1 and shutil.which("ffmpeg"):
        s, e = ranges[0]
        frames = e - s + 1
        start_sec = s / fps
        q = max(0, min(10, int(quality)))
        crf = int(round(28 - q))

        cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.4f}", "-i", video_path, "-frames:v", str(frames)]

        enc = _pick_gpu_encoder() if use_gpu else None
        if enc == "h264_nvenc":
            cmd += ["-c:v", enc, "-preset", "p4", "-cq", str(18 + (10 - q))]
        elif enc == "h264_qsv":
            cmd += ["-c:v", enc, "-global_quality", str(18 + (10 - q))]
        elif enc:
            cmd += ["-c:v", enc, "-q:v", str(18 + (10 - q))]
        else:
            cmd += ["-c:v", "libx264", "-crf", str(crf)]

        cmd += ["-an", "-pix_fmt", "yuv420p", output_path]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if progress_cb: progress_cb(1.0, frames)
            return frames, frames
        except subprocess.CalledProcessError:
            pass

    cap = cv2.VideoCapture(video_path)
    total_frames_to_export = sum(e - s + 1 for s, e in ranges)
    if total_frames_to_export <= 0:
        cap.release()
        return 0, 0

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
        ffmpeg_proc = _open_ffmpeg_pipe_writer(output_path, fps, w, h, quality, use_gpu=use_gpu)
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
        for s, e in ranges:
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_pos != s:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s)

            for i in range(s, e + 1):
                ret, frame = cap.read()
                if not ret: break

                if writer_kind == "ffmpeg":
                    ffmpeg_proc.stdin.write(frame.tobytes())
                elif writer_kind == "imageio":
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(frame)

                written += 1
                if progress_cb and written % 30 == 0:
                    progress_cb(written / total_frames_to_export, written)
    finally:
        _close_video_writer(writer_kind, writer, ffmpeg_proc)

    cap.release()
    return written, total_frames_to_export


def _pick_gpu_encoder() -> str | None:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None

    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"]
    for enc in candidates:
        if enc in out: return enc
    return None


def _open_ffmpeg_pipe_writer(output_path: str, fps: float, w: int, h: int, quality: int, use_gpu: bool):
    q = max(0, min(10, int(quality)))
    crf = int(round(28 - q))
    base_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
    ]

    if use_gpu:
        enc = _pick_gpu_encoder()
        if enc:
            if enc == "h264_nvenc":
                cmd = base_cmd + ["-c:v", enc, "-preset", "p4", "-cq", str(18 + (10 - q)), output_path]
            elif enc == "h264_qsv":
                cmd = base_cmd + ["-c:v", enc, "-global_quality", str(18 + (10 - q)), output_path]
            else:
                cmd = base_cmd + ["-c:v", enc, "-q:v", str(18 + (10 - q)), output_path]
            return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd = base_cmd + ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _close_video_writer(writer_kind, writer, ffmpeg_proc):
    if writer_kind == "ffmpeg" and ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        _, err = ffmpeg_proc.communicate()
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {err.decode('utf-8', errors='ignore')}")
    elif writer_kind == "imageio" and writer:
        writer.close()
    elif writer_kind == "cv2" and writer:
        writer.release()