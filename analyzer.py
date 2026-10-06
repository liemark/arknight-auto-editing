# analyzer.py —— 模板加载 + 段落提取 + 删除掩码
#
# 本模块同时是兼容门面：解码/流水线在 pipeline.py，匹配在 matcher.py，
# 能力探测在 gpu_caps.py，导出在 exporter.py，此处 re-export 老名字。

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import app_core
import exporter
import gpu_caps
import matcher
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


def load_templates(proc_res: tuple = (400, 225)) -> tuple[dict, int]:
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


# ===============================================================
#  单帧匹配（进程池后端使用）
# ===============================================================

_get_best_score = matcher.get_best_score
_classify_gray = matcher.classify_gray

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
                progress_cb(0.5 + (e_i / max(1, total)) * 0.25)

        elif curr in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
            speeds.append({'type': curr, 'start': s_i, 'end': e_i})

    # 边界差分优先取第一遍上下文；不可用则整体回退二次扫片
    context_records = None
    if pauses and analysis_context is not None:
        context_records = context_records_for_pauses(analysis_context, pauses, total)

    if pauses and context_records is not None:
        for p, rec in zip(pauses, context_records):
            diff = float(rec['diff'])
            p['boundary_diff'] = diff
            if diff < boundary_thresh:
                p['mode'] = 'all'
        if progress_cb:
            progress_cb(1.0)
        return pauses, speeds

    if pauses:
        app_core.info("边界上下文不可用，回退二次扫片", "analyzer")
        cap = cv2.VideoCapture(video_path)
        target_indices = sorted(list(set([max(0, p['start'] - 1) for p in pauses] +
                                        [min(total - 1, p['end'] + 1) for p in pauses])))
        target_frames = {}
        curr_idx = 0
        for target in target_indices:
            while curr_idx < target:
                cap.grab()
                curr_idx += 1
            ret, frame = cap.read()
            if ret:
                target_frames[target] = cv2.cvtColor(
                    cv2.resize(frame, proc_res, interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY)
            curr_idx += 1
        cap.release()

        for p in pauses:
            b_idx = max(0, p['start'] - 1)
            a_idx = min(total - 1, p['end'] + 1)
            if b_idx in target_frames and a_idx in target_frames:
                diff = float(cv2.mean(cv2.absdiff(target_frames[b_idx],
                                                  target_frames[a_idx]))[0])
                p['boundary_diff'] = diff
                if diff < boundary_thresh:
                    p['mode'] = 'all'
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
    del_mask = np.zeros(total, dtype=bool)

    for seg in pause_segments:
        s, e = seg['start'], seg['end']
        mode = seg.get('mode', 'auto')
        if mode == 'all':
            del_mask[s:e + 1] = True
        elif mode == 'auto' and 'local_del_mask' in seg:
            m = seg['local_del_mask']
            # 1 自动删除，2 人工强制删除
            del_mask[s:e + 1] = (m == 1) | (m == 2)

    for seg in clip_segments:
        s, e = seg['start'], seg['end']
        ki, ko = seg['keep_in'], seg['keep_out']
        if ki > ko:
            del_mask[s:e + 1] = True
        else:
            if ki > s:
                del_mask[s:ki] = True
            if ko < e:
                del_mask[ko + 1:e + 1] = True

    if speedup_1x:
        del_mask |= _speedup_mask(states, FRAME_TYPE_1X, 2, del_mask)

    if speedup_02 and speedup_02_factor > 1:
        del_mask |= _speedup_mask(states, FRAME_TYPE_0_2X, speedup_02_factor, del_mask)

    return del_mask


# ===============================================================
#  导出（re-export）
# ===============================================================

_kept_frame_ranges = exporter._kept_frame_ranges
_has_audio_stream = exporter._has_audio_stream
_open_ffmpeg_pipe_writer = exporter._open_ffmpeg_pipe_writer
_close_video_writer = exporter._close_video_writer
export_video = exporter.export_video
export_ranges = exporter.export_ranges
