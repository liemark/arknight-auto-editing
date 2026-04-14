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
    'pause':      {'ref_dir': 'templates_pause',  'source_dir': 'source_images_pause'},
    'speed_1x':   {'ref_dir': 'templates_1x',     'source_dir': 'source_images_1x'},
    'speed_2x':   {'ref_dir': 'templates_2x',     'source_dir': 'source_images_2x'},
    'speed_0_2x': {'ref_dir': 'templates_play',   'source_dir': 'source_images_play'},
}

IMG_EXTS = ('.png', '.jpg', '.bmp', '.jpeg')


def load_templates(proc_res: tuple = (400, 225)) -> tuple[dict, int]:
    """
    加载所有模板并预缩放到 proc_res。
    每个模板条目包含：
      cached_proc_res / cached_roi / cached_t / cached_m：
        预计算的 ROI 坐标和缩放后模板，每帧匹配时直接用，跳过 resize。
    """
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
    """对已缩放好的灰度图做模板匹配分类"""
    if configs['pause'] and _get_best_score(gray, configs['pause'], proc_res) >= thresholds['pause']:
        return FRAME_TYPE_PAUSE
    x1s = _get_best_score(gray, configs['speed_1x'], proc_res) if configs['speed_1x'] else -1.0
    x2s = _get_best_score(gray, configs['speed_2x'], proc_res) if configs['speed_2x'] else -1.0
    if x1s >= thresholds['speed_1x'] and x1s > x2s: return FRAME_TYPE_1X
    if x2s >= thresholds['speed_2x'] and x2s > x1s: return FRAME_TYPE_2X
    if configs['speed_0_2x'] and _get_best_score(gray, configs['speed_0_2x'], proc_res) >= thresholds['speed_0_2x']:
        return FRAME_TYPE_0_2X
    return FRAME_TYPE_NORMAL


def classify_frame(frame: np.ndarray, configs: dict,
                   thresholds: dict, proc_res: tuple) -> int:
    """将一帧 BGR 图像分类为 FRAME_TYPE_*（供外部直接调用）"""
    resized = cv2.resize(frame, proc_res, interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return _classify_gray(gray, configs, thresholds, proc_res)


# ---------------------------------------------------------------
#  子进程全局状态（ProcessPoolExecutor initializer 模式）
#
#  子进程只接收已经 resize+cvtColor 过的灰度图（400×225×1），
#  不接收原始 BGR 大帧（1920×1080×3）。
#  pickle 体积从 ~6MB/帧 降到 ~90KB/帧，缓解 IPC 瓶颈。
#
#  帧缓存继承：连续相同帧跳过 matchTemplate，对1帧暂停安全。
# ---------------------------------------------------------------

_worker_configs:    dict  = {}
_worker_thresholds: dict  = {}
_worker_proc_res:   tuple = (400, 225)
_worker_prev_roi:   object = None
_worker_prev_state: int   = FRAME_TYPE_NORMAL

# 帧差只看图标 ROI（400×225 坐标系）
_ROI_A = (180, 90, 260, 135)   # pause 字母区
_ROI_B = (325,  0, 400,  30)   # 1x/2x + play 按钮（右上角）

_SAME_FRAME_THRESH = 1.0   # 均值差阈值：编码噪声 ~0.67，图标变化 >1.5


def _worker_init(configs: dict, thresholds: dict, proc_res: tuple):
    global _worker_configs, _worker_thresholds, _worker_proc_res
    global _worker_prev_roi, _worker_prev_state
    _worker_configs    = configs
    _worker_thresholds = thresholds
    _worker_proc_res   = proc_res
    _worker_prev_roi   = None
    _worker_prev_state = FRAME_TYPE_NORMAL


def _extract_roi_strip(gray_proc: np.ndarray) -> np.ndarray:
    a = gray_proc[_ROI_A[1]:_ROI_A[3], _ROI_A[0]:_ROI_A[2]]
    b = gray_proc[_ROI_B[1]:_ROI_B[3], _ROI_B[0]:_ROI_B[2]]
    return np.concatenate([a.ravel(), b.ravel()])


def _worker_classify_gray(gray: np.ndarray) -> int:
    """
    子进程帧任务：接收已缩放好的灰度图（不再做 resize+cvtColor）。
    1. 提取图标 ROI strip，与上一帧对比
    2. diff < 阈值 → 直接继承，跳过 matchTemplate
    3. 否则走 _classify_gray
    """
    global _worker_prev_roi, _worker_prev_state
    roi_strip = _extract_roi_strip(gray)
    if _worker_prev_roi is not None:
        diff = float(np.mean(np.abs(roi_strip.astype(np.int16) - _worker_prev_roi.astype(np.int16))))
        if diff < _SAME_FRAME_THRESH: return _worker_prev_state

    state = _classify_gray(gray, _worker_configs, _worker_thresholds, _worker_proc_res)
    _worker_prev_roi, _worker_prev_state = roi_strip, state
    return state


# ---------------------------------------------------------------
#  批量分析整段视频
# ---------------------------------------------------------------

def analyze_video(video_path: str, configs: dict, thresholds: dict,
                  proc_res: tuple, batch_size: int, n_threads: int,
                  progress_cb=None) -> np.ndarray:
    """
    流水线 + 多进程分类每帧。

    IO 优化：读帧线程内完成 resize+cvtColor，只向队列放灰度图（~90KB），
    不放原始 BGR 帧（~6MB）。pickle 体积缩小 60x，大幅降低 IPC 开销。

    progress_cb(ratio: float) 在 [0, 1] 之间回调。
    返回 np.ndarray[int8]，每元素为 FRAME_TYPE_*。
    """
    import threading
    from queue import Queue as _Queue

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    states = np.zeros(total, dtype=np.int8)
    n_workers = min(n_threads, multiprocessing.cpu_count())
    pw, ph = proc_res

    # 预处理任务队列
    preproc_q = _Queue(maxsize=batch_size * 2)

    def _reader_and_preprocess():
        """使用线程池并行处理 resize 和 cvtColor"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as tp:
            idx = 0
            futures = []
            while True:
                ret, frame = cap.read()
                if not ret: break

                # 提交预处理任务到线程池
                def process(f, i):
                    g = cv2.cvtColor(cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    return g, i

                futures.append(tp.submit(process, frame, idx))
                idx += 1

                # 保持线程池和队列平衡，防止内存溢出
                if len(futures) > 16:
                    res = futures.pop(0).result()
                    preproc_q.put(res)

            # 清理剩余任务
            for fut in futures:
                preproc_q.put(fut.result())
            preproc_q.put(None)

    reader_thread = threading.Thread(target=_reader_and_preprocess, daemon=True)
    reader_thread.start()

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(configs, thresholds, proc_res)) as ex:

        batch_grays, batch_indices = [], []
        done = False
        while not done:
            while len(batch_grays) < batch_size:
                item = preproc_q.get()
                if item is None:
                    done = True;
                    break
                gray, idx = item
                batch_grays.append(gray);
                batch_indices.append(idx)

            if not batch_grays: break

            # 使用较大的 chunksize 减少 IPC 往返
            chunk = max(4, len(batch_grays) // (n_workers * 2))
            results = list(ex.map(_worker_classify_gray, batch_grays, chunksize=chunk))

            for idx, s in zip(batch_indices, results):
                states[idx] = s
            if progress_cb: progress_cb(batch_indices[-1] / total)
            batch_grays.clear();
            batch_indices.clear()

    reader_thread.join(timeout=5)
    cap.release()
    return states


# ---------------------------------------------------------------
#  段落提取 + DCT/Hist 差异检测
# ---------------------------------------------------------------

def build_segments(states: np.ndarray, video_path: str, proc_res: tuple,
                   compare_cfg: dict, progress_cb=None) -> tuple[list, list]:
    """
    根据帧状态数组生成 pause_segments / speed_segments。
    compare_cfg: {'enabled', 'dct', 'hist', 'keep_n', 'keep_m'}
    """
    total  = len(states)
    keep_n = compare_cfg['keep_n']
    keep_m = compare_cfg['keep_m']
    pauses = []
    speeds = []

    cap = cv2.VideoCapture(video_path)

    i = 0
    while i < total:
        curr = int(states[i])
        s_i  = i
        while i < total and int(states[i]) == curr:
            i += 1
        e_i     = i - 1
        seg_len = e_i - s_i + 1

        if curr == FRAME_TYPE_PAUSE:
            is_diff = True
            dct_v = h_sim = 0.0

            if compare_cfg['enabled']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, s_i - 1))
                _, f_b = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, e_i + 1))
                _, f_a = cap.read()

                if f_b is not None and f_a is not None:
                    g_b = cv2.cvtColor(cv2.resize(f_b, proc_res), cv2.COLOR_BGR2GRAY)
                    g_a = cv2.cvtColor(cv2.resize(f_a, proc_res), cv2.COLOR_BGR2GRAY)

                    dct_v = float(np.sum(np.abs(
                        cv2.dct(np.float32(g_a))[:8, :8] -
                        cv2.dct(np.float32(g_b))[:8, :8])))
                    h_sim = float(cv2.compareHist(
                        cv2.calcHist([g_b], [0], None, [256], [0, 256]),
                        cv2.calcHist([g_a], [0], None, [256], [0, 256]),
                        cv2.HISTCMP_CORREL))

                    if dct_v < compare_cfg['dct'] and h_sim > compare_cfg['hist']:
                        is_diff = False

            if is_diff:
                n = min(keep_n, seg_len)
                m = min(keep_m, seg_len - n)
                trim_in  = s_i + n
                trim_out = e_i - m + 1
                if trim_in > trim_out:
                    trim_in = trim_out = s_i
            else:
                trim_in  = s_i
                trim_out = e_i + 1

            pauses.append({
                'id':       len(pauses),
                'start':    s_i,
                'end':      e_i,
                'trim_in':  trim_in,
                'trim_out': trim_out,
                'dct':      dct_v,
                'hist':     h_sim,
                'is_diff':  is_diff,
            })

        elif curr in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
            speeds.append({'type': curr, 'start': s_i, 'end': e_i})

    cap.release()
    return pauses, speeds


# ---------------------------------------------------------------
#  导出辅助
# ---------------------------------------------------------------

def _speedup_mask(states: np.ndarray, frame_type: int, factor: int,
                  exclude_mask: np.ndarray) -> np.ndarray:
    total     = len(states)
    type_mask = (states == frame_type) & ~exclude_mask

    if not type_mask.any():
        return np.zeros(total, dtype=bool)

    cumsum  = np.cumsum(type_mask)
    shifted = np.empty(total, dtype=bool)
    shifted[0]  = False
    shifted[1:] = type_mask[:-1]
    seg_starts  = np.where(type_mask & ~shifted)[0]

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
    """计算需要删除的帧 bool mask（numpy，比 set 快 10-50x）"""
    del_mask = np.zeros(total, dtype=bool)

    # 1. 暂停段裁剪区（删中间）
    for seg in pause_segments:
        if seg['trim_out'] > seg['trim_in']:
            del_mask[seg['trim_in']:seg['trim_out']] = True

    # 2. 手动裁剪段（删两端，保中间；keep_in > keep_out 时全删）
    for seg in clip_segments:
        s, e = seg['start'], seg['end']
        ki, ko = seg['keep_in'], seg['keep_out']
        if ki > ko:
            del_mask[s:e + 1] = True
        else:
            if ki > s:   del_mask[s:ki] = True
            if ko < e:   del_mask[ko + 1:e + 1] = True

    # 3. 1x 变速抽帧
    if speedup_1x:
        del_mask |= _speedup_mask(states, FRAME_TYPE_1X, 2, del_mask)

    # 4. 0.2x 变速抽帧
    if speedup_02 and speedup_02_factor > 1:
        del_mask |= _speedup_mask(states, FRAME_TYPE_0_2X, speedup_02_factor, del_mask)

    return del_mask


def export_video(video_path: str, output_path: str,to_del,
                 fps: float, quality: int, progress_cb=None,use_gpu: bool = False):
    """顺序读取并写入保留帧"""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if isinstance(to_del, set):
        mask = np.zeros(total, dtype=bool)
        for idx in to_del:
            if 0 <= idx < total:
                mask[idx] = True
        to_del = mask

    ret, sample = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]

    writer_kind = None
    ffmpeg_proc = None
    writer = None

    if shutil.which("ffmpeg"):
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            output_path, fps, w, h, quality, use_gpu=use_gpu)
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
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if not to_del[idx]:
                if writer_kind == "ffmpeg":
                    ffmpeg_proc.stdin.write(frame.tobytes())
                elif writer_kind == "imageio":
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(frame)
                written += 1
            if progress_cb and idx % 60 == 0:
                progress_cb(idx / total, written)
    finally:
        _close_video_writer(writer_kind, writer, ffmpeg_proc)

    cap.release()
    return written, total


def export_frame_range(video_path: str, output_path: str,start_frame: int, end_frame: int,
                       fps: float, quality: int, progress_cb=None,use_gpu: bool = False):
    """导出闭区间 [start_frame, end_frame] 的视频（不含音频）"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("无法读取视频帧")

    s = max(0, min(start_frame, total - 1))
    e = max(0, min(end_frame, total - 1))
    if e < s:
        cap.release()
        raise RuntimeError("分段范围无效")

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)

    ret, sample = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    if not ret:
        cap.release()
        raise RuntimeError("无法读取视频帧")
    h, w = sample.shape[:2]

    writer_kind = None
    ffmpeg_proc = None
    writer = None

    if shutil.which("ffmpeg"):
        ffmpeg_proc = _open_ffmpeg_pipe_writer(
            output_path, fps, w, h, quality, use_gpu=use_gpu)
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

    seg_len = e - s + 1
    written = 0
    try:
        for i in range(seg_len):
            ret, frame = cap.read()
            if not ret:
                break
            if writer_kind == "ffmpeg":
                ffmpeg_proc.stdin.write(frame.tobytes())
            elif writer_kind == "imageio":
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(frame)
            written += 1
            if progress_cb and i % 30 == 0:
                progress_cb((i + 1) / seg_len, written)
    finally:
        _close_video_writer(writer_kind, writer, ffmpeg_proc)

    cap.release()
    return written, seg_len


def _pick_gpu_encoder() -> str | None:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None

    candidates = [
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
        "h264_videotoolbox",
    ]
    for enc in candidates:
        if enc in out:
            return enc
    return None


def _open_ffmpeg_pipe_writer(output_path: str, fps: float, w: int, h: int,
                             quality: int, use_gpu: bool):
    q = max(0, min(10, int(quality)))
    crf = int(round(28 - q))
    base_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
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