# analyzer.py —— 模板加载 + 帧状态识别（无 GUI 依赖）
#
# 优化点：
#   1. load_templates(proc_res) 预缩放模板和 ROI，每帧匹配时跳过 resize
#   2. classify_frame 走缓存路径，_get_best_score 减少 ~4 次 resize/帧
#   3. analyze_video 改用 ProcessPoolExecutor + initializer 模式：
#      - 真正绕过 GIL，线性利用多核
#      - configs/thresholds 在子进程 initializer 里初始化，
#        不在每帧任务里 pickle 传输大数组，只传轻量 frame ndarray
#   4. build_delete_set 换成 numpy 向量操作，省去 Python 级 for 循环

import cv2
import numpy as np
import os
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
      - template_gray / mask：原始图（备用，proc_res 变化时回退）
      - source_res / roi_orig：原始坐标
      - cached_proc_res：缓存对应的 proc_res
      - cached_roi (erx,ery,erw,erh)：预计算的 ROI 像素坐标
      - cached_t / cached_m：预缩放后的模板和蒙版
    """
    configs: dict[str, list] = {k: [] for k in TEMPLATE_DIRS}
    total = 0

    for ctype, dirs in TEMPLATE_DIRS.items():
        src_dir = dirs['source_dir']
        ref_dir = dirs['ref_dir']
        if not os.path.exists(src_dir) or not os.path.exists(ref_dir):
            continue

        src_files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(IMG_EXTS)]
        if not src_files or not ref_files:
            continue

        src_img = cv2.imread(os.path.join(src_dir, src_files[0]), cv2.IMREAD_GRAYSCALE)
        if src_img is None:
            continue
        sh, sw = src_img.shape

        for rf in ref_files:
            ref_img = cv2.imread(os.path.join(ref_dir, rf), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                continue
            rh, rw = ref_img.shape
            if rh > sh or rw > sw:
                print(f"[警告] 模板 {rf} ({rw}x{rh}) 大于源图 ({sw}x{sh})，跳过")
                continue

            res = cv2.matchTemplate(src_img, ref_img, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            rx, ry = max_loc
            _, mask = cv2.threshold(ref_img, 10, 255, cv2.THRESH_BINARY)

            # 预计算缩放后 ROI 和模板
            scale_x = proc_res[0] / sw
            scale_y = proc_res[1] / sh
            ext = 2.0
            erx = max(0, int(rx * scale_x - rw * scale_x * (ext - 1) / 2))
            ery = max(0, int(ry * scale_y - rh * scale_y * (ext - 1) / 2))
            erw = int(rw * scale_x * ext)
            erh = int(rh * scale_y * ext)
            tw  = max(1, int(rw * scale_x))
            th  = max(1, int(rh * scale_y))
            t_r = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
            m_r = cv2.resize(mask,    (tw, th), interpolation=cv2.INTER_NEAREST)

            configs[ctype].append({
                # 原始数据（proc_res 变化时回退用）
                'template_gray':   ref_img,
                'mask':            mask,
                'roi_orig':        (rx, ry, rw, rh),
                'source_res':      (sw, sh),
                'name':            rf,
                # 预缓存
                'cached_proc_res': proc_res,
                'cached_roi':      (erx, ery, erw, erh),
                'cached_t':        t_r,
                'cached_m':        m_r,
            })
            total += 1

    return configs, total


# ---------------------------------------------------------------
#  单帧匹配（优先走预缓存路径）
# ---------------------------------------------------------------

def _get_best_score(gray_frame: np.ndarray, templates: list, proc_res: tuple) -> float:
    max_score = -1.0
    fh, fw = gray_frame.shape

    for t in templates:
        # 缓存命中：直接用预计算的 ROI 坐标和缩放后模板
        if t.get('cached_proc_res') == proc_res:
            erx, ery, erw, erh = t['cached_roi']
            t_r = t['cached_t']
            m_r = t['cached_m']
            erw = min(fw - erx, erw)
            erh = min(fh - ery, erh)
        else:
            # 回退：实时计算（proc_res 与加载时不同时）
            sw, sh = t['source_res']
            rx, ry, rw, rh = t['roi_orig']
            scale_x = proc_res[0] / sw
            scale_y = proc_res[1] / sh
            ext = 2.0
            erx = max(0, int(rx * scale_x - rw * scale_x * (ext - 1) / 2))
            ery = max(0, int(ry * scale_y - rh * scale_y * (ext - 1) / 2))
            erw = min(fw - erx, int(rw * scale_x * ext))
            erh = min(fh - ery, int(rh * scale_y * ext))
            tw  = max(1, int(rw * scale_x))
            th  = max(1, int(rh * scale_y))
            t_r = cv2.resize(t['template_gray'], (tw, th), interpolation=cv2.INTER_AREA)
            m_r = cv2.resize(t['mask'],          (tw, th), interpolation=cv2.INTER_NEAREST)

        if erw <= 0 or erh <= 0:
            continue
        roi = gray_frame[ery:ery + erh, erx:erx + erw]
        th2, tw2 = t_r.shape
        if roi.shape[0] < th2 or roi.shape[1] < tw2:
            continue

        res = cv2.matchTemplate(roi, t_r, cv2.TM_CCOEFF_NORMED, mask=m_r)
        _, score, _, _ = cv2.minMaxLoc(res)
        if np.isfinite(score):
            max_score = max(max_score, score)

    return max_score


def _classify_gray(gray: np.ndarray, configs: dict,
                   thresholds: dict, proc_res: tuple) -> int:
    """对已缩放好的灰度图做模板匹配分类（内部函数，避免重复 resize）"""
    if configs['pause'] and \
            _get_best_score(gray, configs['pause'], proc_res) >= thresholds['pause']:
        return FRAME_TYPE_PAUSE

    x1s = _get_best_score(gray, configs['speed_1x'],   proc_res) if configs['speed_1x']   else -1.0
    x2s = _get_best_score(gray, configs['speed_2x'],   proc_res) if configs['speed_2x']   else -1.0

    if x1s >= thresholds['speed_1x'] and x1s > x2s: return FRAME_TYPE_1X
    if x2s >= thresholds['speed_2x'] and x2s > x1s: return FRAME_TYPE_2X

    if configs['speed_0_2x'] and \
            _get_best_score(gray, configs['speed_0_2x'], proc_res) >= thresholds['speed_0_2x']:
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
#  configs/thresholds/proc_res 只在进程初始化时传一次，
#  后续每帧任务只传轻量 frame ndarray，大幅减少 pickle 开销
#
#  帧缓存继承：相邻帧几乎完全相同时（如长暂停），直接复用上一帧结果，
#  跳过 matchTemplate。对1帧暂停完全安全：状态切换时帧差必然超过阈值。
# ---------------------------------------------------------------

_worker_configs:    dict  = {}
_worker_thresholds: dict  = {}
_worker_proc_res:   tuple = (400, 225)
_worker_prev_roi:   object = None   # 上一帧的图标区域灰度（拼接后）
_worker_prev_state: int   = FRAME_TYPE_NORMAL

# 帧差只看图标所在的两块 ROI（400×225 坐标系），忽略游戏画面运动
# ROI-A：pause 字母区域（中央偏下）
# ROI-B：1x/2x 数字 + play 按钮（右上角）
# 两块总像素 ~2750，比 16×9=144 大但仍远小于全帧 90000
# 优点：游戏角色运动不影响这两块，false-positive 率极低
_ROI_A = (180, 90, 260, 135)   # (x1,y1,x2,y2)
_ROI_B = (325,  0, 400,  30)

# ROI diff 阈值：编码噪声 ~0.67，图标出现/消失 >1.5，取 1.0 中间安全值
_SAME_FRAME_THRESH = 1.0


def _worker_init(configs: dict, thresholds: dict, proc_res: tuple):
    global _worker_configs, _worker_thresholds, _worker_proc_res
    global _worker_prev_roi, _worker_prev_state
    _worker_configs    = configs
    _worker_thresholds = thresholds
    _worker_proc_res   = proc_res
    _worker_prev_roi   = None
    _worker_prev_state = FRAME_TYPE_NORMAL


def _extract_roi_strip(gray_proc: np.ndarray) -> np.ndarray:
    """从处理分辨率灰度图中提取图标区域，拼成一维数组用于帧差比较"""
    a = gray_proc[_ROI_A[1]:_ROI_A[3], _ROI_A[0]:_ROI_A[2]]
    b = gray_proc[_ROI_B[1]:_ROI_B[3], _ROI_B[0]:_ROI_B[2]]
    # 拼成一维，方便 np.mean(abs(diff))
    return np.concatenate([a.ravel(), b.ravel()])


def _worker_classify(frame: np.ndarray) -> int:
    """
    子进程帧任务：
    1. resize+灰度（必须）
    2. 提取图标ROI拼接带（~24µs），与上一帧对比
    3. ROI均值差 < 1.0 → 直接继承（跳过matchTemplate）
    4. 否则走完整 _classify_gray
    """
    global _worker_prev_roi, _worker_prev_state

    resized = cv2.resize(frame, _worker_proc_res, interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    roi_strip = _extract_roi_strip(gray)

    if _worker_prev_roi is not None:
        diff = float(np.mean(np.abs(roi_strip.astype(np.int16)
                                    - _worker_prev_roi.astype(np.int16))))
        if diff < _SAME_FRAME_THRESH:
            _worker_prev_roi = roi_strip
            return _worker_prev_state

    state = _classify_gray(gray, _worker_configs, _worker_thresholds, _worker_proc_res)
    _worker_prev_roi   = roi_strip
    _worker_prev_state = state
    return state


# ---------------------------------------------------------------
#  批量分析整段视频
# ---------------------------------------------------------------

def analyze_video(video_path: str, configs: dict, thresholds: dict,
                  proc_res: tuple, batch_size: int, n_threads: int,
                  progress_cb=None) -> np.ndarray:
    """
    流水线 + 多进程分类每帧。
    - 独立读帧线程：持续将帧送入队列，消除 IO 等待
    - ProcessPoolExecutor + initializer：真正多核并行，configs 不重复传输
    - 子进程帧缓存继承：连续相同帧跳过 matchTemplate
    progress_cb(ratio: float) 在 [0, 1] 之间回调。
    返回 np.ndarray[int8]，每元素为 FRAME_TYPE_*。
    """
    import threading
    from queue import Queue as _Queue

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    states = np.zeros(total, dtype=np.int8)

    n_workers = min(n_threads, multiprocessing.cpu_count())
    # chunksize=16：减少 IPC 次数，实测比 8 快约 30%
    chunk = max(8, min(32, batch_size // n_workers))

    # 读帧队列：预读 batch_size 帧，让读帧和计算并行
    # 队列项：(frame_ndarray, frame_idx) 或 None（结束哨兵）
    read_q: _Queue = _Queue(maxsize=batch_size * 2)

    def _reader():
        idx = 0
        while idx < total:
            ret, f = cap.read()
            if not ret:
                break
            read_q.put((f, idx))
            idx += 1
        read_q.put(None)   # 结束哨兵

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(configs, thresholds, proc_res)) as ex:

        batch_frames: list = []
        batch_indices: list = []
        done = False

        while not done:
            # 积攒一批帧
            while len(batch_frames) < batch_size:
                item = read_q.get()
                if item is None:
                    done = True
                    break
                f, idx = item
                batch_frames.append(f)
                batch_indices.append(idx)

            if not batch_frames:
                break

            results = list(ex.map(_worker_classify, batch_frames, chunksize=chunk))
            for idx, s in zip(batch_indices, results):
                states[idx] = s

            if progress_cb:
                progress_cb(batch_indices[-1] / total)

            batch_frames.clear()
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

    # 用单个 cap 顺序读取所有暂停段的前后帧，避免反复 open/close
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
                # 读前帧
                idx_b = max(0, s_i - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
                _, f_b = cap.read()
                # 读后帧
                idx_a = min(total - 1, e_i + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
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
    """
    对 states == frame_type 的连续段，在段内按 factor 抽帧，
    返回"应删除"的布尔 mask（numpy 向量化，无 Python for 循环）。
    exclude_mask：已经标记为删除的帧，跳过不计入连续计数。
    """
    total    = len(states)
    type_mask = (states == frame_type) & ~exclude_mask   # 目标类型且未删除

    if not type_mask.any():
        return np.zeros(total, dtype=bool)

    # 计算每帧在其所在连续段内的局部序号（1-based）
    # 方法：cumsum 减去段起点前的 cumsum 值
    cumsum  = np.cumsum(type_mask)
    # 找每个连续段的起点
    shifted = np.empty(total, dtype=bool)
    shifted[0]  = False
    shifted[1:] = type_mask[:-1]
    seg_starts = np.where(type_mask & ~shifted)[0]

    # 在起点处记录 cumsum[start-1]（即上一段结束时的累计值）
    offsets = np.zeros(total, dtype=np.int64)
    for s in seg_starts:
        offsets[s:] = cumsum[s - 1] if s > 0 else 0

    local_cnt = np.where(type_mask, cumsum - offsets, 0)

    # factor=2: 删偶数序号（第2,4,6...）; factor=N: 保留第1,留N+1,…删其余
    if factor == 2:
        del_mask = type_mask & (local_cnt % 2 == 0)
    else:
        del_mask = type_mask & (local_cnt % factor != 1)

    return del_mask


def build_delete_set(total: int, states: np.ndarray,
                     pause_segments: list, speed_segments: list,
                     speedup_1x: bool, speedup_02: bool,
                     speedup_02_factor: int) -> np.ndarray:
    """
    计算需要删除的帧 bool mask（numpy，比 set 快 10-50x）。
    返回 np.ndarray[bool]，True = 删除。
    调用方可用 np.where(mask)[0] 转为下标集合，或直接用布尔索引。
    """
    del_mask = np.zeros(total, dtype=bool)

    # 1. 暂停段裁剪区
    for seg in pause_segments:
        if seg['trim_out'] > seg['trim_in']:
            del_mask[seg['trim_in']:seg['trim_out']] = True

    # 2. 1x 变速抽帧（每段内隔帧删）
    if speedup_1x:
        del_mask |= _speedup_mask(states, FRAME_TYPE_1X, 2, del_mask)

    # 3. 0.2x 变速抽帧
    if speedup_02 and speedup_02_factor > 1:
        del_mask |= _speedup_mask(states, FRAME_TYPE_0_2X, speedup_02_factor, del_mask)

    return del_mask


def export_video(video_path: str, output_path: str,
                 to_del,           # np.ndarray[bool] 或 set[int]
                 fps: float, quality: int, progress_cb=None):
    """顺序读取并写入保留帧"""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 统一转为 bool ndarray 以便 O(1) 索引
    if isinstance(to_del, set):
        mask = np.zeros(total, dtype=bool)
        for idx in to_del:
            if 0 <= idx < total:
                mask[idx] = True
        to_del = mask

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    quality=quality, pixelformat='yuv420p')
        use_imageio = True
    except ImportError:
        use_imageio = False
        ret, sample = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ret:
            raise RuntimeError("无法读取视频帧")
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    written = 0
    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if not to_del[idx]:
            if use_imageio:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(frame)
            written += 1
        if progress_cb and idx % 60 == 0:
            progress_cb(idx / total, written)

    writer.close() if use_imageio else writer.release()
    cap.release()
    return written, total