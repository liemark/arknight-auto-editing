# exporter.py —— 导出：阶梯式回退 + 编码器 profile + 关键帧无损直通 + 真实进度 + 可取消
#
# 阶梯（逐级尝试，失败自动降级）：
#   1) 关键帧无损直通：段起点都落在关键帧 → 逐段 -c copy + concat，零重编码；
#   2) 分块并行重编码：段数够多时按块并行 + concat；
#   3) 单遍滤镜编码（硬件或 libx264）；
#   4) 逐帧管道写入器（硬件 → CPU）；
#   5) imageio / cv2.VideoWriter。

from __future__ import annotations

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

        if keys:
            try:
                if _try_copy_export(video_path, output_path, ranges, fps, profile,
                                    ffmpeg_bin, progress_cb, cancel_event,
                                    workers=max(2, workers), keyframes=keys,
                                    status_cb=status_cb, include_audio=export_audio):
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
            return written_fast, total

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
