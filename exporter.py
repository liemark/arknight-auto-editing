# exporter.py —— 导出：阶梯式回退 + 编码器 profile + 关键帧无损直通 + 真实进度 + 可取消
#
# 阶梯（逐级尝试，失败自动降级）：
#   1) 关键帧无损直通：段起点都落在关键帧 → 逐段 -c copy + concat，零重编码；
#   2) 分块并行重编码：段数够多时按块并行 + concat；
#   3) 单遍滤镜编码（硬件或 libx264）；
#   4) 逐帧管道写入器（硬件 → CPU）；
#   5) imageio / cv2.VideoWriter。

from __future__ import annotations

import concurrent.futures
import os
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

_CHUNK_MIN_SEGMENTS = 24
_KEYFRAME_CACHE: dict[str, tuple[float, list[int]]] = {}
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


def keyframe_frames(video_path: str, fps: float,
                    ffmpeg_path: str | None = None) -> list[int]:
    """关键帧帧号列表（时间戳×fps 折算，CFR 下准确）；只解关键帧，按文件 mtime 缓存。"""
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

    probe = _ffprobe_path(gpu_caps.resolve_ffmpeg_path(ffmpeg_path) if ffmpeg_path else None)
    frames: list[int] = []
    if probe and fps > 0:
        try:
            res = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                 "-show_entries", "frame=best_effort_timestamp_time",
                 "-of", "csv=p=0", video_path],
                check=False, capture_output=True, text=True, timeout=600,
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
    with _CACHE_LOCK:
        _KEYFRAME_CACHE[key] = (time.time(), list(frames))
    return frames


# ===============================================================
#  带进度/取消的 ffmpeg 调用
# ===============================================================

def _run_ffmpeg_progress(cmd: list[str], total_frames: int,
                         progress_cb=None, cancel_event=None,
                         ratio_base: float = 0.0, ratio_span: float = 1.0,
                         done_offset: int = 0,
                         timeout: float = 7200.0) -> tuple[bool, str]:
    """跑一个 ffmpeg，解析 `-progress pipe:1`，支持取消。返回 (成功, stderr 摘要)。"""
    stderr_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=stderr_file, text=True, encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW)
    except Exception as exc:
        stderr_file.close()
        return False, f"{type(exc).__name__}: {exc}"

    deadline = time.perf_counter() + timeout
    cancelled = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            line = line.strip()
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
            if time.perf_counter() > deadline:
                proc.kill()
                return False, f"ffmpeg 超时（{timeout:.0f}s）"
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
    """导出用的硬解输入参数（不带 hwaccel_output_format，让滤镜在内存帧上跑）。"""
    if profile is None or not profile.decode_variant:
        return []
    for v in gpu_caps.build_decode_variants(profile.hwaccels):
        if v.key == profile.decode_variant and v.hwaccel:
            return v.hwaccel[:2]
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
                     cancel_event=None, workers: int = 4) -> bool:
    expected = sum(e - s for s, e in ranges)
    if expected <= 0:
        return False
    keys = set(keyframe_frames(video_path, fps, ffmpeg_path=ffmpeg))
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
            if got > 0 and abs(got - expected) <= max(2, len(ranges)):
                app_core.info(f"时长校验存疑但帧数正确（{got} vs {expected}），继续", "exporter")
            else:
                app_core.warn(f"无损直通{why}，改用重编码路径", "exporter")
                return False

        if _has_audio_stream(video_path, ffmpeg_path=ffmpeg):
            _mux_audio_for_ranges(video_path, video_only, output_path, ranges, fps,
                                  ffmpeg_path=ffmpeg)
        else:
            shutil.move(video_only, output_path)
    if progress_cb:
        progress_cb(1.0, expected)
    return True


# ===============================================================
#  2) 分块并行重编码
# ===============================================================

def _write_filter_script(path: str, ranges: list[tuple[int, int]], fps: float,
                         has_audio: bool) -> None:
    lines = []
    labels = []
    for idx, (start, end) in enumerate(ranges):
        lines.append(f"[0:v]trim=start_frame={start}:end_frame={end},"
                     f"setpts=PTS-STARTPTS[v{idx}]")
        labels.append(f"[v{idx}]")
        if has_audio:
            lines.append(f"[0:a]atrim=start={start / fps:.9f}:end={end / fps:.9f},"
                         f"asetpts=PTS-STARTPTS[a{idx}]")
            labels.append(f"[a{idx}]")
    lines.append("".join(labels) + f"concat=n={len(ranges)}:v=1:a={1 if has_audio else 0}"
                 + ("[outv][outa]" if has_audio else "[outv]"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";\n".join(lines))


def _encode_args(quality: int, use_gpu: bool, gpu_encoder: str,
                 ffmpeg: str, preset: str | None) -> list[str]:
    return gpu_caps.video_encoder_args(quality, use_gpu, gpu_encoder,
                                       ffmpeg_path=ffmpeg, preset=preset)


def _export_ranges_with_ffmpeg_filters(
        video_path: str, output_path: str, ranges: list[tuple[int, int]],
        fps: float, quality: int, use_gpu: bool, gpu_encoder: str,
        include_audio: bool, progress_cb=None,
        ffmpeg_path: str | None = None, preset: str | None = None,
        cancel_event=None, profile=None,
        ratio_base: float = 0.02, ratio_span: float = 0.96) -> bool:
    """单遍滤镜 trim+concat 导出（左闭右开区间）。"""
    if not ranges:
        return False
    try:
        ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    except FileNotFoundError:
        return False

    has_audio = include_audio and _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    total_frames = sum(e - s for s, e in ranges)

    with tempfile.TemporaryDirectory() as tmpdir:
        filter_file = os.path.join(tmpdir, "filter.txt")
        _write_filter_script(filter_file, ranges, fps, has_audio)

        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
               *_hwaccel_input_args(profile),
               "-i", video_path,
               "-filter_complex_script", filter_file, "-map", "[outv]"]
        if has_audio:
            cmd += ["-map", "[outa]"]
        cmd += _encode_args(quality, use_gpu, gpu_encoder, ffmpeg, preset)
        cmd += ["-c:a", "aac"] if has_audio else ["-an"]
        cmd += ["-progress", "pipe:1", output_path]

        ok, err = _run_ffmpeg_progress(
            cmd, total_frames, progress_cb=progress_cb, cancel_event=cancel_event,
            ratio_base=ratio_base, ratio_span=ratio_span, timeout=14400.0)
        if not ok:
            enc = gpu_caps.resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg) if use_gpu else None
            app_core.warn(
                f"单遍滤镜导出失败（编码器 {enc or 'libx264'}）：{err[:300]}", "exporter")
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return False
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 0


# ===============================================================
#  3) 音频混流（视频 copy，仅重编码音频）
# ===============================================================

def _mux_audio_for_ranges(video_path: str, video_only_path: str,
                          output_path: str, ranges: list[tuple[int, int]],
                          fps: float, ffmpeg_path: str | None = None) -> None:
    if not _has_audio_stream(video_path, ffmpeg_path=ffmpeg_path):
        os.replace(video_only_path, output_path)
        return

    ffmpeg = gpu_caps.resolve_ffmpeg_path(ffmpeg_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        filter_file = os.path.join(tmpdir, "audio-filter.txt")
        lines = []
        labels = []
        for idx, (start, end) in enumerate(ranges):
            lines.append(f"[0:a]atrim=start={start / fps:.9f}:end={end / fps:.9f},"
                         f"asetpts=PTS-STARTPTS[a{idx}]")
            labels.append(f"[a{idx}]")
        lines.append("".join(labels) + f"concat=n={len(ranges)}:v=0:a=1[outa]")
        with open(filter_file, "w", encoding="utf-8") as fh:
            fh.write(";\n".join(lines))

        try:
            # 不能用 -shortest：音频按 AAC 包粒度对齐会比视频略短，会裁掉尾部视频帧
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                 "-i", video_path, "-i", video_only_path,
                 "-filter_complex_script", filter_file,
                 "-map", "1:v:0", "-map", "[outa]",
                 "-c:v", "copy", "-c:a", "aac", output_path],
                check=True, capture_output=True, timeout=3600,
                creationflags=_NO_WINDOW)
        except Exception as exc:
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
                    f"音频混流失败: {stderr.decode('utf-8', errors='ignore')[:400]}") from exc
            raise


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
                      avoid_seek: bool = False):
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
            _mux_audio_for_ranges(video_path, video_only_path, output_path,
                                  ranges, fps, ffmpeg_path=ffmpeg_bin)
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
                 profile=None):
    """整段剪辑导出。返回 (written, total)。"""
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
        if keyframe_copy:
            try:
                if _try_copy_export(video_path, output_path, ranges, fps, profile,
                                    ffmpeg_bin, progress_cb, cancel_event,
                                    workers=max(2, workers)):
                    return written_fast, total
            except ExportCancelled:
                raise
            except Exception as exc:
                app_core.warn(f"无损直通异常，回退重编码: {exc}", "exporter")

        if workers > 1 and len(ranges) >= _CHUNK_MIN_SEGMENTS:
            try:
                if _export_chunks_parallel(video_path, output_path, ranges, fps,
                                           quality, use_gpu, gpu_encoder, ffmpeg_bin,
                                           workers, export_preset, progress_cb,
                                           cancel_event, profile):
                    return written_fast, total
            except ExportCancelled:
                raise
            except Exception as exc:
                app_core.warn(f"分块并行导出异常，回退单遍滤镜: {exc}", "exporter")

        if _export_ranges_with_ffmpeg_filters(
                video_path, output_path, ranges, fps, quality, use_gpu,
                gpu_encoder, True, progress_cb, ffmpeg_path=ffmpeg_bin,
                preset=export_preset, cancel_event=cancel_event, profile=profile):
            return written_fast, total

    written, tot = _export_per_frame(
        video_path, output_path, to_del, fps, quality, progress_cb,
        use_gpu=use_gpu, gpu_encoder=gpu_encoder, ffmpeg_path=ffmpeg_path,
        preset=export_preset)
    return written, tot


def _export_chunks_parallel(video_path: str, output_path: str,
                            ranges: list[tuple[int, int]], fps: float,
                            quality: int, use_gpu: bool, gpu_encoder: str,
                            ffmpeg: str, workers: int, preset: str | None,
                            progress_cb, cancel_event, profile) -> bool:
    """按块并行重编码再 concat。

    每块用输入 seek 到该块首段起点，块内 trim 用相对帧号，
    因此每块只解码自己那一段区间。
    """
    expected = sum(e - s for s, e in ranges)
    n_chunks = max(1, min(workers, (len(ranges) + _CHUNK_MIN_SEGMENTS - 1) // _CHUNK_MIN_SEGMENTS))
    per = (len(ranges) + n_chunks - 1) // n_chunks
    chunks = [ranges[i:i + per] for i in range(0, len(ranges), per)]
    n_chunks = len(chunks)
    if n_chunks < 2:
        return False

    has_audio = _has_audio_stream(video_path, ffmpeg_path=ffmpeg)
    enc = gpu_caps.resolve_gpu_encoder(gpu_encoder, ffmpeg_path=ffmpeg) if use_gpu else None
    app_core.info(f"分块并行导出：{n_chunks} 块 / 并发 {min(workers, n_chunks)} / "
                  f"编码器 {enc or 'libx264'}", "exporter")

    with tempfile.TemporaryDirectory(prefix="aae_chunk_") as tmpdir:
        parts = [os.path.join(tmpdir, f"c{i:03d}.mp4") for i in range(n_chunks)]
        progress = [0.0] * n_chunks
        lock = threading.Lock()

        def run_chunk(ci: int) -> bool:
            chunk = chunks[ci]
            base = chunk[0][0]
            rel = [(s - base, e - base) for s, e in chunk]
            filter_file = os.path.join(tmpdir, f"f{ci:03d}.txt")
            _write_filter_script(filter_file, rel, fps, has_audio)
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                   "-nostdin", *_hwaccel_input_args(profile),
                   "-ss", f"{base / fps:.6f}", "-i", video_path,
                   "-filter_complex_script", filter_file, "-map", "[outv]"]
            if has_audio:
                cmd += ["-map", "[outa]"]
            cmd += _encode_args(quality, use_gpu, gpu_encoder, ffmpeg, preset)
            cmd += ["-c:a", "aac"] if has_audio else ["-an"]
            cmd += ["-progress", "pipe:1", parts[ci]]

            chunk_total = sum(e - s for s, e in rel)

            def _prog(frac, done, _ci=ci):
                with lock:
                    progress[_ci] = frac
                    if progress_cb:
                        progress_cb(0.02 + 0.88 * (sum(progress) / len(progress)), done)

            ok, err = _run_ffmpeg_progress(
                cmd, chunk_total, progress_cb=_prog, cancel_event=cancel_event,
                ratio_base=0.0, ratio_span=1.0, timeout=14400.0)
            if not ok:
                app_core.warn(f"块 {ci + 1}/{n_chunks} 编码失败：{err[:200]}", "exporter")
            return ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, n_chunks)) as ex:
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
            if got > 0 and abs(got - expected) <= max(2, len(ranges)):
                app_core.info(f"时长校验存疑但帧数正确（{got} vs {expected}），继续", "exporter")
            else:
                app_core.warn(f"分块导出{why}，回退单遍滤镜", "exporter")
                return False

        if has_audio:
            _mux_audio_for_ranges(video_path, video_only, output_path, ranges, fps,
                                  ffmpeg_path=ffmpeg)
        else:
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
                  profile=None):
    """导出一组区间为一个文件（ranges 为左闭右闭的 (start, end)）。

    分段导出走这里：单段且起点是关键帧时用无损 copy，几乎瞬时。
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

    if ffmpeg_bin and keyframe_copy:
        try:
            if _try_copy_export(video_path, output_path, exclusive, fps, profile,
                                ffmpeg_bin, progress_cb, cancel_event, workers=2):
                return total_frames_to_export, total_frames_to_export
        except ExportCancelled:
            raise
        except Exception as exc:
            app_core.warn(f"分段无损直通异常，回退滤镜编码: {exc}", "exporter")

    if ffmpeg_bin and _export_ranges_with_ffmpeg_filters(
            video_path, output_path, exclusive, fps, quality,
            use_gpu, gpu_encoder, False, progress_cb, ffmpeg_path=ffmpeg_bin,
            preset=export_preset, cancel_event=cancel_event, profile=profile,
            ratio_base=0.02, ratio_span=0.96):
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
