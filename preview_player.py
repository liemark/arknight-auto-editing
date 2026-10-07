# preview_player.py — 视频预览播放器

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import threading
import os
import concurrent.futures
import time
from queue import Queue, Empty

from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                         FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)
from video_io import VideoIOThread, CMD_SEEK, CMD_SEEK_LATEST, CMD_PLAY, CMD_STOP
from timeline_widget import TimelineWidget
import app_core


class VideoPreviewPlayer(tk.Frame):
    def __init__(self, parent, settings, video_path=None, width=800, height=450):
        super().__init__(parent)
        self.settings = settings
        self.video_path = video_path

        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame_idx: int = 0

        self.canvas_w = width
        self.canvas_h = height

        self.pause_segments: list = []
        self.speed_segments: list = []
        self.clip_segments: list = []
        self.states_array = None
        self.diffs_array = None

        self.is_playing = False
        self._io: VideoIOThread | None = None
        self._frame_q: Queue = Queue(maxsize=2)
        self._canvas_img_id = None
        self._photo = None                 # 复用同一个 PhotoImage，避免每帧创建/销毁 Tk 图

        self._key_held: str | None = None
        self._key_after_id: str | None = None
        self._key_hold_fired: bool = False
        self._key_preview_id: str | None = None
        self._is_dragging: bool = False

        # 性能/测速相关
        self.profile = None                # gpu_caps.GpuProfile
        self.match_backend: str | None = None
        self._bench_running = False
        self._template_cache: dict = {}    # proc_res -> (configs, loaded)
        self._last_stats: dict = {}

        # 导出取消
        self._export_cancel = threading.Event()
        self._export_running = False
        self._export_t0 = 0.0
        self._seg_cancel = threading.Event()
        self._seg_running = False
        self._seg_t0 = 0.0

        # 跳过区快照缓存：掩码重算成本随暂停帧数线性增长，seek 频繁时必须缓存
        self._skip_segs_cache: list | None = None

        self._setup_ui()
        if video_path:
            self.load_video(video_path)

        self.settings.apply_pause_callback = self.apply_pause_mode
        self.settings.single_pause_callback = self.set_single_pause_mode
        self.settings.export_callback = self.export_video
        self.settings.segment_export_callback = self.export_segments
        self.settings.autotune_callback = self.run_autotune
        self.settings.cancel_export_callback = self.cancel_export
        self.settings.cancel_segment_export_callback = self.cancel_segment_export
        self.timeline.on_pause_select_cb = self._on_timeline_pause_select
        self.timeline.on_mask_changed_cb = self._invalidate_skip_cache

    # ==========================================================
    #  UI 构建
    # ==========================================================
    def _setup_ui(self):
        self.video_canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.video_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Button-1>", lambda e: self.video_canvas.focus_set())

        self.timeline = TimelineWidget(self)
        self.timeline.pack(fill=tk.X, padx=10)
        self.timeline.on_seek_cb = self._on_tl_seek
        self.timeline.on_handle_end_cb = self._on_tl_drag_end
        self.timeline.canvas.bind("<Button-1>", lambda e: self.video_canvas.focus_set(), add='+')

        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, pady=5)

        self.btn_play = ttk.Button(ctrl, text="▶ 播放", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="倍速:").pack(side=tk.LEFT, padx=(10, 2))
        self.preview_speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(
            ctrl, textvariable=self.preview_speed_var,
            values=["0.1x", "0.25x", "0.5x", "1x", "2x", "4x"], width=6, state="readonly")
        speed_combo.pack(side=tk.LEFT, padx=2)

        self.btn_analyze = ttk.Button(ctrl, text="自动模板分析", command=self._start_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        self.skip_trimmed = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="预览时跳过裁剪区", variable=self.skip_trimmed).pack(side=tk.LEFT, padx=5)

        self.lbl_time = ttk.Label(ctrl, text="00:00 / 00:00")
        self.lbl_time.pack(side=tk.RIGHT, padx=10)

        self.lbl_info = ttk.Label(self, text="就绪", foreground="#00CED1", font=("Consolas", 10))
        self.lbl_info.pack(fill=tk.X, padx=10, pady=2)

        # 实际生效的解码/匹配后端与实测耗时
        self.lbl_perf = ttk.Label(self, text="性能：尚未分析", foreground="#666666",
                                  font=("Consolas", 8))
        self.lbl_perf.pack(fill=tk.X, padx=10)

        hint = "← → 逐帧移动  |  空格 播放  |  时间轴：右键点击黄色块手动删除操作，鼠标中键平移"
        ttk.Label(self, text=hint, foreground="#555555", font=("Consolas", 8)).pack(
            fill=tk.X, padx=10, pady=(0, 2))

        self._render_loop()
        self.after_idle(self._bind_keys)

    # ==========================================================
    #  视频加载
    # ==========================================================
    def load_video(self, path: str):
        if self._io and self._io.is_alive():
            self._io.stop_and_quit()
            self._io = None
        while True:
            try:
                self._frame_q.get_nowait()
            except Empty:
                break

        self.video_path = path
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.pause_segments.clear()
        self.speed_segments.clear()
        self.clip_segments.clear()
        self.states_array = None
        self.diffs_array = None
        self.is_playing = False
        self.btn_play.config(text="▶ 播放")
        self._canvas_img_id = None
        self._photo = None
        self._invalidate_skip_cache()
        self._template_cache.clear()
        self.match_backend = None
        self._last_stats = {}
        self.lbl_perf.config(text="性能：尚未分析")
        self.timeline.selected_pause_id = None
        self.settings.set_selected_pause(None, "")

        self._io = VideoIOThread(path, self._frame_q)
        self._io.start()
        self.fps = self._io.fps
        self.total_frames = self._io.total

        self.timeline.total_frames = self.total_frames
        self.timeline.fps = self.fps
        self.timeline.zoom_level = 1.0
        self.timeline.scroll_offset = 0.0
        self.timeline.pause_segments = self.pause_segments
        self.timeline.speed_segments = self.speed_segments
        self.timeline.clip_segments = self.clip_segments
        self.timeline.current_frame_idx = 0
        self.timeline.mark_dirty()

        name, _ = os.path.splitext(path)
        self.settings.output_var.set(f"{name}_clipped.mp4")

        self._seek(0)
        self.timeline.redraw()

    # ==========================================================
    #  IO 线程命令封装
    # ==========================================================
    def _canvas_wh(self) -> tuple:
        cw = self.video_canvas.winfo_width() or self.canvas_w
        ch = self.video_canvas.winfo_height() or self.canvas_h
        return (max(1, cw), max(1, ch))

    def _speed_segs_snap(self) -> list:
        return [(s['start'], s['end'], s['type']) for s in self.speed_segments]

    def _invalidate_skip_cache(self):
        self._skip_segs_cache = None

    def _compute_skip_segs(self) -> list:
        segs = []
        for s in self.pause_segments:
            mode = s.get('mode', 'auto')
            start = s['start']
            if mode == 'all':
                segs.append((start, s['end'] + 1))
            elif mode == 'auto' and 'local_del_mask' in s:
                mask = np.asarray(s['local_del_mask'])
                if mask.size == 0:
                    continue
                is_del = (mask == 1) | (mask == 2)
                # 用 np.diff 找连续删除块边界，替代逐像素 Python 循环
                d = np.diff(is_del.astype(np.int8))
                starts = np.flatnonzero(d == 1) + 1
                ends = np.flatnonzero(d == -1) + 1
                if is_del[0]:
                    starts = np.concatenate(([0], starts))
                if is_del[-1]:
                    ends = np.concatenate((ends, [mask.size]))
                for a, b in zip(starts, ends):
                    if int(b) > int(a):
                        segs.append((start + int(a), start + int(b)))

        for s in self.clip_segments:
            ki, ko = s['keep_in'], s['keep_out']
            if ki > ko:
                segs.append((s['start'], s['end'] + 1))
            else:
                if ki > s['start']:
                    segs.append((s['start'], ki))
                if ko < s['end']:
                    segs.append((ko + 1, s['end'] + 1))
        return segs

    def _all_skip_segs_snap(self) -> list:
        if self._skip_segs_cache is None:
            self._skip_segs_cache = self._compute_skip_segs()
        return self._skip_segs_cache

    def _seek(self, frame_idx: int, skip_trim: bool = False):
        if not self._io:
            return
        self.current_frame_idx = frame_idx
        self.timeline.current_frame_idx = frame_idx
        self._io.send({
            'type': CMD_SEEK_LATEST,
            'frame': frame_idx,
            'canvas_wh': self._canvas_wh(),
            'pause_segs': self._all_skip_segs_snap(),
            'skip_trimmed': skip_trim,
        })

    def _send_play(self, start: int):
        if not self._io:
            return
        p = self.settings.get_params()

        speed_str = self.preview_speed_var.get().rstrip('x')
        try:
            speed = float(speed_str)
        except ValueError:
            speed = 1.0
        if speed >= 1.0:
            preview_step = max(1, int(speed))
            speed_multiplier = 1.0
        else:
            preview_step = 1
            speed_multiplier = 1.0 / max(speed, 0.01)

        self._io.send({
            'type': CMD_PLAY,
            'params': {
                'start_frame': start,
                'preview_step': preview_step,
                'speed_multiplier': speed_multiplier,
                'skip_trimmed': self.skip_trimmed.get(),
                'speedup_1x': p['speedup_1x'],
                'speedup_02': p['speedup_02'],
                'speedup_02_factor': p['speedup_02_factor'],
                'pause_segs': self._all_skip_segs_snap(),
                'speed_segs': self._speed_segs_snap(),
                'canvas_wh': self._canvas_wh(),
            }
        })

    def _send_stop(self):
        if self._io:
            self._io.send({'type': CMD_STOP})

    # ==========================================================
    #  键盘快捷键
    # ==========================================================

    _KEY_PREVIEW_MS = 150

    def _bind_keys(self):
        root = self.winfo_toplevel()
        root.bind('<Left>', self._on_key_press_left, add='+')
        root.bind('<Right>', self._on_key_press_right, add='+')
        root.bind('<KeyRelease-Left>', self._on_key_release, add='+')
        root.bind('<KeyRelease-Right>', self._on_key_release, add='+')
        root.bind('<space>', self._on_key_space)
        for cls in ('TButton', 'Button', 'TCheckbutton', 'TRadiobutton', 'TCombobox', 'TNotebook'):
            root.bind_class(cls, '<space>', lambda e: 'break')

    def _on_key_press_left(self, event):
        if self._key_held == 'Left':
            return
        self._key_held = 'Left'
        self._key_hold_fired = False
        self._step_frame(-1, seek=True)
        self._key_after_id = self.after(400, self._start_repeat, 'Left')

    def _on_key_press_right(self, event):
        if self._key_held == 'Right':
            return
        self._key_held = 'Right'
        self._key_hold_fired = False
        self._step_frame(+1, seek=True)
        self._key_after_id = self.after(400, self._start_repeat, 'Right')

    def _on_key_release(self, event):
        direction = event.keysym
        if self._key_held != direction:
            return
        self._key_held = None
        if self._key_after_id:
            self.after_cancel(self._key_after_id)
            self._key_after_id = None
        if self._key_preview_id:
            self.after_cancel(self._key_preview_id)
            self._key_preview_id = None
        if self._key_hold_fired:
            while True:
                try:
                    self._frame_q.get_nowait()
                except Empty:
                    break
            self._do_preview_seek()
        self._key_hold_fired = False

    def _on_key_space(self, event):
        focused = self.focus_get()
        if isinstance(focused, (ttk.Entry, tk.Entry, ttk.Combobox)):
            return
        self.toggle_play()
        return 'break'

    def _start_repeat(self, direction: str):
        self._key_hold_fired = True
        self._schedule_preview()
        self._repeat_frame(direction)

    def _repeat_frame(self, direction: str):
        if self._key_held != direction:
            return
        delta = -1 if direction == 'Left' else +1
        self._step_frame(delta, seek=False)
        speed = self.settings.key_repeat_speed_var.get()
        interval = max(16, int(1000 / speed))
        self._key_after_id = self.after(interval, self._repeat_frame, direction)

    def _schedule_preview(self):
        self._key_preview_id = self.after(self._KEY_PREVIEW_MS, self._preview_tick)

    def _preview_tick(self):
        if not self._key_held:
            return
        self._do_preview_seek()
        self._key_preview_id = self.after(self._KEY_PREVIEW_MS, self._preview_tick)

    def _do_preview_seek(self):
        if not self._io or self.total_frames <= 0:
            return
        self._io.send({
            'type': CMD_SEEK_LATEST,
            'frame': self.current_frame_idx,
            'canvas_wh': self._canvas_wh(),
            'pause_segs': [],
            'skip_trimmed': False,
        })

    def _step_frame(self, delta: int, seek: bool = True):
        if self.total_frames <= 0:
            return
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        if new_idx == self.current_frame_idx:
            return
        self.current_frame_idx = new_idx
        self.timeline.current_frame_idx = new_idx
        self.timeline._ensure_pointer_visible()
        self.timeline.update_pointer()
        self._update_labels()
        if seek:
            self._seek(new_idx, skip_trim=False)

    # ==========================================================
    #  播放控制
    # ==========================================================
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ 播放")
            self._send_stop()
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸ 暂停")
            self._send_play(self.current_frame_idx)

    def _on_tl_seek(self, frame_idx: int):
        self._is_dragging = True
        self._seek(frame_idx, skip_trim=False)

    def _on_tl_drag_end(self):
        self._is_dragging = False
        self._invalidate_skip_cache()
        while True:
            try:
                self._frame_q.get_nowait()
            except Empty:
                break
        if self.is_playing:
            self._send_play(self.current_frame_idx)

    def _on_timeline_pause_select(self, seg_id: int):
        for seg in self.pause_segments:
            if seg['id'] == seg_id:
                self.settings.set_selected_pause(seg_id, seg.get('mode', 'auto'))
                break

    def _render_loop(self):
        try:
            idx, rgb = self._frame_q.get_nowait()
            if not self._key_hold_fired and not self._is_dragging:
                self.current_frame_idx = idx
                self.timeline.current_frame_idx = idx
                self.timeline._ensure_pointer_visible()
            self._display_rgb(rgb)
        except Empty:
            pass
        self.timeline.update_pointer()
        self.after(16, self._render_loop)

    def _display_rgb(self, rgb: np.ndarray):
        img = PIL.Image.fromarray(rgb)
        cw, ch = self._canvas_wh()
        if self._canvas_img_id is None or self._photo is None \
                or self._photo.width() != img.width or self._photo.height() != img.height:
            self.video_canvas.delete("all")
            self._photo = PIL.ImageTk.PhotoImage(image=img)
            self._canvas_img_id = self.video_canvas.create_image(cw // 2, ch // 2, image=self._photo)
        else:
            # 复用同一个 Tk 图像对象并 paste，避免每帧创建/销毁 PhotoImage
            self._photo.paste(img)
            self.video_canvas.coords(self._canvas_img_id, cw // 2, ch // 2)
        self._update_labels()

    # ==========================================================
    #  标签更新
    # ==========================================================
    def _update_labels(self):
        cur = self.current_frame_idx
        cur_s = cur / self.fps if self.fps else 0
        tot_s = self.total_frames / self.fps if self.fps else 0
        self.lbl_time.config(text=f"{self._fmt(cur_s)} / {self._fmt(tot_s)}")

        info = "普通区域"
        for seg in self.pause_segments:
            if seg['start'] <= cur <= seg['end']:
                mode_str = {'all': '全删', 'keep': '全保留',
                            'auto': '按设置裁剪'}.get(seg.get('mode', 'auto'), '')
                bd_diff = seg.get('boundary_diff', 0.0)
                info = f"暂停 | ID: {seg['id']} | 模式: {mode_str} | 边界差异: {bd_diff:.1f}"
                break
        else:
            p = self.settings.get_params()
            for seg in self.speed_segments:
                if seg['start'] <= cur <= seg['end']:
                    t = seg['type']
                    name = {FRAME_TYPE_1X: '1x', FRAME_TYPE_2X: '2x',
                            FRAME_TYPE_0_2X: '0.2x'}.get(t, '?')
                    eff = 1
                    if t == FRAME_TYPE_1X and p.get('speedup_1x'):
                        eff = 2
                    if t == FRAME_TYPE_0_2X and p.get('speedup_02'):
                        eff = p.get('speedup_02_factor', 10)
                    speed_str = self.preview_speed_var.get().rstrip('x')
                    try:
                        pspeed = float(speed_str)
                    except ValueError:
                        pspeed = 1.0
                    total_eff = eff * pspeed
                    info = f"变速 {name}" + (f"（预览 {total_eff:g}x）" if total_eff != 1 else "")
                    break
        self.lbl_info.config(text=info)

    @staticmethod
    def _fmt(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    # ==========================================================
    #  单段与批量暂停模式控制
    # ==========================================================
    def apply_pause_mode(self, mode: str):
        import analyzer
        p = self.settings.get_params()
        boundary_thresh = p['compare'].get('boundary_thresh', 5.0)
        motion_thresh = p['compare'].get('motion_thresh', 2.0)
        still_time = p['compare'].get('still_time_thresh', 0.1)
        still_frames = max(2, int(self.fps * still_time))

        for seg in self.pause_segments:
            if mode == 'auto':
                if self.diffs_array is not None:
                    new_mask, _ = analyzer._analyze_pause_mask(
                        seg['start'], seg['end'], self.diffs_array, still_frames, motion_thresh)
                    seg['local_del_mask'] = new_mask

                if seg.get('boundary_diff', 0.0) < boundary_thresh:
                    seg['mode'] = 'all'
                else:
                    seg['mode'] = 'auto'
            else:
                seg['mode'] = mode

        if self.settings.selected_pause_id is not None:
            for seg in self.pause_segments:
                if seg['id'] == self.settings.selected_pause_id:
                    self.settings.set_selected_pause(self.settings.selected_pause_id, seg['mode'])
                    break

        self._invalidate_skip_cache()
        self.timeline.mark_dirty()
        self.timeline.redraw()
        if self.is_playing:
            self._send_play(self.current_frame_idx)

    def set_single_pause_mode(self, seg_id: int, mode: str):
        import analyzer
        p = self.settings.get_params()
        motion_thresh = p['compare'].get('motion_thresh', 2.0)
        still_time = p['compare'].get('still_time_thresh', 0.1)
        still_frames = max(2, int(self.fps * still_time))

        for seg in self.pause_segments:
            if seg['id'] == seg_id:
                if mode == 'auto' and self.diffs_array is not None:
                    new_mask, _ = analyzer._analyze_pause_mask(
                        seg['start'], seg['end'], self.diffs_array, still_frames, motion_thresh)
                    seg['local_del_mask'] = new_mask

                seg['mode'] = mode

                self.settings.set_selected_pause(seg_id, seg['mode'])
                self._invalidate_skip_cache()
                self.timeline.mark_dirty()
                self.timeline.redraw()
                if self.is_playing:
                    self._send_play(self.current_frame_idx)
                break

    # ==========================================================
    #  性能：硬件探测 + 匹配后端测速
    # ==========================================================
    def _proc_res_for_video(self) -> tuple:
        p = self.settings.get_params()
        proc_res = list(p['proc_res'])
        if not self.video_path:
            return tuple(proc_res)
        try:
            cap_tmp = cv2.VideoCapture(self.video_path)
            ret, f = cap_tmp.read()
            cap_tmp.release()
            if ret and proc_res[1] == 225:
                h, ww = f.shape[:2]
                proc_res[1] = int(proc_res[0] * h / ww)
        except Exception:
            pass
        return tuple(proc_res)

    def get_templates(self, proc_res: tuple):
        cached = self._template_cache.get(proc_res)
        if cached is None:
            import analyzer
            # load_templates 内部已剔除「区分不了类别」的模板并写日志
            cached = analyzer.load_templates(proc_res)
            self._template_cache[proc_res] = cached
        return cached

    def ensure_profile(self, force: bool = False):
        import gpu_caps
        try:
            self.profile = gpu_caps.detect(
                self.settings.get_params().get('ffmpeg_path'), force=force)
        except Exception as exc:
            app_core.error(f"硬件探测失败: {exc}", "preview")
        return self.profile

    def run_autotune(self, force: bool = False, on_done=None):
        """实测各后端选最快的（结果写入缓存并被后续分析使用）。"""
        if not self.video_path:
            if on_done:
                on_done("请先加载视频再测速")
            return
        if self._bench_running:
            return
        self._bench_running = True
        p = self.settings.get_params()
        proc_res = self._proc_res_for_video()
        self.settings.set_bench_text("正在实测各匹配后端…")

        def worker():
            import matcher
            try:
                profile = self.ensure_profile()
                configs, loaded = self.get_templates(proc_res)
                if loaded == 0:
                    msg = "未找到模板，无法测速"
                    self.after(0, lambda: self.settings.set_bench_text(msg))
                    return
                sample = matcher.sample_gray_frames(
                    self.video_path, proc_res, count=64,
                    decode_backend=p.get('decode_backend', 'auto'),
                    ffmpeg_path=p.get('ffmpeg_path'))
                if len(sample) == 0:
                    self.after(0, lambda: self.settings.set_bench_text("抽帧失败，无法测速"))
                    return
                compiled = matcher.compile_templates(
                    configs, proc_res, frame_wh=(sample.shape[2], sample.shape[1]))
                key = matcher.bench_cache_key(compiled, profile, proc_res)
                cached = None if force else matcher.cached_choice(key)
                if cached and cached.get('chosen'):
                    chosen = cached['chosen']
                    text = (f"（缓存）已选 {matcher.backend_label(chosen)} · " + " ".join(
                        f"{matcher.backend_label(r['backend'])} {r['ms_per_frame']:.3f}"
                        for r in cached.get('results', []) if r.get('ok')))
                else:
                    results, chosen = matcher.autotune(
                        compiled, configs, p['thresholds'], proc_res, sample,
                        profile=profile)
                    matcher.store_choice(key, chosen or matcher.BACKEND_CV2_DIRECT, results)
                    text = matcher.format_results(results, chosen)
                self.match_backend = chosen

                def apply():
                    self.settings.set_bench_text(text)
                    self._update_perf_label()
                    if on_done:
                        on_done(text)
                self.after(0, apply)
            except Exception as exc:
                msg = f"测速失败：{type(exc).__name__}: {exc}"
                app_core.error(msg, "preview")
                self.after(0, lambda m=msg: self.settings.set_bench_text(m))
            finally:
                self._bench_running = False

        threading.Thread(target=worker, daemon=True, name="autotune").start()

    def _update_perf_label(self, stats: dict | None = None):
        import pipeline
        import matcher
        if stats:
            self._last_stats = stats
        st = self._last_stats
        if not st:
            txt = "性能：尚未分析"
            if self.match_backend:
                txt += f" · 匹配将用 {matcher.backend_label(self.match_backend)}"
            self.lbl_perf.config(text=txt)
            return
        dec = pipeline.decode_backend_label(st.get('decode_backend', '?'))
        mb = matcher.backend_label(st.get('match_backend', '?'))
        skip = st.get('skip_ratio', 0.0) * 100
        self.lbl_perf.config(
            text=(f"性能：{dec} + {mb} · 端到端 {st.get('ms_per_frame', 0):.3f} ms/帧 · "
                  f"匹配 {st.get('match_ms_per_frame', 0):.3f} ms/帧 · "
                  f"静止帧跳过 {skip:.0f}%"))

    # ==========================================================
    #  模板分析
    # ==========================================================
    def _start_analysis(self):
        if not self.video_path:
            return
        from tkinter import messagebox
        import analyzer
        import matcher
        import pipeline

        self.btn_analyze.config(state=tk.DISABLED, text="分析中...")
        p = self.settings.get_params()
        proc_res = self._proc_res_for_video()

        def worker():
            profile = self.ensure_profile()
            configs, loaded = self.get_templates(proc_res)
            if loaded == 0:
                self.after(0, lambda: messagebox.showwarning(
                    "模板缺失", "未找到可用模板，将标记所有帧为普通帧。"))

            decode_backend = p.get('decode_backend', 'auto')
            ffmpeg_path = p.get('ffmpeg_path')
            backend_note = ""
            try:
                backend_key = pipeline.normalize_decode_backend(decode_backend)
            except Exception as norm_exc:
                backend_key = pipeline.DECODE_BACKEND_OPENCV
                backend_note = f"（设置无效，已回退 OpenCV：{norm_exc}）"
            resolved_decode = pipeline.resolve_decode_backend(
                backend_key, profile=profile, ffmpeg_path=ffmpeg_path)
            decode_label = pipeline.decode_backend_label(resolved_decode) + backend_note

            # 匹配后端：还没有测速结果就先测一遍（可在「性能/GPU」页关闭）
            match_choice = p.get('match_backend', 'auto')
            if p.get('match_autotune', True) and self.match_backend is None:
                self.run_autotune()
                deadline = time.time() + 120
                while self._bench_running and time.time() < deadline:
                    time.sleep(0.2)
            if self.match_backend:
                match_choice = self.match_backend
            resolved_match = matcher.resolve_backend(match_choice, profile)
            match_label = matcher.backend_label(resolved_match)

            label = f"{decode_label}+{match_label}"

            def prog(r):
                self.after(0, lambda rr=r: self.btn_analyze.config(
                    text=f"{label} {int(rr * 100)}%"))

            app_core.info(
                f"开始分析：解码={pipeline.decode_backend_label(resolved_decode)} "
                f"匹配={match_label} proc_res={proc_res} batch={p['batch']} "
                f"线程={p['threads']} 静止帧跳过={p.get('skip_identical', True)}", "preview")
            try:
                states, diffs, context = analyzer.analyze_video_with_context(
                    self.video_path, configs, p['thresholds'],
                    proc_res, p['batch'], p['threads'], prog,
                    decode_backend=resolved_decode,
                    ffmpeg_path=ffmpeg_path,
                    match_backend=resolved_match,
                    skip_identical=p.get('skip_identical', True),
                    on_stats=lambda st: self.after(0, lambda s=st: self._update_perf_label(s)),
                )
            except Exception as exc:
                err_msg = (
                    f"分析失败（解码 {decode_label} / 匹配 {match_label}）:\n"
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"可改回「不加速(OpenCV)」+「cv2 直算」后重试。"
                )
                self.after(
                    0,
                    lambda msg=err_msg: (
                        self.btn_analyze.config(state=tk.NORMAL, text="自动模板分析"),
                        messagebox.showerror("分析失败", msg),
                    ),
                )
                return

            pauses, speeds = analyzer.build_segments(
                states, diffs, self.video_path, proc_res, p['compare'], self.fps, prog,
                analysis_context=context,
            )
            used_ctx = analyzer.analysis_context_skips_second_scan(
                context, pauses, len(states))

            self.after(
                0,
                lambda: self._finish_analysis(
                    states, diffs, pauses, speeds,
                    decode_label=decode_label, match_label=match_label,
                    used_ctx=used_ctx,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_analysis(self, states, diffs, pauses, speeds,
                         decode_label: str = "", match_label: str = "",
                         used_ctx: bool = False):
        from tkinter import messagebox
        self.states_array = states
        self.diffs_array = diffs
        self.pause_segments = pauses
        self.speed_segments = speeds
        self.clip_segments = self._build_clip_segments(pauses, self.total_frames)
        self._invalidate_skip_cache()

        self.timeline.pause_segments = self.pause_segments
        self.timeline.speed_segments = self.speed_segments
        self.timeline.clip_segments = self.clip_segments

        self.timeline.selected_pause_id = None
        self.settings.set_selected_pause(None, "")

        self.timeline.mark_dirty()
        self.btn_analyze.config(state=tk.NORMAL, text="自动模板分析")
        self.timeline.redraw()
        self._update_perf_label()

        boundary_note = ("边界: 第一遍上下文（已跳过二次扫片）" if used_ctx
                         else "边界: 回退二次扫片或无暂停")
        st = self._last_stats or {}
        perf = ""
        if st:
            perf = (f"\n实测：端到端 {st.get('ms_per_frame', 0):.3f} ms/帧 · "
                    f"匹配 {st.get('match_ms_per_frame', 0):.3f} ms/帧 · "
                    f"静止帧跳过 {st.get('skip_ratio', 0) * 100:.0f}%")
        messagebox.showinfo(
            "分析完成",
            f"解码后端: {decode_label}\n"
            f"匹配后端: {match_label}\n"
            f"{boundary_note}{perf}\n"
            f"识别到 {len(pauses)} 处暂停，{len(speeds)} 个变速区间。",
        )

    @staticmethod
    def _build_clip_segments(pauses: list, total_frames: int) -> list:
        if total_frames <= 0:
            return []
        occupied = sorted([(seg['start'], seg['end']) for seg in pauses])
        clips = []
        clip_id = 0
        prev_end = -1
        for ps, pe in occupied:
            gap_start, gap_end = prev_end + 1, ps - 1
            if gap_end >= gap_start:
                clips.append({'id': clip_id, 'start': gap_start, 'end': gap_end,
                              'keep_in': gap_start, 'keep_out': gap_end})
                clip_id += 1
            prev_end = pe
        tail_start, tail_end = prev_end + 1, total_frames - 1
        if tail_end >= tail_start:
            clips.append({'id': clip_id, 'start': tail_start, 'end': tail_end,
                          'keep_in': tail_start, 'keep_out': tail_end})
        return clips

    # ==========================================================
    #  导出
    # ==========================================================
    def cancel_export(self):
        if self._export_running:
            self._export_cancel.set()
            self.settings.export_status_var.set("正在取消…")

    def cancel_segment_export(self):
        if self._seg_running:
            self._seg_cancel.set()
            self.settings.segment_export_status_var.set("正在取消…")

    @staticmethod
    def _eta_text(ratio: float, t0: float) -> str:
        if ratio <= 0.01 or t0 <= 0:
            return ""
        elapsed = time.perf_counter() - t0
        remain = elapsed * (1.0 - ratio) / ratio
        if remain < 1 or remain > 86400:
            return ""
        m, s = divmod(int(remain), 60)
        return f" · 剩余约 {m:02d}:{s:02d}"

    def export_video(self):
        from tkinter import messagebox
        import analyzer
        import exporter

        if not self.video_path:
            return messagebox.showerror("错误", "请先加载视频")
        p = self.settings.get_params()
        if not p['output']:
            return messagebox.showerror("错误", "请先设置输出路径")

        states = self.states_array if self.states_array is not None \
            else np.zeros(self.total_frames, dtype=np.int8)
        self._export_cancel.clear()
        self._export_running = True
        self._export_t0 = time.perf_counter()
        self.settings.set_export_running(True)
        self.settings.export_progress_var.set(0)
        self.settings.export_status_var.set("准备导出…")

        def worker():
            to_del = analyzer.build_delete_set(
                self.total_frames, states, self.pause_segments, self.speed_segments,
                self.clip_segments, p['speedup_1x'], p['speedup_02'], p['speedup_02_factor'])

            # 把「要删多少帧」写进事件区：一眼能看出是分析没判出暂停，还是掩码没删
            try:
                n_del = int(np.count_nonzero(to_del))
                n_all = sum(1 for s in self.pause_segments if s.get('mode') == 'all')
                empty = sum(1 for s in self.pause_segments
                            if not np.any(np.asarray(s.get('local_del_mask', []))))
                app_core.info(
                    f"导出前统计：删除 {n_del}/{self.total_frames} 帧 · "
                    f"暂停段 {len(self.pause_segments)} 个（整段删 {n_all}，掩码全空 {empty}）· "
                    f"剪辑段 {len(self.clip_segments)} 个", "export")
            except Exception:
                pass

            def prog(ratio, written):
                eta = self._eta_text(ratio, self._export_t0)
                self.after(0, lambda r=ratio, w=int(written), e=eta: (
                    self.settings.export_progress_var.set(r * 100),
                    self.settings.export_status_var.set(f"写入 {int(r * 100)}% · {w} 帧{e}")))

            def phase(text):
                # 显示当前实际阶段，避免界面一直停在「准备导出…」
                self.after(0, lambda t=text: self.settings.export_status_var.set(t))

            try:
                profile = self.ensure_profile()
                written, total = analyzer.export_video(
                    self.video_path, p['output'], to_del, self.fps, p['quality'], prog,
                    use_gpu=p.get('export_use_gpu', False),
                    gpu_encoder=p.get('gpu_encoder', ''),
                    ffmpeg_path=p.get('ffmpeg_path'),
                    export_preset=p.get('export_preset') or None,
                    export_workers=p.get('export_workers') or None,
                    keyframe_copy=p.get('keyframe_copy', True),
                    export_audio=p.get('export_audio', True),
                    cancel_event=self._export_cancel,
                    profile=profile,
                    status_cb=phase)
                enc = p.get('gpu_encoder', '') or '自动'
                workers = p.get('export_workers') or '自动'
                self.after(0, lambda: self.settings.export_status_var.set(
                    f"完成！{written}/{total} 帧（编码器 {enc}）"))
                self.after(0, lambda: messagebox.showinfo(
                    "导出完成",
                    f"输出：{p['output']}\n总帧：{total}，保留：{written}\n"
                    f"编码器：{enc} · 并发 {workers}\n"
                    f"（已保留原始音频；无损直通命中时不重编码视频）"))
            except exporter.ExportCancelled:
                self.after(0, lambda: self.settings.export_status_var.set("已取消"))
                self.after(0, lambda: messagebox.showinfo(
                    "已取消", "导出已取消，临时文件已清理。"))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self.settings.export_status_var.set("导出失败"))
                self.after(0, lambda m=err: messagebox.showerror("导出失败", m))
            finally:
                self._export_running = False
                self.after(0, lambda: self.settings.set_export_running(False))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _speed_label(state: int) -> str:
        return {FRAME_TYPE_2X: '2x', FRAME_TYPE_1X: '1x', FRAME_TYPE_0_2X: '0.2x',
                FRAME_TYPE_NORMAL: 'other'}.get(state, 'other')

    def _build_valid_segments_for_export(self, states: np.ndarray, split_by_speed: bool,
                                         merge_pause: bool) -> list:
        import analyzer
        to_del = analyzer.build_delete_set(
            self.total_frames, states, self.pause_segments, self.speed_segments,
            self.clip_segments, speedup_1x=False, speedup_02=False, speedup_02_factor=1)
        valid = ~to_del

        segs = []
        i = 0
        while i < self.total_frames:
            if not valid[i]:
                i += 1
                continue

            cur_state = int(states[i])
            is_pause = (cur_state == FRAME_TYPE_PAUSE)
            speed_label = self._speed_label(cur_state)

            if is_pause and merge_pause:
                p_end = self.total_frames - 1
                for pseg in self.pause_segments:
                    if pseg['start'] <= i <= pseg['end']:
                        p_end = pseg['end']
                        break

                ranges = []
                j = i
                while j <= p_end and j < self.total_frames:
                    if valid[j] and int(states[j]) == FRAME_TYPE_PAUSE:
                        rs = j
                        while j <= p_end and j < self.total_frames and valid[j] \
                                and int(states[j]) == FRAME_TYPE_PAUSE:
                            j += 1
                        ranges.append((rs, j - 1))
                    else:
                        j += 1

                segs.append({'ranges': ranges, 'label': 'pause_merged'})
                i = p_end + 1
            else:
                s = i
                while i < self.total_frames and valid[i]:
                    st = int(states[i])
                    if is_pause:
                        if st != FRAME_TYPE_PAUSE:
                            break
                    else:
                        if st == FRAME_TYPE_PAUSE:
                            break
                        if split_by_speed and self._speed_label(st) != speed_label:
                            break
                    i += 1
                e = i - 1

                label = 'pause' if is_pause else (speed_label if split_by_speed else 'normal')
                segs.append({'ranges': [(s, e)], 'label': label})

        merged_segs = []
        for seg in segs:
            if not merged_segs:
                merged_segs.append(seg)
            else:
                last_seg = merged_segs[-1]
                if last_seg['label'] == seg['label'] and 'pause' not in seg['label']:
                    last_seg['ranges'].extend(seg['ranges'])
                else:
                    merged_segs.append(seg)

        return merged_segs

    def export_segments(self):
        from tkinter import messagebox
        import analyzer
        import exporter

        if not self.video_path:
            return messagebox.showerror("错误", "请先加载视频")

        p = self.settings.get_params()
        out_path = p.get('output') or ""
        if not out_path:
            return messagebox.showerror("错误", "请先设置导出路径（用于确定分段输出目录）")

        out_root = os.path.dirname(out_path) or os.getcwd()
        base = os.path.splitext(os.path.basename(out_path))[0] or "segments"
        out_dir = os.path.join(out_root, f"{base}_segments")
        os.makedirs(out_dir, exist_ok=True)

        states = self.states_array if self.states_array is not None \
            else np.zeros(self.total_frames, dtype=np.int8)

        split = self.settings.segment_split_by_speed_var.get()
        merge_pause = self.settings.merge_pause_ops_var.get()
        segs = self._build_valid_segments_for_export(states, split, merge_pause)
        if not segs:
            return messagebox.showwarning("提示", "当前时间轴没有可导出的有效片段。")

        self._seg_cancel.clear()
        self._seg_running = True
        self._seg_t0 = time.perf_counter()
        self.settings.set_segment_export_running(True)
        self.settings.segment_export_progress_var.set(0)

        def worker():
            total = len(segs)
            pad = max(1, len(str(total)))
            if p.get('export_use_gpu', False):
                max_w = max(1, int(p.get('export_workers') or 1))
            else:
                max_w = max(1, (os.cpu_count() or 2) // 2)
            enc = p.get('gpu_encoder') or ('libx264' if not p.get('export_use_gpu') else '自动')

            completed = 0
            succeeded = 0
            failures = []
            lock = threading.Lock()

            def export_single(idx_seg):
                idx, seg = idx_seg
                if self._seg_cancel.is_set():
                    return
                stem = f"{idx:0{pad}d}_{seg['label']}"
                final_path = os.path.join(out_dir, f"{stem}.mp4")
                success = False
                error_text = ""
                try:
                    written, _ = analyzer.export_ranges(
                        self.video_path, final_path, seg['ranges'],
                        self.fps, p['quality'],
                        use_gpu=p.get('export_use_gpu', False),
                        gpu_encoder=p.get('gpu_encoder', ''),
                        ffmpeg_path=p.get('ffmpeg_path'),
                        export_preset=p.get('export_preset') or None,
                        keyframe_copy=p.get('keyframe_copy', True),
                        export_audio=p.get('export_audio', True),
                        cancel_event=self._seg_cancel)
                    success = written > 0 and os.path.isfile(final_path) \
                        and os.path.getsize(final_path) > 0
                    if not success:
                        error_text = "未生成有效输出文件"
                except exporter.ExportCancelled:
                    return
                except Exception as e:
                    error_text = str(e)
                    app_core.error(f"分段 {stem} 导出失败: {error_text}", "preview")
                if not success and os.path.isfile(final_path):
                    try:
                        os.remove(final_path)
                    except OSError as cleanup_error:
                        error_text += f"；清理失败: {cleanup_error}"

                nonlocal completed, succeeded
                with lock:
                    completed += 1
                    if success:
                        succeeded += 1
                    else:
                        failures.append((stem, error_text))
                    ratio = completed / total
                    eta = self._eta_text(ratio, self._seg_t0)
                    self.after(0, lambda r=ratio, c=completed, ok=succeeded, t=total, e=eta: (
                        self.settings.segment_export_progress_var.set(r * 100),
                        self.settings.segment_export_status_var.set(
                            f"处理 {c}/{t}，成功 {ok}，失败 {c - ok} · 并发 {max_w} · "
                            f"编码器 {enc}{e}")))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                list(executor.map(export_single, enumerate(segs, start=1)))

            canceled = self._seg_cancel.is_set()
            failed = len(failures)
            self.after(0, lambda: self.settings.segment_export_status_var.set(
                ("已取消" if canceled else
                 f"完成：成功 {succeeded}/{total}，失败 {failed}"
                 f"（分段默认不保留音频 · 并发 {max_w} · 编码器 {enc}）")))
            details = "\n".join(f"- {name}: {error[:500]}" for name, error in failures[:3])
            if canceled:
                self.after(0, lambda: messagebox.showinfo(
                    "已取消", f"分段导出已取消（已完成 {succeeded}/{total}）。"))
            elif failed:
                self.after(0, lambda msg=details: messagebox.showwarning(
                    "分段导出完成",
                    f"输出目录：{out_dir}\n成功：{succeeded}/{total}\n失败：{failed}/{total}"
                    + (f"\n\n部分错误：\n{msg}" if msg else "")))
            else:
                self.after(0, lambda: messagebox.showinfo(
                    "分段导出完成",
                    f"输出目录：{out_dir}\n成功：{succeeded}/{total}\n"
                    f"并发：{max_w} · 编码器：{enc}\n说明：分段导出默认不保留音频。"))
            self._seg_running = False
            self.after(0, lambda: self.settings.set_segment_export_running(False))

        threading.Thread(target=worker, daemon=True).start()
