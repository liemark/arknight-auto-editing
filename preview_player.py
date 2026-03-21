# preview_player.py — 视频预览播放器
# 职责：UI 组装、视频加载/播放控制、模板分析、导出
# 时间轴绘制/交互 → timeline_widget.TimelineWidget
# 视频 IO 线程     → video_io.VideoIOThread

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import threading
from queue import Queue, Empty

from frame_types import FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X
from video_io import VideoIOThread, CMD_SEEK, CMD_SEEK_LATEST, CMD_PLAY, CMD_STOP
from timeline_widget import TimelineWidget


class VideoPreviewPlayer(tk.Frame):
    """
    视频预览播放器主 Widget。
    布局：
      ┌─────────────────────┐
      │   video_canvas      │  ← BGR 帧显示
      ├─────────────────────┤
      │   TimelineWidget    │  ← 时间轴（独立组件）
      ├─────────────────────┤
      │   控制栏            │
      ├─────────────────────┤
      │   状态栏            │
      └─────────────────────┘
    """

    def __init__(self, parent, settings, video_path=None, width=800, height=450):
        super().__init__(parent)
        self.settings   = settings
        self.video_path = video_path

        self.total_frames: int   = 0
        self.fps:          float = 30.0
        self.current_frame_idx: int = 0

        self.canvas_w = width
        self.canvas_h = height

        self.pause_segments: list = []
        self.speed_segments:  list = []
        self.clip_segments:   list = []   # 手动裁剪段（删两端保中间）
        self.states_array         = None

        self.is_playing = False
        self._io: VideoIOThread | None = None
        self._frame_q: Queue = Queue(maxsize=2)
        self._canvas_img_id = None

        # 键盘连续移动状态
        self._key_held:       str | None = None   # 'Left' or 'Right'
        self._key_after_id:   str | None = None   # after() 句柄（移动循环）
        self._key_hold_fired: bool       = False  # 是否已进入连续移动阶段
        self._key_preview_id: str | None = None   # after() 句柄（预览刷新）
        # 连续移动预览刷新间隔（ms）：每隔此时间发一次 seek 显示画面
        _KEY_PREVIEW_MS = 150

        self._setup_ui()
        if video_path:
            self.load_video(video_path)

    # ==========================================================
    #  UI 构建
    # ==========================================================
    def _setup_ui(self):
        # 视频画布
        self.video_canvas = tk.Canvas(self, width=self.canvas_w,
                                      height=self.canvas_h, bg="black")
        self.video_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Button-1>",
                               lambda e: self.video_canvas.focus_set())

        # 时间轴组件
        self.timeline = TimelineWidget(self)
        self.timeline.pack(fill=tk.X, padx=10)
        self.timeline.on_seek_cb       = self._on_tl_seek
        self.timeline.on_handle_end_cb = self._on_tl_drag_end
        # 点击时间轴时把焦点移离输入控件
        self.timeline.canvas.bind("<Button-1>",
            lambda e: self.video_canvas.focus_set(), add='+')

        # 控制栏
        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, pady=5)

        self.btn_play = ttk.Button(ctrl, text="▶ 播放", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="倍速:").pack(side=tk.LEFT, padx=(10, 2))
        # < 1x 靠拉长帧间隔实现慢放；≥ 1x 靠抽帧实现加速
        self.preview_speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(
            ctrl,
            textvariable=self.preview_speed_var,
            values=["0.1x", "0.25x", "0.5x", "1x", "2x", "4x"],
            width=6, state="readonly")
        speed_combo.pack(side=tk.LEFT, padx=2)

        self.btn_analyze = ttk.Button(ctrl, text="自动模板分析",
                                      command=self._start_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        self.skip_trimmed = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="预览时跳过裁剪区",
                        variable=self.skip_trimmed).pack(side=tk.LEFT, padx=5)

        self.lbl_time = ttk.Label(ctrl, text="00:00 / 00:00")
        self.lbl_time.pack(side=tk.RIGHT, padx=10)

        # 状态栏
        self.lbl_info = ttk.Label(self, text="就绪",
                                  foreground="#00CED1", font=("Consolas", 10))
        self.lbl_info.pack(fill=tk.X, padx=10, pady=2)

        # 操作提示
        hint = ("← → 逐帧/连续移动  |  空格 播放/暂停  |  "
                "连续移动速度可在「基本」设置里调整")
        ttk.Label(self, text=hint, foreground="#555555",
                  font=("Consolas", 8)).pack(fill=tk.X, padx=10, pady=(0, 2))

        self._render_loop()

        # 键盘事件绑定到顶层窗口（避免焦点问题）
        # 使用 after_idle 等 UI 完全初始化后再绑定
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
        self.is_playing   = False
        self.btn_play.config(text="▶ 播放")
        self._canvas_img_id = None

        # 启动新 IO 线程
        self._io = VideoIOThread(path, self._frame_q)
        self._io.start()
        self.fps          = self._io.fps
        self.total_frames = self._io.total

        # 同步到时间轴
        self.timeline.total_frames  = self.total_frames
        self.timeline.fps           = self.fps
        self.timeline.zoom_level    = 1.0
        self.timeline.scroll_offset = 0.0
        self.timeline.pause_segments = self.pause_segments
        self.timeline.speed_segments  = self.speed_segments
        self.timeline.clip_segments   = self.clip_segments
        self.timeline.current_frame_idx = 0
        self.timeline.mark_dirty()

        if not self.settings.output_var.get():
            import os
            name, _ = os.path.splitext(path)
            self.settings.output_var.set(f"{name}_clipped.mp4")

        self._seek(0)
        self.timeline.redraw()

    # ==========================================================
    #  IO 线程命令封装
    # ==========================================================
    def _canvas_wh(self) -> tuple:
        cw = self.video_canvas.winfo_width()  or self.canvas_w
        ch = self.video_canvas.winfo_height() or self.canvas_h
        return (max(1, cw), max(1, ch))

    def _pause_segs_snap(self) -> list:
        return [(s['trim_in'], s['trim_out']) for s in self.pause_segments]

    def _speed_segs_snap(self) -> list:
        return [(s['start'], s['end'], s['type']) for s in self.speed_segments]

    def _all_skip_segs_snap(self) -> list:
        """
        合并所有需要预览跳过的区间：
          - pause 段的裁剪区 [trim_in, trim_out)
          - clip  段的两端删除区 [start, keep_in) 和 (keep_out, end]
        IO 线程的 jump_pause 对两种区间的处理完全一样。
        """
        segs = []
        for s in self.pause_segments:
            if s['trim_out'] > s['trim_in']:
                segs.append((s['trim_in'], s['trim_out']))
        for s in self.clip_segments:
            ki, ko = s['keep_in'], s['keep_out']
            if ki > ko:
                # 全删
                segs.append((s['start'], s['end'] + 1))
            else:
                if ki > s['start']:
                    segs.append((s['start'], ki))
                if ko < s['end']:
                    segs.append((ko + 1, s['end'] + 1))
        return segs

    def _seek(self, frame_idx: int, skip_trim: bool = False):
        """发送节流 seek 命令（不影响播放状态）"""
        if not self._io:
            return
        self.current_frame_idx = frame_idx
        self.timeline.current_frame_idx = frame_idx
        self._io.send({
            'type':         CMD_SEEK_LATEST,
            'frame':        frame_idx,
            'canvas_wh':    self._canvas_wh(),
            'pause_segs':   self._all_skip_segs_snap(),
            'skip_trimmed': skip_trim,
        })

    def _send_play(self, start: int):
        if not self._io:
            return
        p = self.settings.get_params()

        # 解析倍速字符串 → step（抽帧）+ multiplier（帧间隔倍数）
        speed_str = self.preview_speed_var.get().rstrip('x')
        try:
            speed = float(speed_str)
        except ValueError:
            speed = 1.0
        if speed >= 1.0:
            preview_step       = max(1, int(speed))
            speed_multiplier   = 1.0
        else:
            preview_step       = 1
            speed_multiplier   = 1.0 / max(speed, 0.01)   # e.g. 0.25x → ×4帧间隔

        self._io.send({
            'type': CMD_PLAY,
            'params': {
                'start_frame':       start,
                'preview_step':      preview_step,
                'speed_multiplier':  speed_multiplier,
                'skip_trimmed':      self.skip_trimmed.get(),
                'speedup_1x':        p['speedup_1x'],
                'speedup_02':        p['speedup_02'],
                'speedup_02_factor': p['speedup_02_factor'],
                'pause_segs':        self._all_skip_segs_snap(),
                'speed_segs':        self._speed_segs_snap(),
                'canvas_wh':         self._canvas_wh(),
            }
        })

    def _send_stop(self):
        if self._io:
            self._io.send({'type': CMD_STOP})

    # ==========================================================
    #  键盘快捷键
    # ==========================================================

    # 连续移动期间的预览刷新间隔（ms）
    _KEY_PREVIEW_MS = 150

    def _bind_keys(self):
        root = self.winfo_toplevel()
        root.bind('<Left>',              self._on_key_press_left,  add='+')
        root.bind('<Right>',             self._on_key_press_right, add='+')
        root.bind('<KeyRelease-Left>',   self._on_key_release,     add='+')
        root.bind('<KeyRelease-Right>',  self._on_key_release,     add='+')
        # 空格统一由我们处理，阻止按钮等控件自己响应
        root.bind('<space>', self._on_key_space)
        for cls in ('TButton', 'Button', 'TCheckbutton', 'TRadiobutton',
                    'TCombobox', 'TNotebook'):
            root.bind_class(cls, '<space>', lambda e: 'break')

    def _on_key_press_left(self, event):
        if self._key_held == 'Left':
            return
        self._key_held       = 'Left'
        self._key_hold_fired = False
        self._step_frame(-1, seek=True)   # 单步：立即 seek 显示画面
        self._key_after_id = self.after(400, self._start_repeat, 'Left')

    def _on_key_press_right(self, event):
        if self._key_held == 'Right':
            return
        self._key_held       = 'Right'
        self._key_hold_fired = False
        self._step_frame(+1, seek=True)
        self._key_after_id = self.after(400, self._start_repeat, 'Right')

    def _on_key_release(self, event):
        direction = event.keysym
        if self._key_held != direction:
            return
        self._key_held = None
        # 取消移动循环
        if self._key_after_id:
            self.after_cancel(self._key_after_id)
            self._key_after_id = None
        # 取消预览循环
        if self._key_preview_id:
            self.after_cancel(self._key_preview_id)
            self._key_preview_id = None
        # 松手时补发一次 seek，确保画面停在最终位置
        if self._key_hold_fired:
            self._do_preview_seek()

    def _on_key_space(self, event):
        # 焦点在输入类控件时放行（用户在打字/输入数字），其余情况空格=播放/暂停
        focused = self.focus_get()
        if isinstance(focused, (ttk.Entry, ttk.Spinbox, tk.Entry, ttk.Combobox)):
            return
        self.toggle_play()
        return 'break'

    def _start_repeat(self, direction: str):
        """长按 400ms 后进入连续移动阶段，同时启动预览刷新定时器"""
        self._key_hold_fired = True
        # 启动预览刷新循环（独立于移动循环，固定 150ms 一次）
        self._schedule_preview()
        self._repeat_frame(direction)

    def _repeat_frame(self, direction: str):
        """
        连续移动循环：只更新 current_frame_idx 和时间轴红条，
        不发 seek（避免每帧解码，IO 线程压力大且画面闪烁）。
        画面刷新由独立的 _preview_tick 定时器负责。
        """
        if self._key_held != direction:
            return
        delta = -1 if direction == 'Left' else +1
        self._step_frame(delta, seek=False)   # 只动指针，不 seek
        speed    = self.settings.key_repeat_speed_var.get()
        interval = max(16, int(1000 / speed))
        self._key_after_id = self.after(interval, self._repeat_frame, direction)

    def _schedule_preview(self):
        """启动预览刷新定时器"""
        self._key_preview_id = self.after(
            self._KEY_PREVIEW_MS, self._preview_tick)

    def _preview_tick(self):
        """定期发一次 seek 显示当前位置的画面"""
        if not self._key_held:
            return
        self._do_preview_seek()
        self._key_preview_id = self.after(
            self._KEY_PREVIEW_MS, self._preview_tick)

    def _do_preview_seek(self):
        """向 IO 线程发送节流 seek，显示当前帧画面"""
        if not self._io or self.total_frames <= 0:
            return
        idx = self.current_frame_idx
        self._io.send({
            'type':         CMD_SEEK_LATEST,
            'frame':        idx,
            'canvas_wh':    self._canvas_wh(),
            'pause_segs':   [],        # 连续移动时不跳裁剪区，显示原始画面
            'skip_trimmed': False,
        })

    def _step_frame(self, delta: int, seek: bool = True):
        """
        移动若干帧。
        seek=True：同时发 CMD_SEEK_LATEST 刷新画面（单步用）
        seek=False：只更新索引和时间轴红条（连续移动用，画面由预览定时器刷新）
        """
        if self.total_frames <= 0:
            return
        new_idx = max(0, min(self.total_frames - 1,
                             self.current_frame_idx + delta))
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

    # ==========================================================
    #  时间轴回调
    # ==========================================================
    def _on_tl_seek(self, frame_idx: int):
        """时间轴点击/拖动红条 → seek（不跳过裁剪区）"""
        self._seek(frame_idx, skip_trim=False)

    def _on_tl_drag_end(self):
        """红条松手：若正在播放则从当前位置重启"""
        if self.is_playing:
            self._send_play(self.current_frame_idx)

    # ==========================================================
    #  渲染循环（主线程 ~60fps）
    # ==========================================================
    def _render_loop(self):
        try:
            idx, rgb = self._frame_q.get_nowait()
            # 连续移动期间：帧的 idx 可能是 150ms 前发 seek 时的旧位置，
            # 不能覆盖已经快速推进的 current_frame_idx（否则红条回弹）。
            # 只在非键盘连续移动时才用帧 idx 更新当前位置。
            if not self._key_hold_fired:
                self.current_frame_idx = idx
                self.timeline.current_frame_idx = idx
                self.timeline._ensure_pointer_visible()
            self._display_rgb(rgb)
        except Empty:
            pass
        self.timeline.update_pointer()
        self.after(16, self._render_loop)

    def _display_rgb(self, rgb: np.ndarray):
        img   = PIL.Image.fromarray(rgb)
        photo = PIL.ImageTk.PhotoImage(image=img)
        cw, ch = self._canvas_wh()
        if self._canvas_img_id is None:
            self.video_canvas.delete("all")
            self._canvas_img_id = self.video_canvas.create_image(
                cw // 2, ch // 2, image=photo)
        else:
            self.video_canvas.coords(self._canvas_img_id, cw // 2, ch // 2)
            self.video_canvas.itemconfig(self._canvas_img_id, image=photo)
        self._photo = photo
        self._update_labels()

    # ==========================================================
    #  标签更新
    # ==========================================================
    def _update_labels(self):
        cur   = self.current_frame_idx
        cur_s = cur / self.fps if self.fps else 0
        tot_s = self.total_frames / self.fps if self.fps else 0
        self.lbl_time.config(text=f"{self._fmt(cur_s)} / {self._fmt(tot_s)}")

        info = "普通区域"
        for seg in self.pause_segments:
            if seg['start'] <= cur <= seg['end']:
                info = (f"暂停 | DCT:{seg.get('dct',0):.0f} | "
                        f"Hist:{seg.get('hist',0):.4f} | "
                        f"{'跳变-保留端点' if seg.get('is_diff') else '无跳变-全删'}")
                break
        else:
            p = self.settings.get_params()
            for seg in self.speed_segments:
                if seg['start'] <= cur <= seg['end']:
                    t    = seg['type']
                    name = {FRAME_TYPE_1X:'1x', FRAME_TYPE_2X:'2x',
                            FRAME_TYPE_0_2X:'0.2x'}.get(t, '?')
                    eff  = 1
                    if t == FRAME_TYPE_1X   and p.get('speedup_1x'):  eff = 2
                    if t == FRAME_TYPE_0_2X and p.get('speedup_02'):  eff = p.get('speedup_02_factor', 10)
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
    #  批量暂停模式（由 settings_panel 三个按钮触发）
    # ==========================================================
    def apply_pause_mode(self, mode: str):
        """
        mode: 'keep' / 'all' / 'auto'
          keep — 全部保留（空裁剪区）
          all  — 全部裁剪
          auto — 按 settings 的 keep_before / keep_after 重算
        """
        if not self.pause_segments:
            return
        p      = self.settings.get_params()
        keep_n = p['compare']['keep_n']
        keep_m = p['compare']['keep_m']

        for seg in self.pause_segments:
            s, e    = seg['start'], seg['end']
            seg_len = e - s + 1
            if mode == 'keep':
                seg['trim_in']  = s
                seg['trim_out'] = s
            elif mode == 'all':
                seg['trim_in']  = s
                seg['trim_out'] = e + 1
            else:
                n = min(keep_n, seg_len)
                m = min(keep_m, seg_len - n)
                ti = s + n
                to = e - m + 1
                if ti > to:
                    ti = to = s
                seg['trim_in']  = ti
                seg['trim_out'] = to

        self.timeline.mark_dirty()
        self.timeline.redraw()

    # ==========================================================
    #  模板分析
    # ==========================================================
    def _start_analysis(self):
        if not self.video_path:
            return
        from tkinter import messagebox
        import analyzer

        self.btn_analyze.config(state=tk.DISABLED, text="分析中...")
        p = self.settings.get_params()

        def worker():
            proc_res = list(p['proc_res'])
            cap_tmp  = cv2.VideoCapture(self.video_path)
            ret, f   = cap_tmp.read()
            cap_tmp.release()
            if ret and proc_res[1] == 225:
                h, ww    = f.shape[:2]
                proc_res[1] = int(proc_res[0] * h / ww)
            proc_res = tuple(proc_res)

            configs, loaded = analyzer.load_templates(proc_res)
            if loaded == 0:
                self.after(0, lambda: messagebox.showwarning(
                    "模板缺失", "未找到可用模板，将标记所有帧为普通帧。"))

            def prog(r):
                self.after(0, lambda: self.btn_analyze.config(
                    text=f"匹配 {int(r*100)}%"))

            states = analyzer.analyze_video(
                self.video_path, configs, p['thresholds'],
                proc_res, p['batch'], p['threads'], prog)

            self.after(0, lambda: self.btn_analyze.config(text="计算段落..."))

            pauses, speeds = analyzer.build_segments(
                states, self.video_path, proc_res, p['compare'])

            self.after(0, lambda: self._finish_analysis(states, pauses, speeds))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_analysis(self, states, pauses, speeds):
        from tkinter import messagebox
        self.states_array   = states
        self.pause_segments = pauses
        self.speed_segments = speeds
        # 自动生成非暂停区间为 clip_segments
        self.clip_segments  = self._build_clip_segments(pauses, self.total_frames)
        # 同步到时间轴
        self.timeline.pause_segments = self.pause_segments
        self.timeline.speed_segments  = self.speed_segments
        self.timeline.clip_segments   = self.clip_segments
        self.timeline.mark_dirty()
        self.btn_analyze.config(state=tk.NORMAL, text="自动模板分析")
        self.timeline.redraw()
        messagebox.showinfo("分析完成",
                            f"识别到 {len(pauses)} 处暂停，{len(speeds)} 个变速区间。")

    @staticmethod
    def _build_clip_segments(pauses: list, total_frames: int) -> list:
        """
        把 [0, total_frames) 里所有不属于任何暂停段的连续区间生成为 clip_segments。
        初始 keep_in = start, keep_out = end（全保留），用户拖手柄来裁两端。
        """
        if total_frames <= 0:
            return []

        # 收集所有暂停段占用的区间，合并后取补集
        occupied = []
        for seg in pauses:
            occupied.append((seg['start'], seg['end']))
        occupied.sort()

        clips = []
        clip_id = 0
        prev_end = -1   # 上一个占用区间的结束帧

        for ps, pe in occupied:
            gap_start = prev_end + 1
            gap_end   = ps - 1
            if gap_end >= gap_start:
                clips.append({
                    'id':       clip_id,
                    'start':    gap_start,
                    'end':      gap_end,
                    'keep_in':  gap_start,
                    'keep_out': gap_end,
                })
                clip_id += 1
            prev_end = pe

        # 最后一个暂停段之后的尾部
        tail_start = prev_end + 1
        tail_end   = total_frames - 1
        if tail_end >= tail_start:
            clips.append({
                'id':       clip_id,
                'start':    tail_start,
                'end':      tail_end,
                'keep_in':  tail_start,
                'keep_out': tail_end,
            })

        return clips

    # ==========================================================
    #  导出
    # ==========================================================
    def export_video(self):
        from tkinter import messagebox
        import analyzer

        if not self.video_path:
            messagebox.showerror("错误", "请先加载视频"); return
        p = self.settings.get_params()
        if not p['output']:
            messagebox.showerror("错误", "请先设置输出路径"); return

        states = self.states_array
        if states is None:
            states = np.zeros(self.total_frames, dtype=np.int8)
        self.settings.export_btn.config(state=tk.DISABLED)

        def worker():
            to_del = analyzer.build_delete_set(
                self.total_frames, states,
                self.pause_segments, self.speed_segments,
                self.clip_segments,
                p['speedup_1x'], p['speedup_02'], p['speedup_02_factor'])

            def prog(ratio, written):
                self.settings.export_progress_var.set(ratio * 100)
                self.settings.export_status_var.set(f"写入 {int(ratio*100)}%")

            try:
                written, total = analyzer.export_video(
                    self.video_path, p['output'], to_del,
                    self.fps, p['quality'], prog)
                self.after(0, lambda: self.settings.export_status_var.set(
                    f"完成！{written}/{total} 帧"))
                self.after(0, lambda: messagebox.showinfo(
                    "导出完成",
                    f"输出：{p['output']}\n总帧：{total}，保留：{written}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("导出失败", str(e)))
            finally:
                self.after(0, lambda: self.settings.export_btn.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()