# preview_player.py — 视频预览播放器 v5
#
# 架构：单一 VideoIO 线程持有唯一 cap，通过命令队列接收指令
#   主线程  ──cmd_q──▶  VideoIO线程（唯一cap）──frame_q──▶  渲染循环（主线程）
#
# 解决：
#   1. FFmpeg async_lock 崩溃：彻底消除多线程并发访问同一/多个cap
#   2. 暂停帧露出：IO线程在seek后、read前先过滤裁剪区，帧绝不进队列
#   3. 倍速：1x/2x/4x 三档，用grab跳帧，帧率固定=视频fps
#   4. 卡顿：grab不解码，暂停区跳帧零解码开销

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import threading
import time
from queue import Queue, Empty

from frame_types import (FRAME_TYPE_NORMAL, FRAME_TYPE_PAUSE,
                          FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X)

# VideoIO 命令类型
_CMD_SEEK    = 'seek'    # {'type': seek,  'frame': int}
_CMD_PLAY    = 'play'    # {'type': play,  'params': dict}
_CMD_STOP    = 'stop'    # {'type': stop}
_CMD_QUIT    = 'quit'    # {'type': quit}  — 关闭线程


class _VideoIOThread(threading.Thread):
    """
    唯一持有 cv2.VideoCapture 的线程。
    接收命令（seek / play / stop / quit），把解码好的帧放入 frame_q。
    """
    def __init__(self, path: str, frame_q: Queue):
        super().__init__(daemon=True)
        self.path    = path
        self.frame_q = frame_q
        self.cmd_q   = Queue()          # 命令入口（其他线程写，本线程读）

        self._cap    = cv2.VideoCapture(path)
        self.fps     = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur    = 0                # cap 当前位置（紧跟 cap 实际状态）

        # 播放状态（仅本线程访问）
        self._playing      = False
        self._play_params  = {}

    # ------------------------------------------------------------------
    def _seek_cap(self, frame_idx: int):
        """把 cap 定位到 frame_idx，更新 _cur"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._cur = frame_idx

    # 小跨度用 grab（连续解封装，帧率平滑），大跨度直接 seek（O(1)，避免阻塞）
    _GRAB_SEEK_THRESHOLD = 30

    def _grab_n(self, n: int):
        """跳过 n 帧：小跨度 grab，大跨度 seek"""
        target = min(self._cur + n, self.total - 1)
        if n > self._GRAB_SEEK_THRESHOLD:
            self._seek_cap(target)   # 单IO线程，此处 seek 完全安全
        else:
            for _ in range(n):
                if not self._cap.grab():
                    break
                self._cur += 1

    def _read_one(self):
        """读一帧并返回 (True, frame) 或 (False, None)"""
        ret, frame = self._cap.read()
        if ret:
            self._cur += 1
        return ret, frame

    # ------------------------------------------------------------------
    @staticmethod
    def _jump_pause(cur: int, pause_segs: list) -> int:
        """若 cur 在裁剪区内，返回裁剪区末尾；否则返回 cur"""
        for ti, to in pause_segs:
            if to > ti and ti <= cur < to:
                return to
        return cur

    @staticmethod
    def _speed_step(cur: int, speed_segs: list,
                    speedup_1x: bool, speedup_02: bool, factor_02: int) -> int:
        for s, e, t in speed_segs:
            if s <= cur <= e:
                if t == FRAME_TYPE_1X and speedup_1x:
                    return 2
                if t == FRAME_TYPE_0_2X and speedup_02:
                    return max(2, factor_02)
        return 1

    # ------------------------------------------------------------------
    def _flush_frame_q(self):
        while True:
            try:
                self.frame_q.get_nowait()
            except Empty:
                break

    def _push_frame(self, cur_idx: int, frame: np.ndarray, canvas_wh: tuple):
        """缩放并推入帧队列（满则丢旧）"""
        cw, ch = canvas_wh
        fh, fw = frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw     = max(1, int(fw * scale))
        nh     = max(1, int(fh * scale))
        small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
            except Empty:
                pass
        self.frame_q.put((cur_idx, rgb))

    # ------------------------------------------------------------------
    def run(self):
        frame_dur = 1.0 / self.fps

        while True:
            # --- 非播放状态：阻塞等命令 ---
            if not self._playing:
                cmd = self.cmd_q.get()
                self._handle_cmd(cmd)
                continue

            # --- 播放状态：尽量非阻塞取命令 ---
            try:
                cmd = self.cmd_q.get_nowait()
                self._handle_cmd(cmd)
                continue
            except Empty:
                pass

            if not self._playing:
                continue

            t0 = time.monotonic()
            pp = self._play_params

            pause_segs = pp['pause_segs']
            skip_trim  = pp['skip_trimmed']
            preview_step = pp['preview_step']
            canvas_wh    = pp['canvas_wh']

            # 1. 跳过暂停裁剪区（小跨度grab，大跨度seek，均O(1)~O(30)）
            if skip_trim:
                jumped = self._jump_pause(self._cur, pause_segs)
                if jumped != self._cur:
                    self._grab_n(jumped - self._cur)
                    self._flush_frame_q()
                    # 跳帧后立即检查命令队列，保证 stop/seek 响应及时
                    try:
                        cmd = self.cmd_q.get_nowait()
                        self._handle_cmd(cmd)
                        continue
                    except Empty:
                        pass

            if self._cur >= self.total:
                self._playing = False
                continue

            # 2. 变速步幅
            speed_step = self._speed_step(
                self._cur, pp['speed_segs'],
                pp['speedup_1x'], pp['speedup_02'], pp['speedup_02_factor'])
            total_step = preview_step * speed_step

            # 3. 读当前帧
            ret, frame = self._read_one()
            if not ret:
                self._playing = False
                continue

            read_pos = self._cur  # read后_cur已+1

            # 4. 跳过后续 (total_step-1) 帧（_grab_n 内部自动选 grab/seek）
            skip = total_step - 1
            if skip > 0:
                if skip_trim:
                    # 先检查落点是否在暂停区
                    landed = self._jump_pause(self._cur + skip, pause_segs)
                    if landed != self._cur + skip:
                        # 落点在暂停区，直接跳到暂停区末尾
                        self._grab_n(landed - self._cur)
                        self._flush_frame_q()
                    else:
                        self._grab_n(skip)
                else:
                    self._grab_n(skip)

            # 5. 推帧
            self._push_frame(read_pos, frame, canvas_wh)

            # 6. 帧率控制
            elapsed = time.monotonic() - t0
            rem = frame_dur - elapsed
            if rem > 0.0005:
                time.sleep(rem)

        # 退出
        self._cap.release()

    # ------------------------------------------------------------------
    def _handle_cmd(self, cmd: dict):
        t = cmd['type']

        if t == _CMD_QUIT:
            self._playing = False
            self._cap.release()
            raise SystemExit   # 退出 run()

        elif t == _CMD_STOP:
            self._playing = False
            self._flush_frame_q()

        elif t == _CMD_SEEK:
            self._playing = False
            self._flush_frame_q()
            frame_idx = cmd['frame']
            canvas_wh = cmd['canvas_wh']
            pause_segs = cmd.get('pause_segs', [])
            skip_trim  = cmd.get('skip_trimmed', True)

            # 跳过裁剪区
            if skip_trim:
                jumped = self._jump_pause(frame_idx, pause_segs)
                frame_idx = jumped

            self._seek_cap(frame_idx)
            ret, frame = self._read_one()
            if ret:
                self._push_frame(frame_idx, frame, canvas_wh)

        elif t == _CMD_PLAY:
            self._flush_frame_q()
            self._play_params = cmd['params']
            # seek 到当前位置（保证 cap 对齐）
            start = cmd['params']['start_frame']
            self._seek_cap(start)
            self._playing = True

    # ------------------------------------------------------------------
    def send(self, cmd: dict):
        """外部调用：发送命令（线程安全）"""
        self.cmd_q.put(cmd)

    def stop_and_join_io(self):
        """关闭IO线程（程序退出时调用）"""
        try:
            self.cmd_q.put({'type': _CMD_QUIT})
        except Exception:
            pass


# ======================================================================
#  播放器 Widget
# ======================================================================
class VideoPreviewPlayer(tk.Frame):
    def __init__(self, parent, settings, video_path=None, width=800, height=450):
        super().__init__(parent)
        self.settings  = settings
        self.video_path = video_path

        self.total_frames = 0
        self.fps          = 30.0
        self.current_frame_idx = 0

        self.canvas_w = width
        self.canvas_h = height

        self.zoom_level    = 1.0
        self.scroll_offset = 0.0

        self.pause_segments: list = []
        self.speed_segments:  list = []
        self.states_array         = None

        self.is_playing = False

        # 唯一IO线程（每次 load_video 重建）
        self._io: _VideoIOThread | None = None
        self._frame_q: Queue = Queue(maxsize=2)

        # 时间轴静态缓存
        self._tl_static_photo = None
        self._tl_dirty        = True

        # 鼠标交互
        self.active_handle       = None
        self._pending_candidates = []
        self._mousedown_x        = 0

        self._setup_ui()
        if video_path:
            self.load_video(video_path)

    # ==========================================================
    #  UI
    # ==========================================================
    def _setup_ui(self):
        self.video_canvas = tk.Canvas(self, width=self.canvas_w,
                                      height=self.canvas_h, bg="black")
        self.video_canvas.pack(pady=5, fill=tk.BOTH, expand=True)

        tl_frame = tk.Frame(self)
        tl_frame.pack(fill=tk.X, padx=10)
        self.timeline_h = 80
        self.timeline_canvas = tk.Canvas(tl_frame, height=self.timeline_h,
                                         bg="#1A1A1A", highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X)
        self.timeline_canvas.bind("<Configure>",       self._on_tl_resize)
        self.timeline_canvas.bind("<Button-1>",        self._on_mousedown)
        self.timeline_canvas.bind("<B1-Motion>",       self._on_mousemove)
        self.timeline_canvas.bind("<ButtonRelease-1>", self._on_mouseup)
        self.timeline_canvas.bind("<MouseWheel>",      self._on_scroll)
        self.timeline_canvas.bind("<Button-3>",        self._pan_start)
        self.timeline_canvas.bind("<B3-Motion>",       self._pan_move)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, pady=5)

        self.btn_play = ttk.Button(ctrl, text="▶ 播放", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="倍速:").pack(side=tk.LEFT, padx=(10, 2))
        self.preview_step_var = tk.IntVar(value=1)
        for label, step in [("1x", 1), ("2x", 2), ("4x", 4)]:
            ttk.Radiobutton(ctrl, text=label,
                            variable=self.preview_step_var,
                            value=step).pack(side=tk.LEFT, padx=2)

        self.btn_analyze = ttk.Button(ctrl, text="自动模板分析",
                                      command=self._start_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        self.skip_trimmed = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="跳过裁剪区",
                        variable=self.skip_trimmed).pack(side=tk.LEFT, padx=5)

        self.lbl_time = ttk.Label(ctrl, text="00:00 / 00:00")
        self.lbl_time.pack(side=tk.RIGHT, padx=10)

        self.lbl_info = ttk.Label(self, text="就绪", foreground="#00CED1",
                                  font=("Consolas", 10))
        self.lbl_info.pack(fill=tk.X, padx=10, pady=2)

        self._render_loop()

    # ==========================================================
    #  视频加载
    # ==========================================================
    def load_video(self, path: str):
        # 先停旧 IO 线程
        if self._io and self._io.is_alive():
            self._io.stop_and_join_io()
            self._io = None

        # 清空帧队列
        while True:
            try:
                self._frame_q.get_nowait()
            except Empty:
                break

        self.video_path = path
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.zoom_level    = 1.0
        self.scroll_offset = 0.0
        self.pause_segments.clear()
        self.speed_segments.clear()
        self.states_array = None
        self._tl_dirty    = True
        self.is_playing   = False
        self.btn_play.config(text="▶ 播放")
        self._canvas_img_id = None   # 新视频强制重建 canvas item

        # 启动新 IO 线程
        self._io = _VideoIOThread(path, self._frame_q)
        self._io.start()
        self.fps          = self._io.fps
        self.total_frames = self._io.total

        if not self.settings.output_var.get():
            import os
            name, _ = os.path.splitext(path)
            self.settings.output_var.set(f"{name}_clipped.mp4")

        # 显示第一帧
        self._send_seek(0)
        self.draw_timeline()

    # ==========================================================
    #  公共命令发送
    # ==========================================================
    def _canvas_wh(self) -> tuple:
        cw = self.video_canvas.winfo_width()  or self.canvas_w
        ch = self.video_canvas.winfo_height() or self.canvas_h
        return (max(1, cw), max(1, ch))

    def _pause_segs_snapshot(self) -> list:
        return [(s['trim_in'], s['trim_out']) for s in self.pause_segments]

    def _speed_segs_snapshot(self) -> list:
        return [(s['start'], s['end'], s['type']) for s in self.speed_segments]

    def _send_seek(self, frame_idx: int):
        """发 seek 命令（同时应用裁剪区跳过）"""
        if not self._io:
            return
        self.is_playing = False
        self.btn_play.config(text="▶ 播放")
        self._io.send({
            'type':        _CMD_SEEK,
            'frame':       frame_idx,
            'canvas_wh':   self._canvas_wh(),
            'pause_segs':  self._pause_segs_snapshot(),
            'skip_trimmed': self.skip_trimmed.get(),
        })

    def _send_play(self, start_frame: int):
        if not self._io:
            return
        p = self.settings.get_params()
        self._io.send({
            'type': _CMD_PLAY,
            'params': {
                'start_frame':       start_frame,
                'preview_step':      self.preview_step_var.get(),
                'skip_trimmed':      self.skip_trimmed.get(),
                'speedup_1x':        p['speedup_1x'],
                'speedup_02':        p['speedup_02'],
                'speedup_02_factor': p['speedup_02_factor'],
                'pause_segs':        self._pause_segs_snapshot(),
                'speed_segs':        self._speed_segs_snapshot(),
                'canvas_wh':         self._canvas_wh(),
            }
        })

    def _send_stop(self):
        if self._io:
            self._io.send({'type': _CMD_STOP})

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
    #  渲染循环（主线程 ~60fps）
    # ==========================================================
    def _render_loop(self):
        try:
            idx, rgb = self._frame_q.get_nowait()
            self.current_frame_idx = idx
            self._display_rgb(rgb)
            self._ensure_pointer_visible()
        except Empty:
            pass
        self._draw_pointer_only()
        self.after(16, self._render_loop)

    def _display_rgb(self, rgb: np.ndarray):
        img   = PIL.Image.fromarray(rgb)
        photo = PIL.ImageTk.PhotoImage(image=img)
        cw, ch = self._canvas_wh()
        # 复用 canvas image item：只 itemconfig，不 delete+create
        if not hasattr(self, '_canvas_img_id') or self._canvas_img_id is None:
            self.video_canvas.delete("all")
            self._canvas_img_id = self.video_canvas.create_image(
                cw // 2, ch // 2, image=photo)
        else:
            self.video_canvas.coords(self._canvas_img_id, cw // 2, ch // 2)
            self.video_canvas.itemconfig(self._canvas_img_id, image=photo)
        self._photo = photo   # 防止 GC
        self._update_labels()

    # ==========================================================
    #  时间轴
    # ==========================================================
    def _mark_timeline_dirty(self):
        self._tl_dirty        = True
        self._tl_static_photo = None

    def draw_timeline(self):
        self._mark_timeline_dirty()
        self._rebuild_static_layer()
        self._draw_pointer_only()

    def _on_tl_resize(self, event=None):
        self._mark_timeline_dirty()
        self._draw_pointer_only()

    def _rebuild_static_layer(self):
        w = self.timeline_canvas.winfo_width()
        if w <= 1:
            w = 600
        h = self.timeline_h

        img = np.full((h, w, 3), (26, 26, 26), dtype=np.uint8)

        if self.total_frames > 0:
            speed_colors = {
                FRAME_TYPE_1X:   (30,  144, 255),
                FRAME_TYPE_2X:   (147, 112, 219),
                FRAME_TYPE_0_2X: (60,  179, 113),
            }
            for seg in self.speed_segments:
                xs = self._f2x(seg['start'], w)
                xe = self._f2x(seg['end'],   w)
                if xe < 0 or xs > w:
                    continue
                c = speed_colors.get(seg['type'])
                if c:
                    x1, x2 = max(0, int(xs)), min(w, int(xe))
                    if x2 > x1:
                        img[h - 20:h, x1:x2] = c

            yellow = (255, 204,   0)
            dark   = ( 68,  51,   0)
            white  = (255, 255, 255)

            for seg in self.pause_segments:
                xs  = self._f2x(seg['start'],    w)
                xe  = self._f2x(seg['end'],      w)
                xti = self._f2x(seg['trim_in'],  w)
                xto = self._f2x(seg['trim_out'], w)
                if xe < 0 or xs > w:
                    continue

                def fill(x1f, x2f, col, _img=img, _h=h):
                    x1i = max(0, int(x1f))
                    x2i = min(_img.shape[1], int(x2f))
                    if x2i > x1i:
                        _img[15:_h - 25, x1i:x2i] = col

                fill(xs,  xti, yellow)
                fill(xto, xe,  yellow)
                fill(xti, xto, dark)

                for hx in (xti, xto):
                    hxi = int(hx)
                    if 0 <= hxi <= w:
                        x1i = max(0, hxi - 3)
                        x2i = min(w, hxi + 4)
                        img[10:h - 20, x1i:x2i] = white

        pil_img = PIL.Image.fromarray(img, mode='RGB')
        self._tl_static_photo = PIL.ImageTk.PhotoImage(image=pil_img)
        self._tl_dirty = False

    def _draw_pointer_only(self):
        if self._tl_dirty or self._tl_static_photo is None:
            self._rebuild_static_layer()

        w = self.timeline_canvas.winfo_width()
        if w <= 1:
            w = 600
        h = self.timeline_h

        self.timeline_canvas.delete("all")
        self.timeline_canvas.create_image(0, 0, anchor=tk.NW,
                                           image=self._tl_static_photo)
        if self.total_frames > 0:
            px = self._f2x(self.current_frame_idx, w)
            if 0 <= px <= w:
                self.timeline_canvas.create_line(
                    px, 14, px, h, fill="#FF4444", width=2)
                self.timeline_canvas.create_polygon(
                    [px - 7, 0, px + 7, 0, px, 14],
                    fill="#FF4444", outline="#CC2222", width=1)

    # ==========================================================
    #  坐标转换
    # ==========================================================
    def _f2x(self, frame_idx: int, canvas_w: int) -> float:
        if self.total_frames <= 0:
            return 0.0
        return (frame_idx / self.total_frames
                - self.scroll_offset) * self.zoom_level * canvas_w

    def _x2f(self, x: float, canvas_w: int) -> int:
        if canvas_w <= 0 or self.total_frames <= 0:
            return 0
        ratio = self.scroll_offset + (x / canvas_w) / self.zoom_level
        return int(max(0.0, min(ratio, 1.0)) * self.total_frames)

    # ==========================================================
    #  时间轴鼠标交互
    # ==========================================================
    def _on_mousedown(self, event):
        if self.total_frames <= 0:
            return
        w  = self.timeline_canvas.winfo_width()
        ey = event.y

        white_cands = []
        for seg in self.pause_segments:
            ix = self._f2x(seg['trim_in'],  w)
            ox = self._f2x(seg['trim_out'], w)
            if abs(event.x - ix) < 10:
                white_cands.append((abs(event.x - ix), seg['id'], 'in'))
            if abs(event.x - ox) < 10:
                white_cands.append((abs(event.x - ox), seg['id'], 'out'))

        px      = self._f2x(self.current_frame_idx, w)
        red_hit = abs(event.x - px) < 8

        if ey < 14:
            if red_hit:
                self.active_handle = ('red',); return
            if white_cands:
                self._pending_candidates = white_cands
                self._mousedown_x = event.x
                self.active_handle = None; return
        else:
            if white_cands:
                self._pending_candidates = white_cands
                self._mousedown_x = event.x
                self.active_handle = None; return
            if red_hit:
                self.active_handle = ('red',); return

        self._do_seek_click(event)

    def _on_mousemove(self, event):
        w = self.timeline_canvas.winfo_width()

        if self.active_handle and self.active_handle[0] == 'red':
            tf = self._x2f(event.x, w)
            # 跳过裁剪区
            if self.skip_trimmed.get():
                for seg in self.pause_segments:
                    if seg['trim_in'] <= tf < seg['trim_out']:
                        tf = seg['trim_out']; break
            self.current_frame_idx = tf
            # 发 seek（IO线程处理，不会崩溃）
            self._send_seek(tf)
            self._draw_pointer_only()
            return

        if self._pending_candidates and self.active_handle is None:
            dx = event.x - self._mousedown_x
            if abs(dx) >= 2:
                if len(self._pending_candidates) == 1:
                    _, sid, ht = self._pending_candidates[0]
                    self.active_handle = ('white', sid, ht)
                else:
                    prefer = 'out' if dx > 0 else 'in'
                    chosen = None
                    for _, sid, ht in sorted(self._pending_candidates):
                        if ht == prefer:
                            chosen = ('white', sid, ht); break
                    self.active_handle = chosen or (
                        'white',
                        self._pending_candidates[0][1],
                        self._pending_candidates[0][2])
                self._pending_candidates = []

        if not self.active_handle:
            return

        if self.active_handle[0] == 'white':
            _, seg_id, h_type = self.active_handle
            tf = self._x2f(event.x, w)
            for seg in self.pause_segments:
                if seg['id'] != seg_id:
                    continue
                s0, s1 = seg['start'], seg['end']
                if h_type == 'in':
                    seg['trim_in'] = max(s0, min(tf, s1 + 1))
                    if seg['trim_in'] > seg['trim_out']:
                        seg['trim_out'] = min(seg['trim_in'], s1 + 1)
                else:
                    seg['trim_out'] = min(s1 + 1, max(tf, s0))
                    if seg['trim_out'] < seg['trim_in']:
                        seg['trim_in'] = max(seg['trim_out'], s0)
                break
            self._mark_timeline_dirty()
            self._draw_pointer_only()

    def _on_mouseup(self, event):
        self.active_handle       = None
        self._pending_candidates = []
        self._mousedown_x        = 0

    def _on_scroll(self, event):
        if self.total_frames <= 0:
            return
        self.zoom_level *= (1.2 if event.delta > 0 else 1 / 1.2)
        self.zoom_level  = max(1.0, min(self.zoom_level, 500.0))
        p  = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
        self._mark_timeline_dirty()
        self._draw_pointer_only()

    def _pan_start(self, event):
        self._pan_x = event.x

    def _pan_move(self, event):
        if self.zoom_level <= 1.0:
            return
        dx = event.x - self._pan_x
        self._pan_x = event.x
        move = (dx / self.timeline_canvas.winfo_width()) / self.zoom_level
        self.scroll_offset = max(0.0, min(
            self.scroll_offset - move, 1.0 - 1.0 / self.zoom_level))
        self._mark_timeline_dirty()
        self._draw_pointer_only()

    def _do_seek_click(self, event):
        w  = self.timeline_canvas.winfo_width()
        tf = self._x2f(event.x, w)
        self._send_seek(tf)

    # ==========================================================
    #  指针可见
    # ==========================================================
    def _ensure_pointer_visible(self):
        if self.total_frames <= 0:
            return
        p  = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        if p < self.scroll_offset or p > self.scroll_offset + vw:
            self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
            self._mark_timeline_dirty()

    # ==========================================================
    #  状态栏
    # ==========================================================
    def _update_labels(self):
        cur   = self.current_frame_idx
        cur_s = cur / self.fps
        tot_s = self.total_frames / self.fps
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
                    if t == FRAME_TYPE_1X   and p.get('speedup_1x'):   eff = 2
                    if t == FRAME_TYPE_0_2X and p.get('speedup_02'):   eff = p.get('speedup_02_factor', 10)
                    pstep = self.preview_step_var.get()
                    total_eff = eff * pstep
                    info = f"变速 {name}" + (f"（预览抽 {total_eff}x）" if total_eff > 1 else "")
                    break
        self.lbl_info.config(text=info)

    @staticmethod
    def _fmt(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    # ==========================================================
    #  分析
    # ==========================================================
    def _start_analysis(self):
        if not self.video_path:
            return
        from tkinter import messagebox
        import analyzer

        self.btn_analyze.config(state=tk.DISABLED, text="分析中...")
        p = self.settings.get_params()

        def worker():
            # 先算实际 proc_res（按视频比例修正高度）
            proc_res = list(p['proc_res'])
            cap_tmp = cv2.VideoCapture(self.video_path)
            ret, f  = cap_tmp.read()
            cap_tmp.release()
            if ret and proc_res[1] == 225:
                h, ww = f.shape[:2]
                proc_res[1] = int(proc_res[0] * h / ww)
            proc_res = tuple(proc_res)

            # 传入 proc_res，load_templates 预缩放模板，分析时跳过 resize
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
        self.states_array   = states
        self.pause_segments = pauses
        self.speed_segments = speeds
        self._mark_timeline_dirty()
        self.btn_analyze.config(state=tk.NORMAL, text="自动模板分析")
        self.draw_timeline()
        from tkinter import messagebox
        messagebox.showinfo("分析完成",
                            f"识别到 {len(pauses)} 处暂停，{len(speeds)} 个变速区间。")

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