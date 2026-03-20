# timeline_widget.py — 时间轴画布（绘制 + 鼠标交互）
# 与视频播放完全解耦，通过回调向外通知。

import tkinter as tk
from tkinter import ttk
import numpy as np
import PIL.Image
import PIL.ImageTk

from frame_types import FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X


class TimelineWidget(tk.Frame):
    """
    时间轴画布组件。
    外部通过以下属性/方法与它交互：
      - total_frames, fps, scroll_offset, zoom_level
      - pause_segments, speed_segments
      - current_frame_idx（只读显示用）
      - on_seek_cb(frame_idx)  — 点击/拖动红条触发
      - on_white_drag_cb()     — 白条拖动后触发（用于刷新 play_params 快照）
    """

    TL_HEIGHT = 80   # 时间轴高度（px）

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)

        # 视频元信息（由外部写入）
        self.total_frames: int   = 0
        self.fps:          float = 30.0
        self.zoom_level:   float = 1.0
        self.scroll_offset: float = 0.0
        self.current_frame_idx: int = 0

        # 段落数据（由外部写入）
        self.pause_segments: list = []
        self.speed_segments:  list = []

        # 回调
        self.on_seek_cb        = None   # (frame_idx: int) → None
        self.on_white_drag_cb  = None   # () → None

        # 静态层缓存
        self._tl_static_photo = None
        self._tl_dirty        = True

        # 交互状态
        self.active_handle:       object = None
        self._pending_candidates: list   = []
        self._mousedown_x:        int    = 0
        self._pan_x:              int    = 0

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        self.canvas = tk.Canvas(self, height=self.TL_HEIGHT,
                                bg="#1A1A1A", highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.canvas.bind("<Configure>",        self._on_resize)
        self.canvas.bind("<Button-1>",         self._on_mousedown)
        self.canvas.bind("<B1-Motion>",        self._on_mousemove)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouseup)
        self.canvas.bind("<MouseWheel>",       self._on_scroll)
        self.canvas.bind("<Button-3>",         self._pan_start)
        self.canvas.bind("<B3-Motion>",        self._pan_move)

    # ------------------------------------------------------------------
    # 外部调用接口
    # ------------------------------------------------------------------

    def mark_dirty(self):
        self._tl_dirty        = True
        self._tl_static_photo = None

    def redraw(self):
        """完整重绘（segments 或 zoom/offset 变化后调用）"""
        self.mark_dirty()
        self._rebuild_static()
        self._draw_dynamic()

    def update_pointer(self):
        """只更新红色指针（播放时每帧调用）"""
        self._draw_dynamic()

    # ------------------------------------------------------------------
    # 静态层（segments 色块 + 白条手柄）
    # ------------------------------------------------------------------

    def _rebuild_static(self):
        w = self.canvas.winfo_width()
        if w <= 1:
            w = 600
        h = self.TL_HEIGHT

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

    # ------------------------------------------------------------------
    # 动态层（刻度 + 红色指针）
    # ------------------------------------------------------------------

    def _draw_dynamic(self):
        if self._tl_dirty or self._tl_static_photo is None:
            self._rebuild_static()

        w = self.canvas.winfo_width()
        if w <= 1:
            w = 600
        h = self.TL_HEIGHT

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tl_static_photo)

        self._draw_ticks(w, h)

        if self.total_frames > 0:
            px = self._f2x(self.current_frame_idx, w)
            if 0 <= px <= w:
                self.canvas.create_line(px, 14, px, h, fill="#FF4444", width=2)
                self.canvas.create_polygon(
                    [px - 7, 0, px + 7, 0, px, 14],
                    fill="#FF4444", outline="#CC2222", width=1)

    def _draw_ticks(self, w: int, h: int):
        """PR 风格帧刻度：每帧宽 >= 4px 时显示"""
        if self.total_frames <= 0 or self.fps <= 0:
            return
        px_per_frame = w * self.zoom_level / self.total_frames
        if px_per_frame < 4:
            return

        start_f = max(0, int(self.scroll_offset * self.total_frames) - 1)
        end_f   = min(self.total_frames,
                      int((self.scroll_offset + 1.0 / self.zoom_level)
                          * self.total_frames) + 2)

        col_minor  = "#444444"
        col_major  = "#666666"
        col_second = "#888888"
        col_label  = "#999999"

        fps_int = max(1, int(round(self.fps)))
        if px_per_frame >= 20:
            step = 1
        elif px_per_frame >= 10:
            step = 2
        elif px_per_frame >= 6:
            step = 5
        else:
            step = max(1, fps_int // 6)

        for f in range(start_f, end_f, step):
            px = self._f2x(f, w)
            if px < 0 or px > w:
                continue
            is_second = (f % fps_int == 0)
            is_major  = (f % 5 == 0)
            if is_second:
                self.canvas.create_line(px, 0, px, h - 20, fill=col_second, width=1)
                sec = f / self.fps
                m, s = divmod(int(sec), 60)
                label = f"{m}:{s:02d}" if m > 0 else f"{s}s"
                self.canvas.create_text(px + 2, 3, anchor=tk.NW, text=label,
                                        fill=col_label, font=("Consolas", 8))
            elif is_major:
                self.canvas.create_line(px, h - 30, px, h - 20, fill=col_major, width=1)
            else:
                self.canvas.create_line(px, h - 25, px, h - 20, fill=col_minor, width=1)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

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

    def _ensure_pointer_visible(self):
        if self.total_frames <= 0:
            return
        p  = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        if p < self.scroll_offset or p > self.scroll_offset + vw:
            self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
            self.mark_dirty()

    # ------------------------------------------------------------------
    # 鼠标交互
    # ------------------------------------------------------------------

    def _on_resize(self, event=None):
        self.mark_dirty()
        self._draw_dynamic()

    def _on_mousedown(self, event):
        if self.total_frames <= 0:
            return
        w  = self.canvas.winfo_width()
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

        # 点击空白区域 → seek
        tf = self._x2f(event.x, w)
        if self.on_seek_cb:
            self.on_seek_cb(tf)

    def _on_mousemove(self, event):
        w = self.canvas.winfo_width()

        if self.active_handle and self.active_handle[0] == 'red':
            tf = self._x2f(event.x, w)
            self.current_frame_idx = tf
            if self.on_seek_cb:
                self.on_seek_cb(tf)
            self._draw_dynamic()
            return

        # 白条消歧义
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
            self.mark_dirty()
            self._draw_dynamic()

    def _on_mouseup(self, event):
        was_red = self.active_handle and self.active_handle[0] == 'red'
        self.active_handle        = None
        self._pending_candidates  = []
        self._mousedown_x         = 0
        if was_red and self.on_white_drag_cb:
            # 复用回调：红条松手也通知外部（外部区分是否播放中决定是否重启）
            self.on_white_drag_cb()

    def _on_scroll(self, event):
        if self.total_frames <= 0:
            return
        self.zoom_level *= (1.2 if event.delta > 0 else 1 / 1.2)
        self.zoom_level  = max(1.0, min(self.zoom_level, 500.0))
        p  = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
        self.mark_dirty()
        self._draw_dynamic()

    def _pan_start(self, event):
        self._pan_x = event.x

    def _pan_move(self, event):
        if self.zoom_level <= 1.0:
            return
        dx = event.x - self._pan_x
        self._pan_x = event.x
        move = (dx / self.canvas.winfo_width()) / self.zoom_level
        self.scroll_offset = max(0.0, min(self.scroll_offset - move,
                                          1.0 - 1.0 / self.zoom_level))
        self.mark_dirty()
        self._draw_dynamic()