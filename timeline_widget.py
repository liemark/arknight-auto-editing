# timeline_widget.py — 时间轴画布（绘制 + 鼠标交互）

import tkinter as tk
from tkinter import ttk
import numpy as np
import PIL.Image
import PIL.ImageTk

from frame_types import FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X

# ---------- 布局常量 ----------
_CLIP_HANDLE_Y1 = 32
_CLIP_HANDLE_Y2 = lambda h: h - 20
_PAUSE_BAND_Y1 = 15
_PAUSE_BAND_Y2 = lambda h: h - 25
_CLIP_BAND_Y1 = 37
_CLIP_BAND_Y2 = lambda h: h - 25


class TimelineWidget(tk.Frame):
    TL_HEIGHT = 80

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)

        self.total_frames: int = 0
        self.fps: float = 30.0
        self.zoom_level: float = 1.0
        self.scroll_offset: float = 0.0
        self.current_frame_idx: int = 0

        self.pause_segments: list = []
        self.speed_segments: list = []
        self.clip_segments: list = []

        # 回调
        self.on_seek_cb = None
        self.on_handle_end_cb = None
        self.on_pause_select_cb = None

        self._tl_static_photo = None
        self._tl_dirty = True
        self.selected_pause_id = None

        self.active_handle: object = None
        self._pending_candidates: list = []
        self._mousedown_x: int = 0
        self._pan_x: int = 0

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        self.canvas = tk.Canvas(self, height=self.TL_HEIGHT, bg="#1A1A1A", highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_mousedown)
        self.canvas.bind("<B1-Motion>", self._on_mousemove)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouseup)
        self.canvas.bind("<MouseWheel>", self._on_scroll)

        # 将平移绑定到鼠标中键（滚轮按下）
        self.canvas.bind("<Button-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        # 将鼠标右键赋予“点选切换小片段保留状态”的功能
        self.canvas.bind("<Button-3>", self._on_right_click)

        hint = ("滚轮缩放 | 中键拖动平移 | 左键选中 | 右键切换保留/删除 | 拖动青条调整边缘")
        tk.Label(self, text=hint, fg="#555555", bg="#1A1A1A",
                 font=("Consolas", 8), anchor="w").pack(fill=tk.X, padx=4, pady=(0, 2))

    # ------------------------------------------------------------------
    # 外部接口
    # ------------------------------------------------------------------

    def mark_dirty(self):
        self._tl_dirty = True
        self._tl_static_photo = None

    def redraw(self):
        self.mark_dirty()
        self._rebuild_static()
        self._draw_dynamic()

    def update_pointer(self):
        self._draw_dynamic()

    # ------------------------------------------------------------------
    # 静态层
    # ------------------------------------------------------------------

    def _rebuild_static(self):
        w = self.canvas.winfo_width()
        if w <= 1: w = 600
        h = self.TL_HEIGHT

        img = np.full((h, w, 3), (26, 26, 26), dtype=np.uint8)

        if self.total_frames <= 0:
            pil_img = PIL.Image.fromarray(img, mode='RGB')
            self._tl_static_photo = PIL.ImageTk.PhotoImage(image=pil_img)
            self._tl_dirty = False
            return

        # 1. 变速色条
        speed_colors = {
            FRAME_TYPE_1X: (30, 144, 255),
            FRAME_TYPE_2X: (147, 112, 219),
            FRAME_TYPE_0_2X: (60, 179, 113),
        }
        sy1, sy2 = h - 20, h
        for seg in self.speed_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)
            if xe < 0 or xs > w: continue
            c = speed_colors.get(seg['type'])
            if c:
                x1, x2 = max(0, int(xs)), min(w, int(xe))
                if x2 > x1: img[sy1:sy2, x1:x2] = c

        # 2. 暂停段色块（带细粒度保留区绘制）
        py1, py2 = _PAUSE_BAND_Y1, _PAUSE_BAND_Y2(h)

        # 定义四种状态对应的色块颜色
        col_map = {
            #0: (255, 204, 0),  # 自动保留：原本的亮黄
            0: (100, 200, 50),  # 自动保留：同样是亮绿色
            1: (68, 51, 0),  # 自动删除：原本的暗色
            2: (180, 50, 50),  # 手动删除（作废）：红棕色
            3: (100, 200, 50),  # 手动保留（抢救）：亮绿色
        }
        sel_color = (255, 255, 255)

        def pfill(x1f, x2f, col, _img=img, _y1=py1, _y2=py2):
            xi1 = max(0, int(x1f))
            xi2 = min(w, int(x2f))
            if xi1 == xi2 and x2f > x1f and xi1 < w: xi2 += 1
            if xi2 > xi1: _img[_y1:_y2, xi1:xi2] = col

        for seg in self.pause_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)
            if xe < 0 or xs > w: continue

            mode = seg.get('mode', 'auto')
            if mode == 'all':
                pfill(xs, xe, col_map[1])
            elif mode == 'keep':
                pfill(xs, xe, col_map[0])
            else:
                mask = seg.get('local_del_mask')
                if mask is None:
                    pfill(xs, xe, col_map[0])
                else:
                    cur = mask[0]
                    st = 0
                    for i in range(1, len(mask)):
                        if mask[i] != cur:
                            fxs = self._f2x(seg['start'] + st, w)
                            fxe = self._f2x(seg['start'] + i, w)
                            pfill(fxs, fxe, col_map.get(cur, col_map[0]))
                            cur = mask[i]
                            st = i
                    fxs = self._f2x(seg['start'] + st, w)
                    fxe = self._f2x(seg['start'] + len(mask), w)
                    pfill(fxs, fxe, col_map.get(cur, col_map[0]))

            # 画选中高亮底边指示
            if getattr(self, 'selected_pause_id', None) == seg['id']:
                pfill(xs, xe, sel_color, _y1=py2, _y2=py2 + 3)

        # 3. clip 段色块
        cy1, cy2 = _CLIP_BAND_Y1, _CLIP_BAND_Y2(h)
        chy1, chy2 = _CLIP_HANDLE_Y1, _CLIP_HANDLE_Y2(h)
        clip_keep = (32, 178, 170)
        clip_del = (20, 60, 60)
        cyan_hdl = (0, 230, 200)

        def cfill(x1f, x2f, col, _img=img, _y1=cy1, _y2=cy2):
            xi1 = max(0, int(x1f))
            xi2 = min(w, int(x2f))
            if xi1 == xi2 and x2f > x1f and xi1 < w: xi2 += 1
            if xi2 > xi1: _img[_y1:_y2, xi1:xi2] = col

        for seg in self.clip_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)
            xki = self._f2x(seg['keep_in'], w)
            xko = self._f2x(seg['keep_out'] + 1, w)
            if xe < 0 or xs > w: continue

            cfill(xs, xki, clip_del)
            cfill(xki, xko, clip_keep)
            cfill(xko, xe, clip_del)

            for hx in (xki, xko):
                hxi = int(hx)
                if 0 <= hxi <= w:
                    xi1 = max(0, hxi - 3)
                    xi2 = min(w, hxi + 4)
                    img[chy1:chy2, xi1:xi2] = cyan_hdl

        pil_img = PIL.Image.fromarray(img, mode='RGB')
        self._tl_static_photo = PIL.ImageTk.PhotoImage(image=pil_img)
        self._tl_dirty = False

    # ------------------------------------------------------------------
    # 动态层（刻度 + 红条）
    # ------------------------------------------------------------------

    def _draw_dynamic(self):
        if self._tl_dirty or self._tl_static_photo is None: self._rebuild_static()

        w = self.canvas.winfo_width()
        if w <= 1: w = 600
        h = self.TL_HEIGHT

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tl_static_photo)
        self._draw_ticks(w, h)

        if self.total_frames > 0:
            px = self._f2x(self.current_frame_idx, w)
            if 0 <= px <= w:
                self.canvas.create_line(px, 14, px, h, fill="#FF4444", width=2)
                self.canvas.create_polygon(
                    [px - 7, 0, px + 7, 0, px, 14], fill="#FF4444", outline="#CC2222", width=1)

    def _draw_ticks(self, w: int, h: int):
        if self.total_frames <= 0 or self.fps <= 0: return
        px_per_frame = w * self.zoom_level / self.total_frames
        if px_per_frame < 4: return

        start_f = max(0, int(self.scroll_offset * self.total_frames) - 1)
        end_f = min(self.total_frames, int((self.scroll_offset + 1.0 / self.zoom_level) * self.total_frames) + 2)
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
            if px < 0 or px > w: continue
            is_second = (f % fps_int == 0)
            is_major = (f % 5 == 0)
            if is_second:
                self.canvas.create_line(px, 0, px, h - 20, fill="#888888", width=1)
                sec = f / self.fps
                m, s = divmod(int(sec), 60)
                label = f"{m}:{s:02d}" if m > 0 else f"{s}s"
                self.canvas.create_text(px + 2, 3, anchor=tk.NW, text=label, fill="#999999", font=("Consolas", 8))
            elif is_major:
                self.canvas.create_line(px, h - 30, px, h - 20, fill="#666666", width=1)
            else:
                self.canvas.create_line(px, h - 25, px, h - 20, fill="#444444", width=1)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def _f2x(self, frame_idx: int, canvas_w: int) -> float:
        if self.total_frames <= 0: return 0.0
        return (frame_idx / self.total_frames - self.scroll_offset) * self.zoom_level * canvas_w

    def _x2f(self, x: float, canvas_w: int) -> int:
        if canvas_w <= 0 or self.total_frames <= 0: return 0
        ratio = self.scroll_offset + (x / canvas_w) / self.zoom_level
        return int(max(0.0, min(ratio, 1.0)) * self.total_frames)

    def _ensure_pointer_visible(self):
        if self.total_frames <= 0: return
        p = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        if p < self.scroll_offset or p > self.scroll_offset + vw:
            self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
            self.mark_dirty()

    # ------------------------------------------------------------------
    # 命中检测辅助
    # ------------------------------------------------------------------

    def _collect_candidates(self, event_x: int, canvas_w: int) -> list:
        cands = []
        RADIUS = 12
        for seg in self.clip_segments:
            ix = self._f2x(seg['keep_in'], canvas_w)
            ox = self._f2x(seg['keep_out'] + 1, canvas_w)
            if abs(event_x - ix) < RADIUS:
                cands.append((abs(event_x - ix), 'clip_in', seg['id']))
            if abs(event_x - ox) < RADIUS:
                cands.append((abs(event_x - ox), 'clip_out', seg['id']))
        return cands

    # ------------------------------------------------------------------
    # 鼠标交互
    # ------------------------------------------------------------------

    def _on_resize(self, event=None):
        self.mark_dirty()
        self._draw_dynamic()

    def _on_mousedown(self, event):
        self.canvas.focus_set()
        if self.total_frames <= 0: return
        w = self.canvas.winfo_width()
        ey = event.y

        cands = self._collect_candidates(event.x, w)
        px = self._f2x(self.current_frame_idx, w)
        red_hit = abs(event.x - px) < 8

        # 选中片段检测
        py1, py2 = _PAUSE_BAND_Y1, _PAUSE_BAND_Y2(self.TL_HEIGHT)
        if py1 <= ey <= py2 + 3:
            tf = self._x2f(event.x, w)
            clicked_seg_id = None
            for seg in self.pause_segments:
                if seg['start'] <= tf <= seg['end']:
                    clicked_seg_id = seg['id']
                    break
            if clicked_seg_id is not None:
                self.selected_pause_id = clicked_seg_id
                self.mark_dirty()
                self._draw_dynamic()
                if self.on_pause_select_cb:
                    self.on_pause_select_cb(clicked_seg_id)

        if ey < 14 and red_hit:
            self.active_handle = ('red',)
            return
        if cands:
            self._pending_candidates = cands
            self._mousedown_x = event.x
            self.active_handle = None
            return
        if red_hit:
            self.active_handle = ('red',)
            return

        tf = self._x2f(event.x, w)
        if self.on_seek_cb: self.on_seek_cb(tf)

    def _on_right_click(self, event):
        """右键切换智能剪辑小片段状态：废弃保留区/复原删除区"""
        if self.total_frames <= 0: return
        w = self.canvas.winfo_width()
        ey = event.y

        py1, py2 = _PAUSE_BAND_Y1, _PAUSE_BAND_Y2(self.TL_HEIGHT)
        if py1 <= ey <= py2 + 3:
            tf = self._x2f(event.x, w)
            for seg in self.pause_segments:
                if seg['start'] <= tf <= seg['end']:
                    if seg.get('mode', 'auto') == 'auto' and 'local_del_mask' in seg:
                        mask = seg['local_del_mask']
                        local_idx = tf - seg['start']
                        if 0 <= local_idx < len(mask):
                            curr_val = mask[local_idx]
                            # 如果是 Keep (0) 或 Manual Keep (3) -> 切换为 Manual Del (2)
                            # 如果是 Del (1) 或 Manual Del (2) -> 切换为 Manual Keep (3)
                            target_val = 2 if curr_val in (0, 3) else 3

                            # 寻找当前状态这连续的一整块区域，一体变色
                            s_i = local_idx
                            while s_i > 0 and mask[s_i - 1] == curr_val:
                                s_i -= 1
                            e_i = local_idx
                            while e_i < len(mask) - 1 and mask[e_i + 1] == curr_val:
                                e_i += 1

                            mask[s_i:e_i + 1] = target_val
                            self.mark_dirty()
                            self._draw_dynamic()
                    break

    def _on_mousemove(self, event):
        w = self.canvas.winfo_width()

        if self.active_handle and self.active_handle[0] == 'red':
            tf = self._x2f(event.x, w)
            self.current_frame_idx = tf
            if self.on_seek_cb: self.on_seek_cb(tf)
            self._draw_dynamic()
            return

        if self._pending_candidates and self.active_handle is None:
            dx = event.x - self._mousedown_x
            self.active_handle = self._resolve_handle(dx)
            self._pending_candidates = []

        if not self.active_handle: return

        kind = self.active_handle[0]
        tf = self._x2f(event.x, w)

        if kind == 'clip_in':
            self._move_clip_handle(self.active_handle[1], 'in', tf)
        elif kind == 'clip_out':
            self._move_clip_handle(self.active_handle[1], 'out', tf)

        self.mark_dirty()
        self._draw_dynamic()

    def _resolve_handle(self, dx: int) -> tuple:
        if len(self._pending_candidates) == 1: return (self._pending_candidates[0][1], self._pending_candidates[0][2])
        if dx > 0:
            prefer = {'clip_in'}
        elif dx < 0:
            prefer = {'clip_out'}
        else:
            prefer = set()
        for _, htype, hid in sorted(self._pending_candidates):
            if htype in prefer: return (htype, hid)
        return (self._pending_candidates[0][1], self._pending_candidates[0][2])

    def _move_clip_handle(self, seg_id: int, side: str, tf: int):
        for seg in self.clip_segments:
            if seg['id'] != seg_id: continue
            s0, s1 = seg['start'], seg['end']
            if side == 'in':
                seg['keep_in'] = max(s0, min(tf, s1 + 1))
                if seg['keep_in'] > seg['keep_out']:
                    seg['keep_out'] = min(seg['keep_in'] - 1, s1)
            else:
                seg['keep_out'] = min(s1, max(tf, s0 - 1))
                if seg['keep_out'] < seg['keep_in']:
                    seg['keep_in'] = max(seg['keep_out'] + 1, s0)
            break

    def _on_mouseup(self, event):
        self.active_handle = None
        self._pending_candidates = []
        self._mousedown_x = 0
        if self.on_handle_end_cb: self.on_handle_end_cb()

    def _on_scroll(self, event):
        if self.total_frames <= 0: return
        self.zoom_level *= (1.2 if event.delta > 0 else 1 / 1.2)

        w = self.canvas.winfo_width()
        if w <= 1: w = 600
        max_zoom = max(500.0, 30.0 * self.total_frames / w)

        self.zoom_level = max(1.0, min(self.zoom_level, max_zoom))

        p = self.current_frame_idx / self.total_frames
        vw = 1.0 / self.zoom_level
        self.scroll_offset = max(0.0, min(p - vw / 2, 1.0 - vw))
        self.mark_dirty()
        self._draw_dynamic()

    def _pan_start(self, event):
        self._pan_x = event.x

    def _pan_move(self, event):
        if self.zoom_level <= 1.0: return
        dx = event.x - self._pan_x
        self._pan_x = event.x
        move = (dx / self.canvas.winfo_width()) / self.zoom_level
        self.scroll_offset = max(0.0, min(self.scroll_offset - move, 1.0 - 1.0 / self.zoom_level))
        self.mark_dirty()
        self._draw_dynamic()