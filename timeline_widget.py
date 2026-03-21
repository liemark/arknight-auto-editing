# timeline_widget.py — 时间轴画布（绘制 + 鼠标交互）
#
# 两类可编辑区段：
#   pause_segments  — 暂停区：保留两端，删中间 [trim_in, trim_out)
#   clip_segments   — 非暂停区：删两端，保留中间 [keep_in, keep_out)
#
# 手柄视觉：
#   橙黄条（暂停）高度 ≈ 50px，与暂停区黄色同色系
#   青色条（clip） 高度 ≈ 28px，明显更短，在重叠区域也可区分
#
# 重叠消歧义：向右拖优先 pause_out/clip_in；向左拖优先 pause_in/clip_out

import tkinter as tk
from tkinter import ttk
import numpy as np
import PIL.Image
import PIL.ImageTk

from frame_types import FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X

# ---------- 布局常量 ----------
# 时间轴纵向布局（h=80px）：
#   [0,  13]   红色指针三角区
#   [14, h-26] 主内容区
#   [h-20, h]  变速色条
#
# 橙黄条（暂停）手柄：y 10 ~ h-20  高度 ≈ 50px
# 青条（clip）  手柄：y 32 ~ h-20  高度 ≈ 28px  → 明显比橙黄条短
_PAUSE_HANDLE_Y1 = 10
_PAUSE_HANDLE_Y2 = lambda h: h - 20
_CLIP_HANDLE_Y1 = 32  # 缩短：从22改为32
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
        self.clip_segments: list = []  # 新增：手动裁剪段

        # 回调
        self.on_seek_cb = None  # (frame_idx) → None
        self.on_handle_end_cb = None  # () → None  任意手柄松手

        # 静态层缓存
        self._tl_static_photo = None
        self._tl_dirty = True

        # 交互状态
        # active_handle: None
        #   | ('red',)
        #   | ('pause_in',  seg_id)
        #   | ('pause_out', seg_id)
        #   | ('clip_in',   seg_id)
        #   | ('clip_out',  seg_id)
        self.active_handle: object = None
        self._pending_candidates: list = []
        self._mousedown_x: int = 0
        self._pan_x: int = 0

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        self.canvas = tk.Canvas(self, height=self.TL_HEIGHT,
                                bg="#1A1A1A", highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_mousedown)
        self.canvas.bind("<B1-Motion>", self._on_mousemove)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouseup)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)

        # 操作提示
        hint = ("滚轮缩放  |  右键拖动平移  |  拖动 ▶ 红条定位  |  "
                "拖动橙黄条调整暂停裁剪  |  拖动青条调整片段裁剪")
        tk.Label(self, text=hint, fg="#555555", bg="#1A1A1A",
                 font=("Consolas", 8), anchor="w").pack(
            fill=tk.X, padx=4, pady=(0, 2))

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
        if w <= 1:
            w = 600
        h = self.TL_HEIGHT

        img = np.full((h, w, 3), (26, 26, 26), dtype=np.uint8)

        if self.total_frames <= 0:
            pil_img = PIL.Image.fromarray(img, mode='RGB')
            self._tl_static_photo = PIL.ImageTk.PhotoImage(image=pil_img)
            self._tl_dirty = False
            return

        # 1. 变速色条（底部）
        speed_colors = {
            FRAME_TYPE_1X: (30, 144, 255),
            FRAME_TYPE_2X: (147, 112, 219),
            FRAME_TYPE_0_2X: (60, 179, 113),
        }
        sy1, sy2 = h - 20, h
        for seg in self.speed_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)  # +1：画到帧的右边界
            if xe < 0 or xs > w: continue
            c = speed_colors.get(seg['type'])
            if c:
                x1, x2 = max(0, int(xs)), min(w, int(xe))
                if x2 > x1:
                    img[sy1:sy2, x1:x2] = c

        # 2. 暂停段色块（中间主区域）
        py1, py2 = _PAUSE_BAND_Y1, _PAUSE_BAND_Y2(h)
        hy1, hy2 = _PAUSE_HANDLE_Y1, _PAUSE_HANDLE_Y2(h)
        yellow = (255, 204, 0)
        dark = (68, 51, 0)
        orange = (255, 160, 0)  # 橙黄色手柄（比背景黄更饱和，易辨认）

        for seg in self.pause_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)  # +1：画到帧的右边界
            xti = self._f2x(seg['trim_in'], w)
            xto = self._f2x(seg['trim_out'], w)
            if xe < 0 or xs > w: continue

            def pfill(x1f, x2f, col, _img=img, _y1=py1, _y2=py2):
                xi1 = max(0, int(x1f));
                xi2 = min(w, int(x2f))
                if xi2 > xi1: _img[_y1:_y2, xi1:xi2] = col

            pfill(xs, xti, yellow)
            pfill(xto, xe, yellow)
            pfill(xti, xto, dark)

            # 橙黄色手柄（全高，与暂停区黄色同色系）
            for hx in (xti, xto):
                hxi = int(hx)
                if 0 <= hxi <= w:
                    xi1 = max(0, hxi - 3);
                    xi2 = min(w, hxi + 4)
                    img[hy1:hy2, xi1:xi2] = orange

        # 3. clip 段色块（青色系，画在暂停段上层，确保手柄可见）
        cy1, cy2 = _CLIP_BAND_Y1, _CLIP_BAND_Y2(h)
        chy1, chy2 = _CLIP_HANDLE_Y1, _CLIP_HANDLE_Y2(h)
        clip_keep = (32, 178, 170)
        clip_del = (20, 60, 60)
        cyan_hdl = (0, 230, 200)

        for seg in self.clip_segments:
            xs = self._f2x(seg['start'], w)
            xe = self._f2x(seg['end'] + 1, w)  # +1：画到帧的右边界
            xki = self._f2x(seg['keep_in'], w)
            xko = self._f2x(seg['keep_out'] + 1, w)  # +1：保留区含最后一帧
            if xe < 0 or xs > w: continue

            def cfill(x1f, x2f, col, _img=img, _y1=cy1, _y2=cy2):
                xi1 = max(0, int(x1f));
                xi2 = min(w, int(x2f))
                if xi2 > xi1: _img[_y1:_y2, xi1:xi2] = col

            cfill(xs, xki, clip_del)
            cfill(xki, xko, clip_keep)
            cfill(xko, xe, clip_del)

            for hx in (xki, xko):
                hxi = int(hx)
                if 0 <= hxi <= w:
                    xi1 = max(0, hxi - 3);
                    xi2 = min(w, hxi + 4)
                    img[chy1:chy2, xi1:xi2] = cyan_hdl

        pil_img = PIL.Image.fromarray(img, mode='RGB')
        self._tl_static_photo = PIL.ImageTk.PhotoImage(image=pil_img)
        self._tl_dirty = False

    # ------------------------------------------------------------------
    # 动态层（刻度 + 红条）
    # ------------------------------------------------------------------

    def _draw_dynamic(self):
        if self._tl_dirty or self._tl_static_photo is None:
            self._rebuild_static()

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
                    [px - 7, 0, px + 7, 0, px, 14],
                    fill="#FF4444", outline="#CC2222", width=1)

    def _draw_ticks(self, w: int, h: int):
        if self.total_frames <= 0 or self.fps <= 0: return
        px_per_frame = w * self.zoom_level / self.total_frames
        if px_per_frame < 4: return

        start_f = max(0, int(self.scroll_offset * self.total_frames) - 1)
        end_f = min(self.total_frames,
                    int((self.scroll_offset + 1.0 / self.zoom_level)
                        * self.total_frames) + 2)

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
                self.canvas.create_text(px + 2, 3, anchor=tk.NW, text=label,
                                        fill="#999999", font=("Consolas", 8))
            elif is_major:
                self.canvas.create_line(px, h - 30, px, h - 20, fill="#666666", width=1)
            else:
                self.canvas.create_line(px, h - 25, px, h - 20, fill="#444444", width=1)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def _f2x(self, frame_idx: int, canvas_w: int) -> float:
        if self.total_frames <= 0: return 0.0
        return (frame_idx / self.total_frames
                - self.scroll_offset) * self.zoom_level * canvas_w

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
        """收集所有在 event_x 附近的手柄候选"""
        cands = []
        RADIUS = 12  # 扩大命中半径，更容易抓到手柄

        for seg in self.pause_segments:
            ix = self._f2x(seg['trim_in'], canvas_w)
            ox = self._f2x(seg['trim_out'], canvas_w)
            if abs(event_x - ix) < RADIUS:
                cands.append((abs(event_x - ix), 'pause_in', seg['id']))
            if abs(event_x - ox) < RADIUS:
                cands.append((abs(event_x - ox), 'pause_out', seg['id']))

        for seg in self.clip_segments:
            ix = self._f2x(seg['keep_in'], canvas_w)
            ox = self._f2x(seg['keep_out'] + 1, canvas_w)  # +1 与绘制对齐
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
        # 点击时间轴抢回焦点，使空格能正常触发播放/暂停
        self.canvas.focus_set()
        if self.total_frames <= 0: return
        w = self.canvas.winfo_width()
        ey = event.y

        cands = self._collect_candidates(event.x, w)
        px = self._f2x(self.current_frame_idx, w)
        red_hit = abs(event.x - px) < 8

        # 优先级：三角区(y<14)红条优先；其余区域手柄优先，其次红条，最后seek
        if ey < 14:
            if red_hit:
                self.active_handle = ('red',);
                return

        if cands:
            self._pending_candidates = cands
            self._mousedown_x = event.x
            self.active_handle = None
            return

        if red_hit:
            self.active_handle = ('red',);
            return

        # seek
        tf = self._x2f(event.x, w)
        if self.on_seek_cb:
            self.on_seek_cb(tf)

    def _on_mousemove(self, event):
        w = self.canvas.winfo_width()

        # 红条拖动
        if self.active_handle and self.active_handle[0] == 'red':
            tf = self._x2f(event.x, w)
            self.current_frame_idx = tf
            if self.on_seek_cb:
                self.on_seek_cb(tf)
            self._draw_dynamic()
            return

        # 首次移动确定手柄（阈值0：按下即激活，不需要先移动2px）
        if self._pending_candidates and self.active_handle is None:
            dx = event.x - self._mousedown_x
            if True:  # 立即激活，dx仅用于消歧义方向
                self.active_handle = self._resolve_handle(dx)
                self._pending_candidates = []

        if not self.active_handle:
            return

        kind = self.active_handle[0]
        tf = self._x2f(event.x, w)

        if kind == 'pause_in':
            self._move_pause_handle(self.active_handle[1], 'in', tf)
        elif kind == 'pause_out':
            self._move_pause_handle(self.active_handle[1], 'out', tf)
        elif kind == 'clip_in':
            self._move_clip_handle(self.active_handle[1], 'in', tf)
        elif kind == 'clip_out':
            self._move_clip_handle(self.active_handle[1], 'out', tf)

        self.mark_dirty()
        self._draw_dynamic()

    def _resolve_handle(self, dx: int) -> tuple:
        """
        根据移动方向从候选中选出最合适的手柄。
        各手柄"向右移动有意义"的方向：
          pause_out → 右边界右移（扩大后保留区）→ 向右
          clip_in   → 左手柄右移（收缩保留区左端）→ 向右
          pause_in  → 左边界左移（扩大前保留区）→ 向左
          clip_out  → 右手柄左移（收缩保留区右端）→ 向左
        """
        if len(self._pending_candidates) == 1:
            _, htype, hid = self._pending_candidates[0]
            return (htype, hid)

        # 向右拖 → 优先选"向右移动自然"的手柄
        # 向左拖 → 优先选"向左移动自然"的手柄
        if dx > 0:
            prefer = {'pause_out', 'clip_in'}
        elif dx < 0:
            prefer = {'pause_in', 'clip_out'}
        else:
            prefer = set()  # dx==0时取距离最近的

        for _, htype, hid in sorted(self._pending_candidates):
            if htype in prefer:
                return (htype, hid)
        # 没有偏好命中，取距离最近的
        _, htype, hid = self._pending_candidates[0]
        return (htype, hid)

    def _move_pause_handle(self, seg_id: int, side: str, tf: int):
        for seg in self.pause_segments:
            if seg['id'] != seg_id: continue
            s0, s1 = seg['start'], seg['end']
            if side == 'in':
                seg['trim_in'] = max(s0, min(tf, s1 + 1))
                if seg['trim_in'] > seg['trim_out']:
                    seg['trim_out'] = min(seg['trim_in'], s1 + 1)
            else:
                seg['trim_out'] = min(s1 + 1, max(tf, s0))
                if seg['trim_out'] < seg['trim_in']:
                    seg['trim_in'] = max(seg['trim_out'], s0)
            break

    def _move_clip_handle(self, seg_id: int, side: str, tf: int):
        for seg in self.clip_segments:
            if seg['id'] != seg_id: continue
            s0, s1 = seg['start'], seg['end']
            if side == 'in':
                # keep_in 可到 s1+1（表示全删，keep_in > keep_out）
                seg['keep_in'] = max(s0, min(tf, s1 + 1))
                if seg['keep_in'] > seg['keep_out']:
                    seg['keep_out'] = min(seg['keep_in'] - 1, s1)
            else:
                # keep_out 可到 s0-1（表示全删，keep_out < keep_in）
                seg['keep_out'] = min(s1, max(tf, s0 - 1))
                if seg['keep_out'] < seg['keep_in']:
                    seg['keep_in'] = max(seg['keep_out'] + 1, s0)
            break

    def _on_mouseup(self, event):
        self.active_handle = None
        self._pending_candidates = []
        self._mousedown_x = 0
        if self.on_handle_end_cb:
            self.on_handle_end_cb()

    def _on_scroll(self, event):
        if self.total_frames <= 0: return
        self.zoom_level *= (1.2 if event.delta > 0 else 1 / 1.2)
        self.zoom_level = max(1.0, min(self.zoom_level, 500.0))
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
        self.scroll_offset = max(0.0, min(self.scroll_offset - move,
                                          1.0 - 1.0 / self.zoom_level))
        self.mark_dirty()
        self._draw_dynamic()