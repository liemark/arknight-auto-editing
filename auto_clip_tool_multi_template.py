# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import threading
import time
import numpy as np
import os
from queue import Queue
import concurrent.futures
import functools

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# --- 帧状态定义 ---
FRAME_TYPE_NORMAL  = 0
FRAME_TYPE_PAUSE   = 1
FRAME_TYPE_1X      = 2
FRAME_TYPE_2X      = 3
FRAME_TYPE_0_2X    = 4


# ============================================================
#  设置面板（独立 Frame，可嵌入主窗口）
# ============================================================
class SettingsPanel(ttk.LabelFrame):
    """包含旧版全部参数设置的面板"""
    def __init__(self, parent, **kw):
        super().__init__(parent, text="处理参数", padding=8, **kw)
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_basic = ttk.Frame(nb, padding=6)
        tab_match  = ttk.Frame(nb, padding=6)
        tab_pause  = ttk.Frame(nb, padding=6)
        tab_export = ttk.Frame(nb, padding=6)

        nb.add(tab_basic,  text="基本")
        nb.add(tab_match,  text="匹配阈值")
        nb.add(tab_pause,  text="暂停处理")
        nb.add(tab_export, text="导出")

        # ---- 基本 ----
        r = 0
        ttk.Label(tab_basic, text="批处理大小:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=128)
        ttk.Spinbox(tab_basic, from_=1, to=512, textvariable=self.batch_size_var, width=6).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_basic, text="处理宽度:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.proc_w_var = tk.IntVar(value=400)
        ttk.Spinbox(tab_basic, from_=100, to=1920, textvariable=self.proc_w_var, width=6).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_basic, text="处理高度:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.proc_h_var = tk.IntVar(value=225)
        ttk.Spinbox(tab_basic, from_=100, to=1080, textvariable=self.proc_h_var, width=6).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_basic, text="线程数:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.thread_var = tk.IntVar(value=max(1, os.cpu_count() or 4))
        ttk.Spinbox(tab_basic, from_=1, to=64, textvariable=self.thread_var, width=6).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Separator(tab_basic, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=4)

        r += 1
        self.speedup_1x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_basic, text="1x 区域以 2x 播放", variable=self.speedup_1x_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W)

        r += 1
        self.speedup_02x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_basic, text="0.2x 区域加速", variable=self.speedup_02x_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W)

        r += 1
        ttk.Label(tab_basic, text="0.2x 加速倍率:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.speedup_02x_factor_var = tk.IntVar(value=10)
        ttk.Spinbox(tab_basic, from_=2, to=20, textvariable=self.speedup_02x_factor_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)

        # ---- 匹配阈值 ----
        self.thr_pause_var  = tk.DoubleVar(value=0.7)
        self.thr_1x_var     = tk.DoubleVar(value=0.7)
        self.thr_2x_var     = tk.DoubleVar(value=0.7)
        self.thr_02x_var    = tk.DoubleVar(value=0.7)
        rows = [
            ("暂停阈值:",  self.thr_pause_var),
            ("1x 阈值:",   self.thr_1x_var),
            ("2x 阈值:",   self.thr_2x_var),
            ("0.2x 阈值:", self.thr_02x_var),
        ]
        for i, (lbl, var) in enumerate(rows):
            ttk.Label(tab_match, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(tab_match, from_=0.1, to=1.0, increment=0.01, textvariable=var, width=8).grid(
                row=i, column=1, sticky=tk.W, padx=4)

        # ---- 暂停处理 ----
        r = 0
        self.compare_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_pause, text="启用画面对比 (防止跳帧)", variable=self.compare_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2)

        r += 1
        ttk.Label(tab_pause, text="DCT 低频差异阈值:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.dct_thresh_var = tk.DoubleVar(value=50000.0)
        ttk.Spinbox(tab_pause, from_=1000, to=1000000, increment=5000, textvariable=self.dct_thresh_var,
                    width=10).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_pause, text="Hist 相似度阈值:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.hist_thresh_var = tk.DoubleVar(value=0.95)
        ttk.Spinbox(tab_pause, from_=0.1, to=1.0, increment=0.01, textvariable=self.hist_thresh_var,
                    width=8).grid(row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_pause, text="保留前段 (帧):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.keep_before_var = tk.IntVar(value=0)
        ttk.Spinbox(tab_pause, from_=0, to=300, textvariable=self.keep_before_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_pause, text="保留后段 (帧):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.keep_after_var = tk.IntVar(value=60)
        ttk.Spinbox(tab_pause, from_=0, to=300, textvariable=self.keep_after_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)

        # ---- 导出 ----
        r = 0
        ttk.Label(tab_export, text="输出路径:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar()
        ttk.Entry(tab_export, textvariable=self.output_var, width=14).grid(row=r, column=1, sticky=tk.EW, padx=4)
        ttk.Button(tab_export, text="浏览", command=self._browse_output).grid(row=r, column=2)
        tab_export.columnconfigure(1, weight=1)

        r += 1
        ttk.Label(tab_export, text="视频质量 (0-10):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.quality_var = tk.IntVar(value=6)
        ttk.Spinbox(tab_export, from_=0, to=10, textvariable=self.quality_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)

        r += 1
        self.export_btn = ttk.Button(tab_export, text="导出剪辑视频", command=self._on_export)
        self.export_btn.grid(row=r, column=0, columnspan=3, pady=8)

        r += 1
        self.export_progress_var = tk.DoubleVar()
        self.export_progress = ttk.Progressbar(tab_export, variable=self.export_progress_var, maximum=100)
        self.export_progress.grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=2)

        r += 1
        self.export_status_var = tk.StringVar(value="就绪")
        ttk.Label(tab_export, textvariable=self.export_status_var).grid(row=r, column=0, columnspan=3)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".mp4",
                                         filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("所有", "*.*")])
        if p:
            self.output_var.set(p)

    def _on_export(self):
        # 委托给主应用
        if hasattr(self, 'export_callback') and self.export_callback:
            self.export_callback()

    def get_params(self):
        return {
            'batch':        self.batch_size_var.get(),
            'proc_res':     (self.proc_w_var.get(), self.proc_h_var.get()),
            'threads':      self.thread_var.get(),
            'speedup_1x':   self.speedup_1x_var.get(),
            'speedup_02':   self.speedup_02x_var.get(),
            'speedup_02_factor': self.speedup_02x_factor_var.get(),
            'thresholds':   {
                'pause':     self.thr_pause_var.get(),
                'speed_1x':  self.thr_1x_var.get(),
                'speed_2x':  self.thr_2x_var.get(),
                'speed_0_2x': self.thr_02x_var.get(),
            },
            'compare': {
                'enabled': self.compare_var.get(),
                'dct':     self.dct_thresh_var.get(),
                'hist':    self.hist_thresh_var.get(),
                'keep_n':  self.keep_before_var.get(),
                'keep_m':  self.keep_after_var.get(),
            },
            'output':   self.output_var.get(),
            'quality':  self.quality_var.get(),
        }


# ============================================================
#  视频预览播放器
# ============================================================
class VideoPreviewPlayer(tk.Frame):
    def __init__(self, parent, settings: SettingsPanel, video_path=None, width=640, height=360):
        super().__init__(parent)
        self.settings = settings
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame_idx = 0

        self.canvas_w = width
        self.canvas_h = height

        self.zoom_level = 1.0
        self.scroll_offset = 0.0

        self.template_config = {
            'pause':      {'templates': [], 'ref_dir': 'templates_pause',  'source_dir': 'source_images_pause'},
            'speed_1x':   {'templates': [], 'ref_dir': 'templates_1x',     'source_dir': 'source_images_1x'},
            'speed_2x':   {'templates': [], 'ref_dir': 'templates_2x',     'source_dir': 'source_images_2x'},
            'speed_0_2x': {'templates': [], 'ref_dir': 'templates_play',   'source_dir': 'source_images_play'},
        }

        self.pause_segments = []
        self.speed_segments  = []
        self.states_array    = None   # np 数组，供导出时使用

        self.is_playing  = False
        self.stop_thread = False
        self.playback_speed = 1.0
        self.frame_queue = Queue(maxsize=1)

        self.active_handle = None
        self._pending_candidates = []
        self._mousedown_x = 0

        self.setup_ui()
        if video_path:
            self.load_video(video_path)

    # ----------------------------------------------------------
    #  UI 构建
    # ----------------------------------------------------------
    def setup_ui(self):
        self.video_canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.video_canvas.pack(pady=5, fill=tk.BOTH, expand=True)

        self.timeline_frame = tk.Frame(self)
        self.timeline_frame.pack(fill=tk.X, padx=10)

        self.timeline_h = 80
        self.timeline_canvas = tk.Canvas(self.timeline_frame, height=self.timeline_h, bg="#1A1A1A",
                                         highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, side=tk.TOP)

        self.timeline_canvas.bind("<Button-1>",       self.on_mousedown)
        self.timeline_canvas.bind("<B1-Motion>",      self.on_mousemove)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.on_mouseup)
        self.timeline_canvas.bind("<MouseWheel>",     self.on_timeline_scroll)
        self.timeline_canvas.bind("<Button-3>",       self.start_pan)
        self.timeline_canvas.bind("<B3-Motion>",      self.pan_timeline)

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, pady=5)

        self.btn_play = ttk.Button(ctrl_frame, text="播放/暂停", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="预览倍速:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_combo = ttk.Combobox(ctrl_frame, values=["0.2", "0.5", "1.0", "1.5", "2.0"], width=5)
        self.speed_combo.set("1.0")
        self.speed_combo.pack(side=tk.LEFT, padx=5)
        self.speed_combo.bind("<<ComboboxSelected>>", self.update_speed)

        self.btn_analyze = ttk.Button(ctrl_frame, text="自动模板分析", command=self.start_analysis_thread)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        self.skip_trimmed = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="预览时跳过裁剪区", variable=self.skip_trimmed).pack(side=tk.LEFT, padx=5)

        self.lbl_time = ttk.Label(ctrl_frame, text="00:00 / 00:00")
        self.lbl_time.pack(side=tk.RIGHT, padx=10)

        self.lbl_info1 = ttk.Label(self, text="时间轴滚轮缩放，右键拖动",
                                   foreground="#118811", font=("Consolas", 10))
        self.lbl_info1.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_info2 = ttk.Label(self, text="黄色为暂停区域，两侧白条可拖动，自定义剪辑",
                                   foreground="#118811", font=("Consolas", 10))
        self.lbl_info2.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_info = ttk.Label(self, text="状态: 就绪 | DCT: - | Hist: -",
                                  foreground="#00CED1", font=("Consolas", 10))
        self.lbl_info.pack(fill=tk.X, padx=10, pady=2)

        self.update_canvas_loop()

    # ----------------------------------------------------------
    #  视频加载
    # ----------------------------------------------------------
    def load_video(self, path):
        self.video_path = path
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.scroll_offset = 0.0
        self.pause_segments.clear()
        self.speed_segments.clear()
        self.states_array = None
        # 自动填充输出路径
        if not self.settings.output_var.get():
            name, ext = os.path.splitext(path)
            self.settings.output_var.set(f"{name}_clipped.mp4")
        self.seek(0)
        self.draw_timeline()

    # ----------------------------------------------------------
    #  播放控制
    # ----------------------------------------------------------
    def update_speed(self, event=None):
        self.playback_speed = float(self.speed_combo.get())

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            threading.Thread(target=self.video_decode_worker, daemon=True).start()

    def check_and_jump_frame(self, frame_idx):
        if not self.skip_trimmed.get():
            return frame_idx
        for seg in self.pause_segments:
            if seg["trim_in"] <= frame_idx < seg["trim_out"]:
                return seg["trim_out"]
        return frame_idx

    def get_frame_step(self, frame_idx):
        p = self.settings.get_params()
        for seg in self.speed_segments:
            if seg["start"] <= frame_idx <= seg["end"]:
                if seg["type"] == FRAME_TYPE_1X and p['speedup_1x']:
                    return 2
                if seg["type"] == FRAME_TYPE_0_2X and p['speedup_02']:
                    return max(2, p['speedup_02_factor'])
        return 1

    def video_decode_worker(self):
        while self.is_playing and not self.stop_thread:
            start_time = time.time()
            target_idx = self.check_and_jump_frame(self.current_frame_idx)
            step = self.get_frame_step(target_idx)
            if target_idx != self.current_frame_idx:
                self.current_frame_idx = target_idx
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                break

            if step > 1:
                for _ in range(step - 1):
                    self.cap.grab()
                self.current_frame_idx += step
            else:
                self.current_frame_idx += 1

            self.ensure_pointer_visible()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)

            base_delay = 1.0 / self.fps
            target_delay = base_delay / max(self.playback_speed, 0.01)
            elapsed = time.time() - start_time
            time.sleep(max(target_delay - elapsed, 0))

    def ensure_pointer_visible(self):
        if self.total_frames <= 0: return
        p_ratio = self.current_frame_idx / self.total_frames
        vw_ratio = 1.0 / self.zoom_level
        if p_ratio < self.scroll_offset or p_ratio > self.scroll_offset + vw_ratio:
            self.scroll_offset = max(0, min(p_ratio - vw_ratio / 2, 1.0 - vw_ratio))

    def seek(self, frame_idx):
        if not self.cap: return
        idx = self.check_and_jump_frame(max(0, min(frame_idx, self.total_frames - 1)))
        self.current_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.draw_timeline()

    def display_frame(self, frame):
        img = PIL.Image.fromarray(frame)
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw > 1:
            img.thumbnail((cw, ch))
        self.photo = PIL.ImageTk.PhotoImage(image=img)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            max(cw, self.canvas_w) // 2,
            max(ch, self.canvas_h) // 2,
            image=self.photo)
        self.update_time_label()

    def update_canvas_loop(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.display_frame(frame)
            self.draw_timeline()
        self.after(10, self.update_canvas_loop)

    # ----------------------------------------------------------
    #  模板加载
    # ----------------------------------------------------------
    def prepare_templates_and_rois(self):
        total_loaded = 0
        for ctype, config in self.template_config.items():
            config['templates'].clear()
            if not os.path.exists(config['source_dir']): continue
            src_files = [f for f in os.listdir(config['source_dir'])
                         if f.lower().endswith(('.png', '.jpg', '.bmp'))]
            if not src_files: continue

            src_img = cv2.imread(os.path.join(config['source_dir'], src_files[0]), cv2.IMREAD_GRAYSCALE)
            if src_img is None: continue
            sh, sw = src_img.shape

            if not os.path.exists(config['ref_dir']): continue
            ref_files = [f for f in os.listdir(config['ref_dir'])
                         if f.lower().endswith(('.png', '.jpg', '.bmp'))]
            for rf in ref_files:
                ref_img = cv2.imread(os.path.join(config['ref_dir'], rf), cv2.IMREAD_GRAYSCALE)
                if ref_img is None: continue

                rh, rw = ref_img.shape
                # 兼容性：若模板大于源图，跳过
                if rh > sh or rw > sw:
                    print(f"[警告] 模板 {rf} ({rw}x{rh}) 大于源图 ({sw}x{sh})，已跳过")
                    continue

                res = cv2.matchTemplate(src_img, ref_img, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                _, mask = cv2.threshold(ref_img, 10, 255, cv2.THRESH_BINARY)

                config['templates'].append({
                    'template_gray': ref_img,
                    'mask':          mask,
                    'roi_orig':      (max_loc[0], max_loc[1], rw, rh),
                    'source_res':    (sw, sh),
                    'name':          rf,
                })
                total_loaded += 1
        return total_loaded

    # ----------------------------------------------------------
    #  帧匹配
    # ----------------------------------------------------------
    def get_best_score(self, gray_frame, templates, proc_res):
        max_score = -1.0
        for t in templates:
            sw, sh = t['source_res']
            rx, ry, rw, rh = t['roi_orig']

            scale_x = proc_res[0] / sw
            scale_y = proc_res[1] / sh

            ext = 2.0
            erx = max(0, int(rx * scale_x - rw * scale_x * (ext - 1) / 2))
            ery = max(0, int(ry * scale_y - rh * scale_y * (ext - 1) / 2))
            erw = min(gray_frame.shape[1] - erx, int(rw * scale_x * ext))
            erh = min(gray_frame.shape[0] - ery, int(rh * scale_y * ext))

            if erw <= 0 or erh <= 0: continue
            roi_img = gray_frame[ery:ery + erh, erx:erx + erw]

            tw = max(1, int(rw * scale_x))
            th = max(1, int(rh * scale_y))
            if roi_img.shape[0] < th or roi_img.shape[1] < tw: continue

            t_resized = cv2.resize(t['template_gray'], (tw, th), interpolation=cv2.INTER_AREA)
            m_resized = cv2.resize(t['mask'],          (tw, th), interpolation=cv2.INTER_NEAREST)

            res = cv2.matchTemplate(roi_img, t_resized, cv2.TM_CCOEFF_NORMED, mask=m_resized)
            _, score, _, _ = cv2.minMaxLoc(res)
            if np.isfinite(score):
                max_score = max(max_score, score)
        return max_score

    def analyze_worker(self, frame, thresholds, proc_res, scaled_configs):
        resized = cv2.resize(frame, proc_res, interpolation=cv2.INTER_AREA)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 优先级：暂停 > 1x/2x > 0.2x > 普通
        if scaled_configs['pause'] and \
                self.get_best_score(gray, scaled_configs['pause'], proc_res) >= thresholds['pause']:
            return FRAME_TYPE_PAUSE

        x1s = self.get_best_score(gray, scaled_configs['speed_1x'],   proc_res) if scaled_configs['speed_1x']   else -1.0
        x2s = self.get_best_score(gray, scaled_configs['speed_2x'],   proc_res) if scaled_configs['speed_2x']   else -1.0

        if x1s >= thresholds['speed_1x']  and x1s > x2s: return FRAME_TYPE_1X
        if x2s >= thresholds['speed_2x']  and x2s > x1s: return FRAME_TYPE_2X

        if scaled_configs['speed_0_2x'] and \
                self.get_best_score(gray, scaled_configs['speed_0_2x'], proc_res) >= thresholds['speed_0_2x']:
            return FRAME_TYPE_0_2X

        return FRAME_TYPE_NORMAL

    # ----------------------------------------------------------
    #  分析主流程
    # ----------------------------------------------------------
    def start_analysis_thread(self):
        if not self.video_path: return
        self.btn_analyze.config(state=tk.DISABLED, text="加载模板中...")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        loaded = self.prepare_templates_and_rois()
        if loaded == 0:
            self.after(0, lambda: messagebox.showwarning(
                "模板缺失", "未检测到可用模板，将标记所有帧为普通帧。"))

        p = self.settings.get_params()
        proc_res   = p['proc_res']
        thresholds = p['thresholds']
        batch_size = p['batch']
        threads    = p['threads']
        compare    = p['compare']

        cap   = cv2.VideoCapture(self.video_path)
        total = self.total_frames

        # 若用户未手动设置处理尺寸，则按视频比例自动算高度
        ret, test_f = cap.read()
        if ret:
            h, w = test_f.shape[:2]
            if proc_res == (400, 225):           # 默认值时自动修正
                proc_res = (400, int(400 * h / w))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        states = np.zeros(total, dtype=int)
        configs = {k: v['templates'] for k, v in self.template_config.items()}

        # 1. 帧状态识别
        if loaded > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
                for i in range(0, total, batch_size):
                    frames, indices = [], []
                    for j in range(i, min(i + batch_size, total)):
                        ret, f = cap.read()
                        if not ret: break
                        frames.append(f)
                        indices.append(j)

                    results = list(ex.map(
                        functools.partial(self.analyze_worker,
                                          thresholds=thresholds,
                                          proc_res=proc_res,
                                          scaled_configs=configs),
                        frames))
                    for idx, s in zip(indices, results):
                        states[idx] = s

                    p_pct = (min(i + batch_size, total) / total) * 100
                    self.after(0, lambda v=p_pct: self.btn_analyze.config(text=f"模板匹配 {int(v)}%"))
        cap.release()

        self.states_array = states   # 保存供导出使用

        # 2. 生成段列表
        self.after(0, lambda: self.btn_analyze.config(text="对比片段差异..."))
        pauses = []
        speeds = []

        keep_n = compare['keep_n']
        keep_m = compare['keep_m']

        i = 0
        while i < total:
            curr_state = states[i]
            start_i = i
            while i < total and states[i] == curr_state: i += 1
            end_i = i - 1
            seg_len = end_i - start_i + 1

            if curr_state == FRAME_TYPE_PAUSE:
                # 前后对比帧
                cap_tmp = cv2.VideoCapture(self.video_path)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_i - 1))
                _, f_b = cap_tmp.read()
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, end_i + 1))
                _, f_a = cap_tmp.read()
                cap_tmp.release()

                is_diff = True
                dct_v, h_sim = 0.0, 0.0

                if compare['enabled'] and f_b is not None and f_a is not None:
                    g_b = cv2.cvtColor(cv2.resize(f_b, proc_res), cv2.COLOR_BGR2GRAY)
                    g_a = cv2.cvtColor(cv2.resize(f_a, proc_res), cv2.COLOR_BGR2GRAY)

                    dct_b = cv2.dct(np.float32(g_b))[:8, :8]
                    dct_a = cv2.dct(np.float32(g_a))[:8, :8]
                    dct_v = float(np.sum(np.abs(dct_a - dct_b)))

                    hist_b = cv2.calcHist([g_b], [0], None, [256], [0, 256])
                    hist_a = cv2.calcHist([g_a], [0], None, [256], [0, 256])
                    h_sim  = float(cv2.compareHist(hist_b, hist_a, cv2.HISTCMP_CORREL))

                    if dct_v < compare['dct'] and h_sim > compare['hist']:
                        is_diff = False

                # -------------------------------------------------------
                # [修复] 计算 trim_in / trim_out
                #   - is_diff=True : 保留前 keep_n 帧 + 后 keep_m 帧，删中间
                #   - is_diff=False: 全删
                #   - 段长不足时：尽量各取一半，保证 trim_in <= trim_out
                # -------------------------------------------------------
                if is_diff:
                    n = min(keep_n, seg_len)          # 实际可保留前段
                    m = min(keep_m, seg_len - n)       # 实际可保留后段（不超过剩余）
                    trim_in  = start_i + n             # 删除区起点
                    trim_out = end_i - m + 1           # 删除区终点（exclusive 写入时用）

                    # 若 trim_in > trim_out（段太短，前后保留撑满），
                    # 则不裁剪该段（保留全部）
                    if trim_in > trim_out:
                        trim_in  = start_i
                        trim_out = start_i             # 空删除区
                else:
                    trim_in  = start_i
                    trim_out = end_i + 1               # 全删

                pauses.append({
                    "id":       len(pauses),
                    "start":    start_i,
                    "end":      end_i,
                    "trim_in":  trim_in,
                    "trim_out": trim_out,
                    "dct":      dct_v,
                    "hist":     h_sim,
                    "is_diff":  is_diff,
                })

            elif curr_state in (FRAME_TYPE_1X, FRAME_TYPE_2X, FRAME_TYPE_0_2X):
                speeds.append({
                    "type":  curr_state,
                    "start": start_i,
                    "end":   end_i,
                })

        self.after(0, lambda: self.finish_analysis(pauses, speeds))

    def finish_analysis(self, pauses, speeds):
        self.pause_segments = pauses
        self.speed_segments  = speeds
        self.btn_analyze.config(state=tk.NORMAL, text="自动模板分析")
        self.draw_timeline()
        messagebox.showinfo("分析完成",
                            f"识别到 {len(pauses)} 处暂停，{len(speeds)} 个变速区间。")

    # ----------------------------------------------------------
    #  导出
    # ----------------------------------------------------------
    def export_video(self):
        """由 SettingsPanel 的导出按钮触发"""
        if not self.video_path:
            messagebox.showerror("错误", "请先加载视频")
            return
        p = self.settings.get_params()
        out_path = p['output']
        if not out_path:
            messagebox.showerror("错误", "请先设置输出路径")
            return

        self.settings.export_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._export_worker, args=(p, out_path), daemon=True).start()

    def _export_worker(self, params, output_path):
        try:
            total    = self.total_frames
            states   = self.states_array
            pauses   = self.pause_segments
            speeds   = self.speed_segments

            # 如果未做分析，states 为 None，默认全普通
            if states is None:
                states = np.zeros(total, dtype=int)

            # 建立删除帧集合
            frames_to_delete = set()

            # 1. 暂停段裁剪
            for seg in pauses:
                trim_in  = seg["trim_in"]
                trim_out = seg["trim_out"]
                if trim_out > trim_in:
                    frames_to_delete.update(range(trim_in, trim_out))

            # 2. 1x 变速抽帧（每隔 1 帧删 1 帧，等效 2x）
            if params['speedup_1x']:
                cnt = 0
                for idx in range(total):
                    if states[idx] == FRAME_TYPE_1X and idx not in frames_to_delete:
                        cnt += 1
                        if cnt % 2 == 0:
                            frames_to_delete.add(idx)
                    else:
                        cnt = 0

            # 3. 0.2x 变速抽帧
            if params['speedup_02'] and params['speedup_02_factor'] > 1:
                cnt    = 0
                factor = params['speedup_02_factor']
                for idx in range(total):
                    if states[idx] == FRAME_TYPE_0_2X and idx not in frames_to_delete:
                        cnt += 1
                        if cnt % factor != 1:
                            frames_to_delete.add(idx)
                    else:
                        cnt = 0

            # 4. 写入
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

            def upd(pct, msg):
                self.settings.export_progress_var.set(pct)
                self.settings.export_status_var.set(msg)

            self.after(0, lambda: upd(0, "正在写入..."))

            if HAS_IMAGEIO:
                import imageio
                writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                            quality=params['quality'], pixelformat='yuv420p')
                written = 0
                for idx in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    if idx not in frames_to_delete:
                        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        written += 1
                    if idx % 100 == 0:
                        self.after(0, lambda v=(idx / total) * 100: upd(v, f"写入中: {int(v)}%"))
                writer.close()
            else:
                # 降级：使用 OpenCV VideoWriter
                ret, sample = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not ret:
                    raise RuntimeError("无法读取视频帧")
                h, w = sample.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                written = 0
                for idx in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    if idx not in frames_to_delete:
                        vw.write(frame)
                        written += 1
                    if idx % 100 == 0:
                        self.after(0, lambda v=(idx / total) * 100: upd(v, f"写入中: {int(v)}%"))
                vw.release()

            cap.release()
            self.after(0, lambda: upd(100, f"完成！保留 {written}/{total} 帧"))
            self.after(0, lambda: messagebox.showinfo(
                "导出完成", f"输出：{output_path}\n总帧数：{total}，保留：{written}"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("导出失败", str(e)))
        finally:
            self.after(0, lambda: self.settings.export_btn.config(state=tk.NORMAL))

    # ----------------------------------------------------------
    #  时间轴交互
    # ----------------------------------------------------------
    def on_timeline_scroll(self, event):
        if self.total_frames <= 0: return
        self.zoom_level = self.zoom_level * 1.2 if event.delta > 0 else self.zoom_level / 1.2
        self.zoom_level = max(1.0, min(self.zoom_level, 200.0))
        p_ratio  = self.current_frame_idx / self.total_frames
        vw_ratio = 1.0 / self.zoom_level
        self.scroll_offset = max(0, min(p_ratio - vw_ratio / 2, 1.0 - vw_ratio))
        self.draw_timeline()

    def start_pan(self, event):
        self.pan_start_x = event.x

    def pan_timeline(self, event):
        if self.zoom_level <= 1.0: return
        dx = event.x - self.pan_start_x
        self.pan_start_x = event.x
        move_ratio = (dx / self.timeline_canvas.winfo_width()) / self.zoom_level
        self.scroll_offset = max(0, min(self.scroll_offset - move_ratio, 1.0 - 1.0 / self.zoom_level))
        self.draw_timeline()

    def on_mousedown(self, event):
        if self.total_frames <= 0: return
        w = self.timeline_canvas.winfo_width()

        # 收集所有候选手柄：(距离, seg_id, h_type)
        candidates = []
        for seg in self.pause_segments:
            in_x  = self.frame_to_x(seg["trim_in"],  w)
            out_x = self.frame_to_x(seg["trim_out"], w)
            if abs(event.x - in_x)  < 10: candidates.append((abs(event.x - in_x),  seg["id"], "in"))
            if abs(event.x - out_x) < 10: candidates.append((abs(event.x - out_x), seg["id"], "out"))

        if not candidates:
            self.on_timeline_click(event)
            return

        # 重叠时的消歧义策略：
        #   若只有一个候选 → 直接选它
        #   若有多个（两条重叠）→ 根据鼠标相对于重叠位置偏向哪侧来选：
        #       鼠标在重叠点左半（或正中）→ 选 "out"（右条往左拉更自然）
        #       鼠标在重叠点右半         → 选 "in" （左条往右推更自然）
        #   实际上：重叠时 in_x ≈ out_x，我们按 h_type 优先级：
        #       按下时不知道拖动方向，先都记下，在首次 mousemove 时再决定。
        self._pending_candidates = candidates   # 待 mousemove 确认
        self._mousedown_x = event.x
        self.active_handle = None               # 尚未确认

    def on_mousemove(self, event):
        w = self.timeline_canvas.winfo_width()

        # --- 首次移动：从候选中用方向消歧义确定 active_handle ---
        if hasattr(self, '_pending_candidates') and self._pending_candidates and self.active_handle is None:
            dx = event.x - self._mousedown_x
            if abs(dx) >= 2:                    # 至少移动 2px 才判断方向
                chosen = None
                if len(self._pending_candidates) == 1:
                    chosen = (self._pending_candidates[0][1], self._pending_candidates[0][2])
                else:
                    # 有多个候选（重叠）：向右拖 → 选 "out"（推右条向右）
                    #                      向左拖 → 选 "in" （推左条向左）
                    prefer = "out" if dx > 0 else "in"
                    for _, sid, htype in sorted(self._pending_candidates):
                        if htype == prefer:
                            chosen = (sid, htype); break
                    if chosen is None:           # 没找到偏好类型，取最近的
                        chosen = (self._pending_candidates[0][1], self._pending_candidates[0][2])
                self.active_handle = chosen
                self._pending_candidates = []

        if not self.active_handle:
            # 还没确认手柄但已经在拖（可能还没超过 2px 阈值）
            if event.state & 0x0100 and not hasattr(self, '_pending_candidates'):
                self.on_timeline_click(event)
            return

        target_frame = self.x_to_frame(event.x, w)
        seg_id, h_type = self.active_handle

        for seg in self.pause_segments:
            if seg["id"] != seg_id: continue
            seg_start = seg["start"]
            seg_end   = seg["end"]

            if h_type == "in":
                new_in = max(seg_start, min(target_frame, seg_end + 1))
                seg["trim_in"] = new_in
                # 推挤：若 trim_in 超过了 trim_out，带着 trim_out 一起走
                if seg["trim_in"] > seg["trim_out"]:
                    seg["trim_out"] = min(seg["trim_in"], seg_end + 1)
            else:
                new_out = min(seg_end + 1, max(target_frame, seg_start))
                seg["trim_out"] = new_out
                # 推挤：若 trim_out 退到了 trim_in 左边，带着 trim_in 一起走
                if seg["trim_out"] < seg["trim_in"]:
                    seg["trim_in"] = max(seg["trim_out"], seg_start)
            break

        self.draw_timeline()

    def on_mouseup(self, event):
        self.active_handle = None
        self._pending_candidates = []
        self._mousedown_x = 0

    def on_timeline_click(self, event):
        if self.total_frames <= 0: return
        self.is_playing = False
        target_frame = self.x_to_frame(event.x, self.timeline_canvas.winfo_width())
        self.seek(target_frame)

    def frame_to_x(self, frame_idx, canvas_width):
        global_ratio   = frame_idx / self.total_frames
        relative_ratio = (global_ratio - self.scroll_offset) * self.zoom_level
        return relative_ratio * canvas_width

    def x_to_frame(self, x, canvas_width):
        view_ratio   = x / canvas_width
        global_ratio = self.scroll_offset + (view_ratio / self.zoom_level)
        return int(max(0, min(global_ratio, 1.0)) * self.total_frames)

    # ----------------------------------------------------------
    #  时间轴绘制
    # ----------------------------------------------------------
    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        w = self.timeline_canvas.winfo_width()
        if w <= 1: w = 600
        h = self.timeline_h
        self.timeline_canvas.create_rectangle(0, 0, w, h, fill="#1A1A1A", outline="")

        if self.total_frames <= 0: return

        # 变速区域（底部条）
        for seg in self.speed_segments:
            x_s = self.frame_to_x(seg["start"], w)
            x_e = self.frame_to_x(seg["end"],   w)
            if x_e < 0 or x_s > w: continue
            color = {FRAME_TYPE_1X: "#1E90FF", FRAME_TYPE_2X: "#9370DB",
                     FRAME_TYPE_0_2X: "#3CB371"}.get(seg["type"], "")
            if color:
                self.timeline_canvas.create_rectangle(
                    max(0, x_s), h - 20, min(w, x_e), h, fill=color, outline="")

        # 暂停保留 / 裁剪区域
        for seg in self.pause_segments:
            x_start    = self.frame_to_x(seg["start"],    w)
            x_end      = self.frame_to_x(seg["end"],      w)
            x_trim_in  = self.frame_to_x(seg["trim_in"],  w)
            x_trim_out = self.frame_to_x(seg["trim_out"], w)
            if x_end < 0 or x_start > w: continue

            # 保留区（亮黄）：暂停段起始 → trim_in
            if x_trim_in > x_start:
                self.timeline_canvas.create_rectangle(
                    max(0, x_start), 15, min(w, x_trim_in), h - 25, fill="#FFCC00", outline="")
            # 保留区（亮黄）：trim_out → 暂停段结束
            if x_end > x_trim_out:
                self.timeline_canvas.create_rectangle(
                    max(0, x_trim_out), 15, min(w, x_end), h - 25, fill="#FFCC00", outline="")
            # 裁剪区（深棕）
            if x_trim_out > x_trim_in:
                self.timeline_canvas.create_rectangle(
                    max(0, x_trim_in), 15, min(w, x_trim_out), h - 25, fill="#443300", outline="")

            # 白色手柄（trim_in 和 trim_out 均在视口内才画）
            if 0 <= x_trim_in <= w:
                self.timeline_canvas.create_rectangle(
                    x_trim_in - 3, 10, x_trim_in + 3, h - 20, fill="#FFFFFF", outline="")
            if 0 <= x_trim_out <= w:
                self.timeline_canvas.create_rectangle(
                    x_trim_out - 3, 10, x_trim_out + 3, h - 20, fill="#FFFFFF", outline="")

        # 当前播放指针
        px = self.frame_to_x(self.current_frame_idx, w)
        if 0 <= px <= w:
            self.timeline_canvas.create_line(px, 0, px, h, fill="#FF0000", width=2)
            self.timeline_canvas.create_polygon([px - 6, 0, px + 6, 0, px, 10], fill="#FF0000")

    # ----------------------------------------------------------
    #  状态栏
    # ----------------------------------------------------------
    def update_time_label(self):
        cur_sec = self.current_frame_idx / self.fps
        tot_sec = self.total_frames / self.fps
        self.lbl_time.config(text=f"{self.format_time(cur_sec)} / {self.format_time(tot_sec)}")

        info_text = "普通区域"
        in_pause  = False

        for seg in self.pause_segments:
            if seg["start"] <= self.current_frame_idx <= seg["end"]:
                info_text = (f"暂停区域 | DCT差异: {seg.get('dct', 0):.1f} | "
                             f"Hist相似度: {seg.get('hist', 0):.4f} | "
                             f"画面跳变: {'是 (保留端点)' if seg.get('is_diff') else '否 (全部剪除)'}")
                in_pause = True
                break

        if not in_pause:
            p = self.settings.get_params()
            for seg in self.speed_segments:
                if seg["start"] <= self.current_frame_idx <= seg["end"]:
                    type_str = {FRAME_TYPE_1X: "1x", FRAME_TYPE_2X: "2x",
                                FRAME_TYPE_0_2X: "0.2x"}.get(seg["type"], "未知")
                    speed_eff = 1
                    if seg["type"] == FRAME_TYPE_1X   and p['speedup_1x']:   speed_eff = 2
                    if seg["type"] == FRAME_TYPE_0_2X and p['speedup_02']:   speed_eff = p['speedup_02_factor']
                    info_text = (f"变速区域: {type_str}"
                                 + (f" (预览抽取 {speed_eff}x)" if speed_eff > 1 else ""))
                    break

        self.lbl_info.config(text=info_text)

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


# ============================================================
#  主程序
# ============================================================
def main():
    root = tk.Tk()
    root.title("明日方舟剪辑工具")
    root.geometry("1600x900")

    # 顶部：打开按钮
    top_bar = ttk.Frame(root)
    top_bar.pack(fill=tk.X, padx=10, pady=5)
    ttk.Label(top_bar, text="输入视频:").pack(side=tk.LEFT)
    input_var = tk.StringVar()
    ttk.Entry(top_bar, textvariable=input_var, width=50).pack(side=tk.LEFT, padx=5)

    # 右侧：设置面板
    right_panel = ttk.Frame(root)
    right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

    settings = SettingsPanel(right_panel)
    settings.pack(fill=tk.BOTH, expand=True)

    # 左侧：播放器
    left_panel = ttk.Frame(root)
    left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    player = VideoPreviewPlayer(left_panel, settings=settings, width=800, height=450)
    player.pack(fill=tk.BOTH, expand=True)

    # 绑定导出按钮
    settings.export_callback = player.export_video

    def open_file():
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if path:
            input_var.set(path)
            player.load_video(path)
            # 自动填充输出路径
            if not settings.output_var.get():
                name, _ = os.path.splitext(path)
                settings.output_var.set(f"{name}_clipped.mp4")

    ttk.Button(top_bar, text="打开视频", command=open_file).pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    main()