# settings_panel.py —— 独立设置面板，所有参数集中在此

import tkinter as tk
from tkinter import ttk, filedialog
import os
import subprocess


class SettingsPanel(ttk.LabelFrame):
    """包含全部参数设置的面板（与 VideoPreviewPlayer 解耦）"""

    def __init__(self, parent, **kw):
        super().__init__(parent, text="处理参数", padding=8, **kw)
        self.export_callback = None
        self.segment_export_callback = None
        self._build()

    # ----------------------------------------------------------
    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_basic  = ttk.Frame(nb, padding=6)
        tab_match  = ttk.Frame(nb, padding=6)
        tab_pause  = ttk.Frame(nb, padding=6)
        tab_export = ttk.Frame(nb, padding=6)
        nb.add(tab_basic,  text="基本")
        nb.add(tab_match,  text="匹配阈值")
        nb.add(tab_pause,  text="暂停处理")
        nb.add(tab_export, text="导出")

        # ---- 基本 ----
        rows_basic = [
            ("批处理大小:",   "batch_size_var",        tk.IntVar,    128,   1,    512,  1),
            ("处理宽度:",     "proc_w_var",            tk.IntVar,    400,   100,  1920, 1),
            ("处理高度:",     "proc_h_var",            tk.IntVar,    225,   100,  1080, 1),
            ("线程数:",       "thread_var",            tk.IntVar,
             max(1, os.cpu_count() or 4),              1, 64, 1),
        ]
        for r, (lbl, attr, vtype, default, mn, mx, step) in enumerate(rows_basic):
            var = vtype(value=default)
            setattr(self, attr, var)
            ttk.Label(tab_basic, text=lbl).grid(row=r, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(tab_basic, from_=mn, to=mx, increment=step,
                        textvariable=var, width=7).grid(row=r, column=1, sticky=tk.W, padx=4)

        sep_r = len(rows_basic)
        ttk.Separator(tab_basic, orient=tk.HORIZONTAL).grid(
            row=sep_r, column=0, columnspan=2, sticky=tk.EW, pady=4)

        self.speedup_1x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_basic, text="1x 区域以 2x 播放",
                        variable=self.speedup_1x_var).grid(
            row=sep_r+1, column=0, columnspan=2, sticky=tk.W)

        self.speedup_02x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_basic, text="0.2x 区域加速",
                        variable=self.speedup_02x_var).grid(
            row=sep_r+2, column=0, columnspan=2, sticky=tk.W)

        self.speedup_02x_factor_var = tk.IntVar(value=10)
        ttk.Label(tab_basic, text="0.2x 加速倍率:").grid(
            row=sep_r+3, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_basic, from_=2, to=20,
                    textvariable=self.speedup_02x_factor_var, width=7).grid(
            row=sep_r+3, column=1, sticky=tk.W, padx=4)

        ttk.Separator(tab_basic, orient=tk.HORIZONTAL).grid(
            row=sep_r+4, column=0, columnspan=2, sticky=tk.EW, pady=4)

        self.key_repeat_speed_var = tk.IntVar(value=30)
        ttk.Label(tab_basic, text="←→ 连续移动速度\n(帧/秒):").grid(
            row=sep_r+5, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_basic, from_=1, to=120,
                    textvariable=self.key_repeat_speed_var, width=7).grid(
            row=sep_r+5, column=1, sticky=tk.W, padx=4)

        # ---- 匹配阈值 ----
        self.thr_pause_var = tk.DoubleVar(value=0.7)
        self.thr_1x_var    = tk.DoubleVar(value=0.7)
        self.thr_2x_var    = tk.DoubleVar(value=0.7)
        self.thr_02x_var   = tk.DoubleVar(value=0.7)
        for i, (lbl, var) in enumerate([
            ("暂停阈值:",  self.thr_pause_var),
            ("1x 阈值:",   self.thr_1x_var),
            ("2x 阈值:",   self.thr_2x_var),
            ("0.2x 阈值:", self.thr_02x_var),
        ]):
            ttk.Label(tab_match, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(tab_match, from_=0.1, to=1.0, increment=0.01,
                        textvariable=var, width=8).grid(row=i, column=1, sticky=tk.W, padx=4)

        # ---- 暂停处理 ----
        r = 0
        self.compare_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_pause, text="启用画面对比 (防止跳帧)",
                        variable=self.compare_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2)
        r += 1

        self.dct_thresh_var = tk.DoubleVar(value=50000.0)
        ttk.Label(tab_pause, text="DCT 低频差异阈值:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=1000, to=1_000_000, increment=5000,
                    textvariable=self.dct_thresh_var, width=10).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.hist_thresh_var = tk.DoubleVar(value=0.95)
        ttk.Label(tab_pause, text="Hist 相似度阈值:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=0.1, to=1.0, increment=0.01,
                    textvariable=self.hist_thresh_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.keep_before_var = tk.IntVar(value=0)
        ttk.Label(tab_pause, text="保留前段 (帧):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=0, to=300,
                    textvariable=self.keep_before_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.keep_after_var = tk.IntVar(value=60)
        ttk.Label(tab_pause, text="保留后段 (帧):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=0, to=300,
                    textvariable=self.keep_after_var, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        ttk.Separator(tab_pause, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky=tk.EW, pady=6)
        r += 1

        ttk.Label(tab_pause, text="批量应用到所有暂停:").grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=(2, 4))
        r += 1

        btn_frame = ttk.Frame(tab_pause)
        btn_frame.grid(row=r, column=0, columnspan=2, sticky=tk.EW)
        # 三个按钮竖排
        self.apply_pause_btn_keep = ttk.Button(
            btn_frame, text="全部保留",
            command=lambda: self._on_apply_pause('keep'))
        self.apply_pause_btn_keep.pack(fill=tk.X, pady=2)
        self.apply_pause_btn_auto = ttk.Button(
            btn_frame, text="按设置裁剪",
            command=lambda: self._on_apply_pause('auto'))
        self.apply_pause_btn_auto.pack(fill=tk.X, pady=2)
        self.apply_pause_btn_all = ttk.Button(
            btn_frame, text="全部裁剪",
            command=lambda: self._on_apply_pause('all'))
        self.apply_pause_btn_all.pack(fill=tk.X, pady=2)

        # ---- 导出 ----
        r = 0
        tab_export.columnconfigure(0, weight=0)
        tab_export.columnconfigure(1, weight=1)   # Entry 拉伸
        self.output_var = tk.StringVar()
        ttk.Label(tab_export, text="输出路径:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(tab_export, textvariable=self.output_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4)
        ttk.Button(tab_export, text="浏览",
                   command=self._browse_output).grid(row=r, column=2)
        r += 1

        self.quality_var = tk.IntVar(value=6)
        ttk.Label(tab_export, text="视频质量 (0-10):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_export, from_=0, to=10,
                    textvariable=self.quality_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.export_btn = ttk.Button(tab_export, text="导出剪辑视频",
                                     command=self._on_export)
        self.export_btn.grid(row=r, column=0, columnspan=3, pady=8)
        r += 1

        self.export_progress_var = tk.DoubleVar()
        ttk.Progressbar(tab_export, variable=self.export_progress_var,
                        maximum=100).grid(row=r, column=0, columnspan=3,
                                          sticky=tk.EW, pady=2)
        r += 1

        self.export_status_var = tk.StringVar(value="就绪")
        ttk.Label(tab_export, textvariable=self.export_status_var).grid(
            row=r, column=0, columnspan=3)
        r += 1

        ttk.Separator(tab_export, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky=tk.EW, pady=8)
        r += 1

        seg_frame = ttk.LabelFrame(tab_export, text="分段导出", padding=6)
        seg_frame.grid(row=r, column=0, columnspan=3, sticky=tk.EW)
        seg_frame.columnconfigure(0, weight=1)
        r += 1



        tip_text = ("仅导出时间轴的有效部分\n被裁剪掉的区间不会导出）\n"
                    "若不想导出暂停片段\n请先到“暂停处理”页点击“全部裁剪”")
        ttk.Label(seg_frame, text=tip_text, foreground="#666666",
                  justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))

        self.segment_split_by_speed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            seg_frame,
            text="按变速类型进一步区分\n（pause/2x/1x/0.2x/other）",
            variable=self.segment_split_by_speed_var).grid(
            row=1, column=0, sticky=tk.W, pady=2)

        self.segment_export_btn = ttk.Button(
            seg_frame, text="导出分段视频", command=self._on_segment_export)
        self.segment_export_btn.grid(row=2, column=0, sticky=tk.W, pady=(4, 2))

        self.segment_export_progress_var = tk.DoubleVar()
        ttk.Progressbar(seg_frame, variable=self.segment_export_progress_var,
                        maximum=100).grid(row=3, column=0, sticky=tk.EW, pady=2)

        self.segment_export_status_var = tk.StringVar(value="就绪")
        ttk.Label(seg_frame, textvariable=self.segment_export_status_var).grid(
            row=4, column=0, sticky=tk.W, pady=(2, 0))

        self.export_use_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab_export, text="启用 GPU 导出加速(FFmpeg)",
            variable=self.export_use_gpu_var).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
        r += 1

        self.gpu_encoder_hint = tk.StringVar(value="")
        ttk.Label(tab_export, textvariable=self.gpu_encoder_hint,
                  foreground="#666666", justify=tk.LEFT).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        r += 1

        self._detect_gpu_encoder()

        #audio_tip = "提示：分段导出默认不保留音频\n避免音画错位/拖尾"
        #ttk.Label(seg_frame, text=audio_tip, foreground="#666666").grid(
        #    row=5, column=0, sticky=tk.W, pady=(4, 0))

    # ----------------------------------------------------------
    def _browse_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("所有", "*.*")])
        if p:
            self.output_var.set(p)

    def _on_export(self):
        if self.export_callback:
            self.export_callback()

    def _on_segment_export(self):
        if self.segment_export_callback:
            self.segment_export_callback()

    def _on_apply_pause(self, mode: str):
        """触发播放器的批量暂停模式应用"""
        if hasattr(self, 'apply_pause_callback') and self.apply_pause_callback:
            self.apply_pause_callback(mode)

    def _detect_gpu_encoder(self):
        """检测可用的 GPU 编码器并设置导出默认值。"""
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-encoders"],
                text=True, stderr=subprocess.STDOUT)
            candidates = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox")
            gpu_encoder = next((enc for enc in candidates if enc in out), None)
            if gpu_encoder:
                self.gpu_encoder_hint.set(f"检测到可用的GPU编码器:\n {gpu_encoder}")
                self.export_use_gpu_var.set(True)
            else:
                self.gpu_encoder_hint.set("未检测到GPU编码器，将使用CPU编码")
                self.export_use_gpu_var.set(False)
        except Exception as e:
            self.gpu_encoder_hint.set("未找到FFmpeg，请确认已安装并添加到PATH")
            self.export_use_gpu_var.set(False)
            print(f"GPU编码器检测失败: {e}")

    def get_params(self) -> dict:
        """一次性读取所有参数，返回纯 Python 字典（无 tkinter 依赖）"""
        return {
            'batch':   self.batch_size_var.get(),
            'proc_res': (self.proc_w_var.get(), self.proc_h_var.get()),
            'threads': self.thread_var.get(),
            'speedup_1x':  self.speedup_1x_var.get(),
            'speedup_02':  self.speedup_02x_var.get(),
            'speedup_02_factor': self.speedup_02x_factor_var.get(),
            'key_repeat_speed':  self.key_repeat_speed_var.get(),
            'thresholds': {
                'pause':      self.thr_pause_var.get(),
                'speed_1x':   self.thr_1x_var.get(),
                'speed_2x':   self.thr_2x_var.get(),
                'speed_0_2x': self.thr_02x_var.get(),
            },
            'compare': {
                'enabled': self.compare_var.get(),
                'dct':     self.dct_thresh_var.get(),
                'hist':    self.hist_thresh_var.get(),
                'keep_n':  self.keep_before_var.get(),
                'keep_m':  self.keep_after_var.get(),
            },
            'output':  self.output_var.get(),
            'quality': self.quality_var.get(),
            'export_use_gpu': self.export_use_gpu_var.get(),
        }