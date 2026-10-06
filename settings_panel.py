# settings_panel.py —— 独立设置面板，所有参数集中在此

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from queue import Empty, Queue

import gpu_caps

import matcher
import pipeline
import app_core


class SettingsPanel(ttk.LabelFrame):
    """包含全部参数设置的面板（与 VideoPreviewPlayer 解耦）"""

    def __init__(self, parent, **kw):
        super().__init__(parent, text="处理参数", padding=8, **kw)
        self.export_callback = None
        self.segment_export_callback = None
        self.autotune_callback = None
        self.cancel_export_callback = None
        self.cancel_segment_export_callback = None
        self.selected_pause_id = None

        self.profile = None
        self._decode_choices: dict[str, str] = {}
        self._match_choices: dict[str, str] = {}
        self._probe_queue: Queue = Queue(maxsize=4)

        self._build()
        app_core.register(self._on_status_event)
        self.after(300, lambda: self._detect_gpu_encoder())
        self.after(1500, self.maybe_ask_gpu_pack)

    # ----------------------------------------------------------
    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.notebook = nb

        tab_basic = ttk.Frame(nb, padding=6)
        tab_perf = ttk.Frame(nb, padding=6)
        tab_match = ttk.Frame(nb, padding=6)
        tab_pause = ttk.Frame(nb, padding=6)
        tab_export = ttk.Frame(nb, padding=6)
        nb.add(tab_basic, text="基本")
        nb.add(tab_perf, text="性能/GPU")
        nb.add(tab_match, text="匹配阈值")
        nb.add(tab_pause, text="暂停处理")
        nb.add(tab_export, text="导出")

        self._build_basic(tab_basic)
        self._build_perf(tab_perf)
        self._build_match(tab_match)
        self._build_pause(tab_pause)
        self._build_export(tab_export)

    # ---- 基本 ----
    def _build_basic(self, tab_basic):
        rows_basic = [
            ("批处理大小:", "batch_size_var", tk.IntVar, 128, 1, 512, 1),
            ("处理宽度:", "proc_w_var", tk.IntVar, 400, 100, 1920, 1),
            ("处理高度:", "proc_h_var", tk.IntVar, 225, 100, 1080, 1),
            ("线程数:", "thread_var", tk.IntVar, max(1, os.cpu_count() or 4), 1, 64, 1),
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
        ttk.Checkbutton(tab_basic, text="1x 区域以 2x 播放", variable=self.speedup_1x_var).grid(
            row=sep_r + 1, column=0, columnspan=2, sticky=tk.W)

        self.speedup_02x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_basic, text="0.2x 区域加速", variable=self.speedup_02x_var).grid(
            row=sep_r + 2, column=0, columnspan=2, sticky=tk.W)

        self.speedup_02x_factor_var = tk.IntVar(value=10)
        ttk.Label(tab_basic, text="0.2x 加速倍率:").grid(row=sep_r + 3, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_basic, from_=2, to=20, textvariable=self.speedup_02x_factor_var,
                    width=7).grid(row=sep_r + 3, column=1, sticky=tk.W, padx=4)

        ttk.Separator(tab_basic, orient=tk.HORIZONTAL).grid(
            row=sep_r + 4, column=0, columnspan=2, sticky=tk.EW, pady=4)

        self.ffmpeg_path_var = tk.StringVar(value="auto")
        ttk.Label(tab_basic, text="FFmpeg 路径:").grid(row=sep_r + 5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(tab_basic, textvariable=self.ffmpeg_path_var, width=18).grid(
            row=sep_r + 5, column=1, sticky=tk.W, padx=4)
        ttk.Label(
            tab_basic,
            text="auto=程序目录/PATH/imageio_ffmpeg；分析与导出/GPU 探测共用。\n"
                 "硬件编解码需要完整构建（Gyan/BtbN 的 full/gpl 版）。",
            foreground="#666666", justify=tk.LEFT,
        ).grid(row=sep_r + 6, column=0, columnspan=2, sticky=tk.W)

        ttk.Separator(tab_basic, orient=tk.HORIZONTAL).grid(
            row=sep_r + 7, column=0, columnspan=2, sticky=tk.EW, pady=4)

        self.key_repeat_speed_var = tk.IntVar(value=30)
        ttk.Label(tab_basic, text="←→ 连续移动速度\n(帧/秒):").grid(
            row=sep_r + 8, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_basic, from_=1, to=120, textvariable=self.key_repeat_speed_var,
                    width=7).grid(row=sep_r + 8, column=1, sticky=tk.W, padx=4)

    # ---- 性能 / GPU ----
    def _build_perf(self, tab_perf):
        tab_perf.columnconfigure(0, weight=1)

        cap_frame = ttk.LabelFrame(tab_perf, text="硬件能力（实测，不是查表）", padding=6)
        cap_frame.grid(row=0, column=0, sticky=tk.EW)
        cap_frame.columnconfigure(0, weight=1)

        self.profile_text = tk.StringVar(value="尚未探测…")
        tk.Label(cap_frame, textvariable=self.profile_text, justify=tk.LEFT, anchor="w",
                 font=("Consolas", 8), fg="#0A7").grid(row=0, column=0, sticky=tk.EW)

        self.probe_btn = ttk.Button(cap_frame, text="重新探测硬件",
                                    command=lambda: self._detect_gpu_encoder(force=True))
        self.probe_btn.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))

        dec_frame = ttk.LabelFrame(tab_perf, text="解码与匹配", padding=6)
        dec_frame.grid(row=1, column=0, sticky=tk.EW, pady=(6, 0))
        dec_frame.columnconfigure(1, weight=1)

        ttk.Label(dec_frame, text="解码加速:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.decode_backend_var = tk.StringVar(value="自动（硬件优先）")
        self.decode_backend_combo = ttk.Combobox(
            dec_frame, textvariable=self.decode_backend_var, state="readonly", width=26,
            values=("自动（硬件优先）", "不加速(OpenCV)", "FFmpeg 软件(A_PT)"))
        self.decode_backend_combo.grid(row=0, column=1, sticky=tk.W, padx=4)
        self.decode_backend_combo.current(0)

        ttk.Label(dec_frame, text="匹配计算:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.match_backend_var = tk.StringVar(value="自动测速")
        self.match_backend_combo = ttk.Combobox(
            dec_frame, textvariable=self.match_backend_var, state="readonly", width=26,
            values=("自动测速",))
        self.match_backend_combo.grid(row=1, column=1, sticky=tk.W, padx=4)
        self.match_backend_combo.current(0)

        self.skip_identical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dec_frame, text="静止帧跳过（完全相同帧复用上一帧判定，结果等价）",
                        variable=self.skip_identical_var).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.match_autotune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dec_frame, text="分析前自动测速选最快后端",
                        variable=self.match_autotune_var).grid(
            row=3, column=0, columnspan=2, sticky=tk.W)

        self.autotune_btn = ttk.Button(dec_frame, text="重新测速",
                                       command=self._on_autotune)
        self.autotune_btn.grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
        self.bench_text = tk.StringVar(value="尚未测速")
        tk.Label(dec_frame, textvariable=self.bench_text, justify=tk.LEFT, anchor="w",
                 wraplength=320, font=("Consolas", 8), fg="#555").grid(
            row=5, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))

        pack_frame = ttk.LabelFrame(tab_perf, text="CUDA 匹配加速包（可选）", padding=6)
        pack_frame.grid(row=2, column=0, sticky=tk.EW, pady=(6, 0))
        pack_frame.columnconfigure(0, weight=1)

        self.pack_text = tk.StringVar(value=gpu_caps.pack_status_text())
        tk.Label(pack_frame, textvariable=self.pack_text, justify=tk.LEFT, anchor="w",
                 wraplength=320, font=("Consolas", 8), fg="#555").grid(
            row=0, column=0, columnspan=2, sticky=tk.EW)

        self.pack_btn = ttk.Button(pack_frame, text="下载并启用（约 2.5GB）",
                                   command=self._on_install_pack)
        self.pack_btn.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))

        self.pack_noask_var = tk.BooleanVar(value=gpu_caps.is_declined())
        ttk.Checkbutton(pack_frame, text="不再询问", variable=self.pack_noask_var,
                        command=self._on_pack_noask).grid(row=1, column=1, sticky=tk.W, padx=6)

        self.pack_progress = tk.DoubleVar()
        ttk.Progressbar(pack_frame, variable=self.pack_progress,
                        maximum=100).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)

        evt_frame = ttk.LabelFrame(tab_perf, text="运行事件（降级/回退原因都会在这里）", padding=6)
        evt_frame.grid(row=3, column=0, sticky=tk.EW, pady=(6, 0))
        evt_frame.columnconfigure(0, weight=1)

        self.event_text = tk.Text(evt_frame, height=7, width=44, wrap="word",
                                  font=("Consolas", 8), state=tk.DISABLED,
                                  background="#111", foreground="#CCC")
        self.event_text.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(evt_frame, text="查看完整日志",
                   command=self._show_log).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self._refresh_events()

    # ---- 匹配阈值 ----
    def _build_match(self, tab_match):
        self.thr_pause_var = tk.DoubleVar(value=0.7)
        self.thr_1x_var = tk.DoubleVar(value=0.7)
        self.thr_2x_var = tk.DoubleVar(value=0.7)
        self.thr_02x_var = tk.DoubleVar(value=0.7)
        for i, (lbl, var) in enumerate([
            ("暂停阈值:", self.thr_pause_var),
            ("1x 阈值:", self.thr_1x_var),
            ("2x 阈值:", self.thr_2x_var),
            ("0.2x 阈值:", self.thr_02x_var),
        ]):
            ttk.Label(tab_match, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(tab_match, from_=0.1, to=1.0, increment=0.01,
                        textvariable=var, width=8).grid(row=i, column=1, sticky=tk.W, padx=4)

    # ---- 暂停处理 ----
    def _build_pause(self, tab_pause):
        r = 0
        lbl_frame_sel = ttk.LabelFrame(tab_pause, text="当前选中暂停片段 (时间轴左键选中)", padding=4)
        lbl_frame_sel.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=2)
        r += 1

        self.lbl_selected_pause = ttk.Label(lbl_frame_sel, text="未选中任何暂停片段")
        self.lbl_selected_pause.grid(row=0, column=0, columnspan=3, pady=2)

        btn_keep_sel = ttk.Button(lbl_frame_sel, text="全部保留",
                                  command=lambda: self._on_single_pause('keep'))
        btn_keep_sel.grid(row=1, column=0, padx=2, pady=2)
        btn_auto_sel = ttk.Button(lbl_frame_sel, text="按设置裁剪",
                                  command=lambda: self._on_single_pause('auto'))
        btn_auto_sel.grid(row=1, column=1, padx=2, pady=2)
        btn_all_sel = ttk.Button(lbl_frame_sel, text="全部裁剪",
                                 command=lambda: self._on_single_pause('all'))
        btn_all_sel.grid(row=1, column=2, padx=2, pady=2)

        self.btn_keep_sel = btn_keep_sel
        self.btn_auto_sel = btn_auto_sel
        self.btn_all_sel = btn_all_sel
        self._update_single_buttons(False)

        self.still_time_thresh_var = tk.DoubleVar(value=0.1)
        ttk.Label(tab_pause, text="静止无动作缓冲时长(秒):").grid(
            row=r, column=0, sticky=tk.W, pady=(8, 2))
        ttk.Spinbox(tab_pause, from_=0.01, to=2.0, increment=0.01,
                    textvariable=self.still_time_thresh_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4, pady=(8, 2))
        r += 1

        self.motion_thresh_var = tk.DoubleVar(value=2.0)
        ttk.Label(tab_pause, text="动作检测灵敏度(差异阈值):").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=0.1, to=50.0, increment=0.5,
                    textvariable=self.motion_thresh_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.boundary_thresh_var = tk.DoubleVar(value=5.0)
        ttk.Label(tab_pause, text="无操作阈值(前后差异<该值全删):").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_pause, from_=0.0, to=50.0, increment=0.5,
                    textvariable=self.boundary_thresh_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        ttk.Separator(tab_pause, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky=tk.EW, pady=6)
        r += 1

        lbl_frame_batch = ttk.LabelFrame(tab_pause, text="批量应用 (修改所有暂停片段)", padding=4)
        lbl_frame_batch.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=2)
        r += 1

        self.apply_pause_btn_keep = ttk.Button(
            lbl_frame_batch, text="全部保留", command=lambda: self._on_apply_pause('keep'))
        self.apply_pause_btn_keep.pack(fill=tk.X, pady=2)
        self.apply_pause_btn_auto = ttk.Button(
            lbl_frame_batch, text="全部按设置裁剪 (Auto)", command=lambda: self._on_apply_pause('auto'))
        self.apply_pause_btn_auto.pack(fill=tk.X, pady=2)
        self.apply_pause_btn_all = ttk.Button(
            lbl_frame_batch, text="全部裁剪", command=lambda: self._on_apply_pause('all'))
        self.apply_pause_btn_all.pack(fill=tk.X, pady=2)

    # ---- 导出 ----
    def _build_export(self, tab_export):
        r = 0
        tab_export.columnconfigure(0, weight=0)
        tab_export.columnconfigure(1, weight=1)
        self.output_var = tk.StringVar()
        ttk.Label(tab_export, text="输出路径:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(tab_export, textvariable=self.output_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4)
        ttk.Button(tab_export, text="浏览", command=self._browse_output).grid(row=r, column=2)
        r += 1

        self.quality_var = tk.IntVar(value=6)
        ttk.Label(tab_export, text="视频质量 (0-10):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_export, from_=0, to=10, textvariable=self.quality_var, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        # ---- 编码器设置（全部读实测结果） ----
        ttk.Label(tab_export, text="编码器:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.gpu_encoder_var = tk.StringVar()
        self.gpu_encoder_combo = ttk.Combobox(
            tab_export, textvariable=self.gpu_encoder_var, state="readonly", width=20)
        self.gpu_encoder_combo.grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=4)
        self.gpu_encoder_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_presets())
        r += 1

        ttk.Label(tab_export, text="编码速度预设:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.export_preset_var = tk.StringVar()
        self.export_preset_combo = ttk.Combobox(
            tab_export, textvariable=self.export_preset_var, state="readonly", width=20)
        self.export_preset_combo.grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=4)
        r += 1

        ttk.Label(tab_export, text="导出并发数:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.export_workers_var = tk.IntVar(value=max(1, (os.cpu_count() or 2) // 2))
        ttk.Spinbox(tab_export, from_=1, to=8, textvariable=self.export_workers_var,
                    width=8).grid(row=r, column=1, sticky=tk.W, padx=4)
        r += 1

        self.gpu_encoder_hint = tk.StringVar(value="正在实际测试编码器…")
        tk.Label(tab_export, textvariable=self.gpu_encoder_hint, foreground="#666666",
                 justify=tk.LEFT, wraplength=320).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        r += 1

        self.export_use_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab_export, text="启用 GPU 导出加速(FFmpeg)",
                        variable=self.export_use_gpu_var).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=(2, 2))
        r += 1

        self.keyframe_copy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_export, text="极速无损模式（关键帧对齐时自动启用）",
                        variable=self.keyframe_copy_var).grid(
            row=r, column=0, columnspan=3, sticky=tk.W)
        r += 1

        self.export_btn = ttk.Button(tab_export, text="导出整段剪辑视频", command=self._on_export)
        self.export_btn.grid(row=r, column=0, columnspan=2, pady=8, sticky=tk.W)
        self.cancel_export_btn = ttk.Button(tab_export, text="取消导出",
                                            command=self._on_cancel_export, state=tk.DISABLED)
        self.cancel_export_btn.grid(row=r, column=2, pady=8, sticky=tk.W)
        r += 1

        self.export_progress_var = tk.DoubleVar()
        ttk.Progressbar(tab_export, variable=self.export_progress_var,
                        maximum=100).grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=2)
        r += 1

        self.export_status_var = tk.StringVar(value="就绪")
        ttk.Label(tab_export, textvariable=self.export_status_var, wraplength=320,
                  justify=tk.LEFT).grid(row=r, column=0, columnspan=3, sticky=tk.W)
        r += 1

        ttk.Separator(tab_export, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky=tk.EW, pady=8)
        r += 1

        seg_frame = ttk.LabelFrame(tab_export, text="碎片化分段独立导出 (支持多线程极速秒切)", padding=6)
        seg_frame.grid(row=r, column=0, columnspan=3, sticky=tk.EW)
        seg_frame.columnconfigure(0, weight=1)
        r += 1

        ttk.Label(seg_frame, text="不合并，仅将时间轴上留下的每一个断档部分单独保存",
                  foreground="#666666", justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))

        self.segment_split_by_speed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(seg_frame, text="按变速类型进一步区分碎片",
                        variable=self.segment_split_by_speed_var).grid(row=1, column=0, sticky=tk.W, pady=2)

        self.merge_pause_ops_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(seg_frame, text="合并同一暂停区内的所有零碎操作 (推荐)",
                        variable=self.merge_pause_ops_var).grid(row=2, column=0, sticky=tk.W, pady=2)

        seg_btn_row = ttk.Frame(seg_frame)
        seg_btn_row.grid(row=3, column=0, sticky=tk.W, pady=(4, 2))
        self.segment_export_btn = ttk.Button(seg_btn_row, text="一键导出所有分段碎片",
                                             command=self._on_segment_export)
        self.segment_export_btn.pack(side=tk.LEFT)
        self.cancel_segment_btn = ttk.Button(seg_btn_row, text="取消",
                                             command=self._on_cancel_segment_export,
                                             state=tk.DISABLED)
        self.cancel_segment_btn.pack(side=tk.LEFT, padx=6)

        self.segment_export_progress_var = tk.DoubleVar()
        ttk.Progressbar(seg_frame, variable=self.segment_export_progress_var,
                        maximum=100).grid(row=4, column=0, sticky=tk.EW, pady=2)

        self.segment_export_status_var = tk.StringVar(value="就绪")
        ttk.Label(seg_frame, textvariable=self.segment_export_status_var, wraplength=320,
                  justify=tk.LEFT).grid(row=5, column=0, sticky=tk.W, pady=(2, 0))

        # 探测完成前的合理默认：预设按 libx264 给一组，探测回来后会按实际编码器刷新
        self._refresh_presets()

    # ----------------------------------------------------------
    #  运行事件区
    # ----------------------------------------------------------
    def _on_status_event(self, event: dict) -> None:
        # 事件可能来自任意线程：必须切回 Tk 主线程
        try:
            self.after(0, lambda e=event: self._append_event(e))
        except Exception:
            pass

    def _append_event(self, event: dict) -> None:
        try:
            if not self.event_text.winfo_exists():
                return
            self.event_text.config(state=tk.NORMAL)
            self.event_text.insert(tk.END, app_core.format_event(event) + "\n")
            # 只保留最近 200 行
            lines = int(self.event_text.index("end-1c").split(".")[0])
            if lines > 200:
                self.event_text.delete("1.0", f"{lines - 200}.0")
            self.event_text.see(tk.END)
            self.event_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def _refresh_events(self) -> None:
        try:
            self.event_text.config(state=tk.NORMAL)
            self.event_text.delete("1.0", tk.END)
            for e in app_core.last_events(40):
                self.event_text.insert(tk.END, app_core.format_event(e) + "\n")
            self.event_text.see(tk.END)
            self.event_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def _show_log(self) -> None:
        win = tk.Toplevel(self)
        win.title("运行日志")
        win.geometry("720x420")
        text = tk.Text(win, wrap="word", font=("Consolas", 9))
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", app_core.history_text(300))
        text.config(state=tk.DISABLED)
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=4)

    # ----------------------------------------------------------
    #  硬件探测 / 测速 / 加速包
    # ----------------------------------------------------------
    def _ffmpeg_path(self):
        raw = (self.ffmpeg_path_var.get() or "auto").strip()
        return None if raw.lower() in ("", "auto") else raw

    def _detect_gpu_encoder(self, force: bool = False) -> None:
        self.probe_btn.config(state=tk.DISABLED, text="探测中…")
        self.profile_text.set("正在实测硬件能力（编码器/解码路径）…")
        result_queue: Queue = Queue(maxsize=1)
        ffmpeg_path = self._ffmpeg_path()

        def worker():
            try:
                prof = gpu_caps.detect(ffmpeg_path, force=force)
                result_queue.put((prof, None))
            except Exception as exc:
                result_queue.put((None, exc))

        def poll():
            try:
                prof, err = result_queue.get_nowait()
            except Empty:
                self.after(120, poll)
                return
            self._apply_profile(prof, err)

        threading.Thread(target=worker, daemon=True).start()
        self.after(120, poll)

    def _apply_profile(self, profile, err) -> None:
        self.probe_btn.config(state=tk.NORMAL, text="重新探测硬件")
        if err is not None or profile is None:
            self.profile_text.set(f"探测失败：{err}")
            return
        self.profile = profile
        self.profile_text.set("\n".join(gpu_caps.describe(profile)))

        # 解码下拉：自动 / 不加速 / 软件 + 实测通过的硬件变体
        choices = {"自动（硬件优先）": pipeline.DECODE_BACKEND_AUTO,
                   "不加速(OpenCV)": pipeline.DECODE_BACKEND_OPENCV,
                   "FFmpeg 软件(A_PT)": pipeline.DECODE_BACKEND_FFMPEG_SW}
        for v in profile.decode_variants:
            if v.get("ok"):
                choices[f"硬件 {v['label']}"] = v["key"]
        self._decode_choices = choices
        self.decode_backend_combo["values"] = list(choices.keys())
        current_key = self._decode_choices.get(self.decode_backend_var.get())
        if current_key is None:
            self.decode_backend_combo.current(0)

        # 匹配下拉：可用在前，不可用标注在后
        avail = matcher.available_backends(profile)
        all_backends = [matcher.BACKEND_TORCH_CUDA, matcher.BACKEND_TORCH_DML,
                        matcher.BACKEND_OPENCL, matcher.BACKEND_CPU_BATCH,
                        matcher.BACKEND_CV2_DIRECT, matcher.BACKEND_CPU_POOL]
        match_choices = {"自动测速": "auto"}
        for b in avail:
            match_choices[matcher.backend_label(b)] = b
        for b in all_backends:
            if b not in avail:
                match_choices[f"{matcher.backend_label(b)}（不可用）"] = b
        self._match_choices = match_choices
        self.match_backend_combo["values"] = list(match_choices.keys())
        if self.match_backend_var.get() not in match_choices:
            self.match_backend_combo.current(0)

        # 编码器下拉
        working = list(profile.encoders_working)
        listed = [e for e in profile.encoders_listed if e not in working]
        values = ["自动"] + working + listed + ["libx264", "libx265"]
        self.gpu_encoder_combo["values"] = values
        if self.gpu_encoder_var.get() not in values:
            self.gpu_encoder_var.set("自动")
        self.export_workers_var.set(max(1, int(profile.export_workers or 1)))

        if working:
            hint = "实测可用: " + ", ".join(working)
            if listed:
                hint += "\n未通过测试: " + ", ".join(listed) + "（手动选择会回退 CPU）"
            self.export_use_gpu_var.set(True)
        else:
            hint = ("没有实测可用的硬件编码器，将使用 libx264"
                    if listed else "FFmpeg 未提供硬件编码器，将使用 libx264")
            self.export_use_gpu_var.set(False)
        self.gpu_encoder_hint.set(hint)
        self._refresh_presets()
        self.pack_text.set(gpu_caps.pack_status_text())

    def _refresh_presets(self) -> None:
        raw = (self.gpu_encoder_var.get() or "自动").strip()
        if raw in ("", "自动"):
            enc = (self.profile.encoder if self.profile else "") or "libx264"
        else:
            enc = raw
        opts = gpu_caps.preset_options(enc)
        self.export_preset_combo["values"] = opts
        if opts and self.export_preset_var.get() not in opts:
            self.export_preset_var.set(gpu_caps.default_preset(enc) or opts[0])
        elif not opts:
            self.export_preset_var.set("")

    def set_bench_text(self, text: str) -> None:
        self.bench_text.set(text)

    def _on_autotune(self) -> None:
        if not self.autotune_callback:
            self.bench_text.set("尚未加载视频，无法测速")
            return
        self.autotune_btn.config(state=tk.DISABLED, text="测速中…")
        self.bench_text.set("正在实测各匹配后端…")

        def done(_text):
            self.autotune_btn.config(state=tk.NORMAL, text="重新测速")

        self.autotune_callback(True, done)

    def _on_pack_noask(self) -> None:
        gpu_caps.set_declined(self.pack_noask_var.get())

    def maybe_ask_gpu_pack(self) -> None:
        """N 卡但没装 CUDA 加速包时问一次（可在界面上勾选不再询问）。"""
        try:
            if not gpu_caps.should_ask():
                return
        except Exception:
            return
        if messagebox.askyesno(
                "可选的 CUDA 匹配加速",
                "检测到 NVIDIA 显卡，但当前没有可用的 CUDA 匹配加速。\n\n"
                "下载并启用加速包（约 2.5GB）可以让模板匹配跑在显卡上，\n"
                "初步实测可把匹配耗时再降低数倍。\n\n"
                "现在下载并启用吗？（也可以稍后在「性能/GPU」页操作）"):
            self._on_install_pack()
        else:
            # 选「否」就记住，不再每次启动都问（本页仍可手动启用或取消勾选）
            gpu_caps.set_declined(True)
            self.pack_noask_var.set(True)

    def _on_install_pack(self) -> None:
        self.pack_btn.config(state=tk.DISABLED, text="下载中…")
        self.pack_progress.set(0)

        def progress(frac, msg):
            self.after(0, lambda f=frac, m=msg: (
                self.pack_progress.set(max(0.0, min(1.0, f)) * 100),
                self.pack_text.set(f"{m} {max(0.0, min(1.0, f)) * 100:.0f}%")))

        def worker():
            ok, msg = gpu_caps.install(progress_cb=progress)

            def done():
                self.pack_btn.config(state=tk.NORMAL, text="下载并启用（约 2.5GB）")
                self.pack_text.set(msg)
                if ok:
                    messagebox.showinfo("CUDA 匹配加速已启用", msg + "\n\n建议在「性能/GPU」页点一次「重新测速」。")
                    self._detect_gpu_encoder(force=True)
                else:
                    messagebox.showwarning("启用失败", msg)
            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------------------------------------
    #  选中暂停片段
    # ----------------------------------------------------------
    def _update_single_buttons(self, enabled=True, mode_str=""):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_keep_sel.config(state=state)
        self.btn_auto_sel.config(state=state)
        self.btn_all_sel.config(state=state)
        if enabled:
            name_map = {'keep': '全保留', 'auto': '按设置裁剪', 'all': '全删'}
            self.lbl_selected_pause.config(
                text=f"ID: {self.selected_pause_id}  |  状态: {name_map.get(mode_str, mode_str)}")
        else:
            self.lbl_selected_pause.config(text="未选中任何暂停片段")

    def set_selected_pause(self, seg_id, mode_str):
        self.selected_pause_id = seg_id
        self._update_single_buttons(seg_id is not None, mode_str)

    def _on_single_pause(self, mode: str):
        if self.selected_pause_id is not None and self.single_pause_callback:
            self.single_pause_callback(self.selected_pause_id, mode)

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

    def _on_cancel_export(self):
        if self.cancel_export_callback:
            self.cancel_export_callback()

    def _on_cancel_segment_export(self):
        if self.cancel_segment_export_callback:
            self.cancel_segment_export_callback()

    def set_export_running(self, running: bool) -> None:
        self.export_btn.config(state=tk.DISABLED if running else tk.NORMAL)
        self.cancel_export_btn.config(state=tk.NORMAL if running else tk.DISABLED)

    def set_segment_export_running(self, running: bool) -> None:
        self.segment_export_btn.config(state=tk.DISABLED if running else tk.NORMAL)
        self.cancel_segment_btn.config(state=tk.NORMAL if running else tk.DISABLED)

    def _on_apply_pause(self, mode: str):
        if self.apply_pause_callback:
            self.apply_pause_callback(mode)

    # 兼容旧名字：外部（main/preview）可能仍在用
    single_pause_callback = None
    apply_pause_callback = None

    # ----------------------------------------------------------
    #  参数汇总
    # ----------------------------------------------------------
    def _decode_backend_key(self) -> str:
        label = (self.decode_backend_var.get() or "").strip()
        if label in self._decode_choices:
            return self._decode_choices[label]
        # 探测尚未完成时的兜底
        if "A_PT" in label or "软件" in label:
            return pipeline.DECODE_BACKEND_FFMPEG_SW
        if "自动" in label:
            return pipeline.DECODE_BACKEND_AUTO
        return pipeline.DECODE_BACKEND_OPENCV

    def _match_backend_key(self) -> str:
        label = (self.match_backend_var.get() or "").strip()
        if label in self._match_choices:
            return self._match_choices[label]
        return "auto"

    def get_params(self) -> dict:
        return {
            'batch': self.batch_size_var.get(),
            'proc_res': (self.proc_w_var.get(), self.proc_h_var.get()),
            'threads': self.thread_var.get(),
            'speedup_1x': self.speedup_1x_var.get(),
            'speedup_02': self.speedup_02x_var.get(),
            'speedup_02_factor': self.speedup_02x_factor_var.get(),
            'key_repeat_speed': self.key_repeat_speed_var.get(),
            'decode_backend': self._decode_backend_key(),
            'match_backend': self._match_backend_key(),
            'match_autotune': self.match_autotune_var.get(),
            'skip_identical': self.skip_identical_var.get(),
            'ffmpeg_path': self._ffmpeg_path(),
            'thresholds': {
                'pause': self.thr_pause_var.get(),
                'speed_1x': self.thr_1x_var.get(),
                'speed_2x': self.thr_2x_var.get(),
                'speed_0_2x': self.thr_02x_var.get(),
            },
            'compare': {
                'still_time_thresh': self.still_time_thresh_var.get(),
                'motion_thresh': self.motion_thresh_var.get(),
                'boundary_thresh': self.boundary_thresh_var.get(),
            },
            'output': self.output_var.get(),
            'quality': self.quality_var.get(),
            'export_use_gpu': self.export_use_gpu_var.get(),
            'gpu_encoder': ("" if (self.gpu_encoder_var.get() or "").strip() == "自动"
                            else self.gpu_encoder_var.get()),
            'export_preset': (self.export_preset_var.get() or "").strip(),
            'export_workers': self.export_workers_var.get(),
            'keyframe_copy': self.keyframe_copy_var.get(),
            'merge_pause_ops': self.merge_pause_ops_var.get(),
        }
