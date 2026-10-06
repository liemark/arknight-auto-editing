# main.py —— 程序入口（PanedWindow 实现可拖动左右分隔）

import tkinter as tk
from tkinter import ttk, filedialog
import os
import multiprocessing   # ProcessPoolExecutor 需要在入口处 freeze_support

from settings_panel import SettingsPanel
from preview_player import VideoPreviewPlayer


<<<<<<< Updated upstream
=======
def _run_check() -> int:
    """无界面自检：打包后的 exe 用它验证依赖/模板/ffmpeg 是否都带全了。

    windowed exe 没有控制台，所以结果写进程序目录的 check-report.txt 并以退出码表示成败。
    """
    import traceback
    lines = []
    ok = True
    report = os.path.join(
        os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False)
                                       else __file__)), "check-report.txt")

    def flush():
        # 每步落盘：原生 abort（如 DLL 冲突）不会给 traceback，只能靠已写出的内容定位
        try:
            with open(report, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def add(name, good, detail=""):
        nonlocal ok
        ok = ok and bool(good)
        lines.append(f"[{'PASS' if good else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        flush()

    def note(text):
        lines.append(text)
        flush()

    add("Python", True, sys.version.split()[0])
    # frozen 只作信息：源码运行时为 False 属于正常，不算失败
    note(f"[INFO] frozen={bool(getattr(sys, 'frozen', False))} "
         f"meipass={getattr(sys, '_MEIPASS', '')}")

    try:
        import numpy
        add("numpy", True, numpy.__version__)
    except Exception as exc:
        add("numpy", False, repr(exc))
    try:
        import cv2
        add("cv2", True, f"{cv2.__version__} opencl={cv2.ocl.haveOpenCL()}")
    except Exception as exc:
        add("cv2", False, repr(exc))
    try:
        import PIL.ImageTk  # noqa: F401
        add("Pillow/ImageTk", True)
    except Exception as exc:
        add("Pillow/ImageTk", False, repr(exc))
    try:
        import imageio  # noqa: F401
        add("imageio", True)
    except Exception as exc:
        add("imageio", False, repr(exc))

    try:
        import analyzer
        configs, loaded = analyzer.load_templates((400, 225))
        per = {k: len(v) for k, v in configs.items()}
        add("模板加载", loaded > 0, f"共 {loaded} 个 {per}")
    except Exception as exc:
        add("模板加载", False, repr(exc))

    try:
        import gpu_caps
        path = gpu_caps.resolve_ffmpeg_path()
        info = gpu_caps.ffmpeg_info(path)
        add("ffmpeg", True, f"{path} | hwaccels={','.join(info.get('hwaccels') or [])}")
        working = gpu_caps.list_working_gpu_encoders(path)
        add("硬件编码器实测", True, ", ".join(working) or "无（将用 libx264）")
    except Exception as exc:
        add("ffmpeg", False, f"{type(exc).__name__}: {exc}")

    try:
        import gpu_caps
        pack = gpu_caps.torch_dir()
        note(f"[INFO] torch_dir={pack} "
             f"has_torch={os.path.isdir(os.path.join(pack, 'torch'))}")
        gpu_caps._register_paths(pack)
        note("[STEP] pack dir registered, importing torch ...")
        import torch  # noqa: F401
        note(f"[STEP] torch imported: {torch.__version__} ({torch.__file__})")
        note(f"[STEP] cuda_available={torch.cuda.is_available()}")
        note(f"[STEP] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
        st = gpu_caps.torch_status(refresh=True)
        add("CUDA 匹配加速", bool(st.get("cuda_available")), gpu_caps.pack_status_text())
        note(f"[INFO] torch_status={st}")
    except Exception as exc:
        add("CUDA 匹配加速", False, f"{type(exc).__name__}: {exc}")
        note("[TRACE] " + traceback.format_exc())

    try:
        import matcher
        add("匹配后端可用性", True, ", ".join(matcher.available_backends()))
    except Exception as exc:
        add("匹配后端可用性", False, repr(exc))

    # 可选：给一段视频就真的跑一次完整分析（打包后的端到端验收）
    video = None
    if "--video" in sys.argv:
        i = sys.argv.index("--video")
        if i + 1 < len(sys.argv):
            video = sys.argv[i + 1]
    if video and os.path.isfile(video):
        try:
            import time as _time
            import analyzer
            configs, _loaded = analyzer.load_templates((400, 225))
            thr = {"pause": 0.7, "speed_1x": 0.7, "speed_2x": 0.7, "speed_0_2x": 0.7}
            stats = {}
            t0 = _time.perf_counter()
            states, diffs, ctx = analyzer.analyze_video_with_context(
                video, configs, thr, (400, 225), 128, 8,
                decode_backend="auto", match_backend="auto",
                skip_identical=True, on_stats=stats.update)
            dt = _time.perf_counter() - t0
            uniq = {int(k): int(v) for k, v in zip(*__import__("numpy").unique(states, return_counts=True))}
            add("端到端分析", len(states) > 0,
                f"{len(states)} 帧 / {dt:.2f}s / {dt / max(1, len(states)) * 1000:.3f} ms/帧 / "
                f"解码={stats.get('decode_backend')} 匹配={stats.get('match_backend')} / "
                f"状态分布={uniq} / context完整={ctx.get('complete')}")
        except Exception as exc:
            add("端到端分析", False, f"{type(exc).__name__}: {exc}")

        # 模板自匹配 + 自动测速（与界面里「测速」走的同一条路）
        try:
            import matcher
            proc = (400, 225)
            cfgs, _n = analyzer.load_templates(proc)
            self_scores = analyzer.template_self_scores(cfgs, proc)
            note("[INFO] 模板自匹配分数 " +
                 str({k: round(v, 4) for k, v in self_scores.items()}))
            low = [k for k, v in self_scores.items() if v < thr.get(k, 0.7) + 0.05]
            if low:
                note(f"[WARN] 模板自匹配余量不足（真实视频上容易漏判/串类）: {low}")
            sample = matcher.sample_gray_frames(video, proc, count=32,
                                                decode_backend="auto")
            compiled = matcher.compile_templates(
                cfgs, proc, frame_wh=(sample.shape[2], sample.shape[1]))
            results, chosen = matcher.autotune(compiled, cfgs, thr, proc, sample)
            for r in results:
                note(f"[AUTOTUNE] {r.backend:<12} ok={int(r.ok)} "
                     f"Δ={r.max_delta:.2e} {r.ms_per_frame:7.4f} ms/帧 {r.error}")
            note(f"[AUTOTUNE] 已选 {chosen}")
        except Exception as exc:
            note(f"[WARN] 模板自匹配/测速检查失败: {type(exc).__name__}: {exc}")
    elif video:
        add("端到端分析", False, f"视频不存在: {video}")

    add("multiprocessing.freeze_support", True)

    note(f"\n结果: {'全部通过' if ok else '有失败项'}")
    flush()
    for line in lines:
        print(line, flush=True)
    return 0 if ok else 1


>>>>>>> Stashed changes
def main():
    root = tk.Tk()
    root.title("明日方舟剪辑工具")
    root.geometry("1380x920")
    root.minsize(900, 600)

    # 顶部工具栏
    top = ttk.Frame(root)
    top.pack(fill=tk.X, padx=10, pady=6)
    ttk.Label(top, text="视频:").pack(side=tk.LEFT)
    input_var = tk.StringVar()
    ttk.Entry(top, textvariable=input_var, width=60).pack(side=tk.LEFT, padx=5)

    # ---- 可拖动左右面板 ----
    paned = tk.PanedWindow(root, orient=tk.HORIZONTAL,
                           sashwidth=6,          # 分隔条宽度（px）
                           sashrelief=tk.RAISED,
                           bg="#555555")
    paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

    # 左：播放器
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, stretch="always", minsize=500)

    # 右：设置面板（默认宽度 360，可拖到更宽）
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, stretch="never", minsize=240, width=360)

    settings = SettingsPanel(right_frame)
    settings.pack(fill=tk.BOTH, expand=True)

    player = VideoPreviewPlayer(left_frame, settings=settings)
    player.pack(fill=tk.BOTH, expand=True)

    # 绑定导出
    settings.export_callback = player.export_video
    settings.segment_export_callback = player.export_segments
    # 绑定批量暂停模式按钮
    settings.apply_pause_callback = player.apply_pause_mode

    def open_file():
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                       ("所有文件",  "*.*")])
        if not path:
            return
        input_var.set(path)
        if not settings.output_var.get():
            name, _ = os.path.splitext(path)
            settings.output_var.set(f"{name}_clipped.mp4")
        player.load_video(path)

    ttk.Button(top, text="打开视频", command=open_file).pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    # Windows 下用 PyInstaller/cx_Freeze 打包时必须调用，
    # 否则 ProcessPoolExecutor 会递归启动子进程崩溃
    multiprocessing.freeze_support()
    main()