# main.py —— 程序入口（PanedWindow 实现可拖动左右分隔）

import tkinter as tk
from tkinter import ttk, filedialog
import os
import multiprocessing   # ProcessPoolExecutor 需要在入口处 freeze_support

from settings_panel import SettingsPanel
from preview_player import VideoPreviewPlayer


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

    # 右：设置面板（默认宽度 300，可拖到更宽）
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, stretch="never", minsize=220, width=300)

    settings = SettingsPanel(right_frame)
    settings.pack(fill=tk.BOTH, expand=True)

    player = VideoPreviewPlayer(left_frame, settings=settings)
    player.pack(fill=tk.BOTH, expand=True)

    # 绑定导出
    settings.export_callback = player.export_video

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