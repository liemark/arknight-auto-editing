# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置（onefile + windowed）
#
# 关键点：
#   * 平铺 templates_* / source_images_* 八个素材目录；
#   * 显式排除 torch / tensorflow / librosa / pandas 等重型包（本机 .venv 与
#     其他项目共用，不排除会把 exe 撑到 3GB+）；这些包在可选加速包里自带；
#   * **整包收集标准库**：torch 是从加速包目录运行时动态导入的，PyInstaller
#     静态分析看不到它的依赖，缺任一标准库模块都会让 import torch 失败
#     （实测缺 timeit 时直接 ModuleNotFoundError）。
#
# 由 build.ps1 调用并传入 AAE_EXE_NAME。

import os
import sysconfig

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

SPEC_DIR = os.path.abspath(SPECPATH)
ROOT = SPEC_DIR if os.path.isfile(os.path.join(SPEC_DIR, "main.py")) \
    else os.path.dirname(SPEC_DIR)

EXE_NAME = os.environ.get("AAE_EXE_NAME") or "arknight-auto-editing"

DATA_DIRS = [
    "templates_pause", "templates_1x", "templates_2x", "templates_play",
    "source_images_pause", "source_images_1x", "source_images_2x", "source_images_play",
]

datas = [(os.path.join(ROOT, d), d) for d in DATA_DIRS
         if os.path.isdir(os.path.join(ROOT, d))]

# imageio 在 import 时会 importlib.metadata.version("imageio")，缺 dist-info 会报
# PackageNotFoundError，需要把发行版元数据一起打进去。
try:
    datas += copy_metadata("imageio")
except Exception:
    pass

binaries = []
for exe in ("ffmpeg.exe", "ffprobe.exe", "uv.exe"):
    src = os.path.join(ROOT, exe)
    if os.path.isfile(src):
        binaries.append((src, "."))

EXCLUDES = [
    "torch", "torchvision", "torchaudio", "torch_directml", "torchgen",
    "tensorflow", "keras", "tensorboard", "tensorflow_io",
    "transformers", "tokenizers", "safetensors", "huggingface_hub", "hf_xet",
    "librosa", "soundfile", "audioread", "soxr", "pyloudnorm", "scaper",
    "audiomentations", "pyroomacoustics", "fast_mp3_augment", "loudness",
    "pandas", "narwhals", "pyarrow",
    "scipy", "sklearn", "joblib", "numba", "llvmlite",
    "matplotlib", "contourpy", "fonttools", "kiwisolver", "cycler",
    "networkx", "sympy", "mpmath",
    "UnityPy", "texture2ddecoder", "etcpak", "faiss", "faiss_cpu",
    "IPython", "jupyter", "notebook", "pytest",
    "setuptools", "pip", "pkg_resources",
    # 标准库里用不到的大块头
    "idlelib", "lib2to3", "ensurepip", "venv", "turtledemo", "test", "tkinter.test",
]


def _stdlib_hiddenimports():
    """标准库所有顶层模块/包及其子模块。

    只加顶层包名不够：PyInstaller 只跟随包 __init__ 里的导入，像
    unittest.mock 这种惰性导入的子模块会漏掉（torch 就会踩到）。
    """
    stdlib = sysconfig.get_paths()["stdlib"]
    skip = set(EXCLUDES) | {"site-packages", "__pycache__"}
    names: set[str] = set()
    try:
        entries = os.listdir(stdlib)
    except OSError:
        return []
    for entry in sorted(entries):
        if entry in skip or entry.startswith("."):
            continue
        path = os.path.join(stdlib, entry)
        if entry.endswith(".py"):
            names.add(entry[:-3])
        elif os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            names.add(entry)
            try:
                names.update(collect_submodules(entry))
            except Exception:
                pass
    names -= {n for n in names if n.split(".")[0] in skip}
    return sorted(names)


hiddenimports = [
    "imageio",          # exporter 的逐帧兜底写入器是函数内延迟导入
    "PIL.ImageTk",
]
hiddenimports += _stdlib_hiddenimports()

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)   # noqa: F821

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
