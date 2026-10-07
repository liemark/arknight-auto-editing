# matcher.py —— 实现已并入 analyzer.py（单文件），这里做一次转发。
# 用模块级 __getattr__ 而不是 import *，避免 analyzer 初始化期间的循环导入。
import analyzer as _analyzer


def __getattr__(name):
    return getattr(_analyzer, name)
