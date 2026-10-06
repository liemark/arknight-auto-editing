"""audio_inverse: 明日方舟音效/语音反相抵消系统。

    x  ≈  Σ_i g_i · T_θ(a_i)  +  background

流程：
    1. 训练 + 识别   `train/` 下的脚本（PCEN 前端 + 卷积 stem + 注意力栈 + onset/原型/offset 头，
                     6576 类模板库，纯合成数据现场生成）
    2. 后处理        `audio_inverse.labcompat.postprocess` —— 读识别产出的 timeline，
                     用本包的 48 kHz 渲染链重新拟合增益、出反相轨与 PSR 报告

配套工具：
    audio_inverse.labcompat.prepare_lab_assets   造训练脚本要的数据文件 + 资产状态表
"""
from .config import Cfg, load_cfg, PKG_ROOT, REPO_ROOT

__all__ = ["Cfg", "load_cfg", "PKG_ROOT", "REPO_ROOT", "__version__"]
__version__ = "2.0.0"
