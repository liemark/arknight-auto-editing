"""模型包。

只保留反相渲染链需要的部分：`synthesize`（事件列表 -> 波形）。
识别/训练用的网络（atom_search / frontend / model / transform）不在本项目里 ——
训练与识别由 `train/` 下的脚本负责。
"""
from .synthesize import (Event, band_psr, event_segments, ideal_cancel,
                         render_events, residual_metrics)

__all__ = ["Event", "render_events", "band_psr", "residual_metrics",
           "event_segments", "ideal_cancel"]
