"""后处理与数据准备。

  * `assets`  准备训练脚本要读的数据文件（bg44.npy / long44/index_long.json）+ 资产状态表
  * `render`  识别产出的 timeline -> 反相轨（本包 48 kHz 渲染链）+ PSR 报告
  * `export`  把 Matching 结果铺成【试听用】单轨（matched / resid / orig / A-B 交替）
"""
