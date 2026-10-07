# arknight-auto-editing
![alt text](https://github.com/liemark/arknight-auto-editing/blob/main/README.png)  
明日方舟可视化剪暂停与变速工具  
根据文件夹中的模板图片识别视频中每一帧的状态  
并根据暂停事件前后帧的差异决定该暂停是否保留  
如果该暂停被保留，则寻找暂停区间内有操作的部分并保留，
亮绿色部分视为有效操作，红色部分视为无效操作  
对亮绿色/深棕色/红色片段右键单击可切换是否去掉该片段（例如去掉无效操作）  
对于1倍速事件与0.2倍速事件可倍速播放  
默认参数已经有较好的剪辑效果  
还提供了时间轴用于暂停事件的精细化调整与视频预览  

## 性能（本机实测：RTX 4060 Laptop + 20 线程，40s 1080p60 HEVC 素材 2400 帧）

分析整段视频的端到端耗时（同一份素材、默认 400×225、batch 128，`tools/bench.py decode`）：

| 路径 | ms/帧 | 相对改造前 | 分类一致率 |
|---|---|---|---|
| **改造前**：OpenCV 解码 + 串行预处理 + 进程池逐帧匹配 | 27.27 | 1.00× | — |
| 新流水线 + OpenCV 解码 + GPU 匹配 | 16.21 | 1.68× | 2400/2400 |
| FFmpeg 软件解码 + GPU 匹配 | 5.06 | 5.39× | 2400/2400 |
| **NVDEC 硬解 + scale_cuda + GPU 匹配** | **1.92** | **14.24×** | **2400/2400** |

模板匹配单帧耗时（128 帧批量实测，同一批模板，`tools/bench.py match`）：

| 匹配后端 | ms/帧 | 说明 |
|---|---|---|
| CUDA(torch) | 0.100 | 需要可选 CUDA 加速包；比 cv2 直算快 ~25× |
| CPU 批量（band 堆叠 + TM_CCORR） | 0.674 | 零新依赖，全平台可用 |
| OpenCL(T-API) | 2.13 | 免依赖的 GPU 路径（NVIDIA/AMD/Intel 通用） |
| cv2 直算（改造前） | 2.53 | 基准 |

导出（40s 素材切出 150 帧，`tools/bench.py export`）：

| 方式 | 耗时 | 帧数校验 | 说明 |
|---|---|---|---|
| 无损直通（段起点=关键帧） | 1.8–2.0s | 精确 | 逐段 `-c copy` + concat，零重编码零画质损失 |
| 分块并行 + NVENC（4 块） | 3.96s | 精确 | 比单遍滤镜快 ~1.8× |
| 单遍滤镜 + NVENC | 6.98s | 精确 | 硬件编解码 |
| 单遍滤镜 + libx264 veryfast | 5.28s | 精确 | 本片段里解码占比大，所以与硬件编码接近 |
| 单遍滤镜 + libx264 medium | 6.78s | 精确 | 改造前等价 baseline |

> 硬件编码（NVENC）在同档质量参数下体积约为 libx264 的 3 倍（本片段 42.2MB vs 13.5MB），
> 这是显卡编码的常见特性；需要体积时在导出页选 libx264，需要速度时选硬件编码。
> 上面是同一台机器一轮连续测量的结果，绝对数字会随机器状态浮动。

### 加速来自哪里

1. **解码/预处理搬到显卡**：FFmpeg `-hwaccel cuda -hwaccel_output_format cuda` +
   `scale_cuda` + `hwdownload,format=gray`，直接把 proc_res 的灰度帧送进内存；
   AMD 走 `d3d11va`，Intel 走 `qsv`。所有变体都是**实测**选中，不是查表。
2. **匹配批量化 + GPU**：掩码 NCC 用 `num/den` 两次互相关表达（公式已与
   `cv2.matchTemplate(TM_CCOEFF_NORMED, mask=)` 逐位对齐，max|Δ|≈1e-6），
   因此可以整批跑在 torch(CUDA) / OpenCL / CPU 上；再由「自动测速」挑最快的。
3. **流水线化**：读帧线程 + 预处理线程池 + 批匹配，解码与计算重叠。
4. **算法热点**：`_speedup_mask` 从 O(段数×帧数) 改成 O(帧数)
   （20 万帧实测 1389ms → 11ms）；暂停掩码、时间轴静态层、
   跳过区快照全部向量化并加缓存。

### 识别正确性

所有 GPU 路径都以「cv2 直算」为基准做逐帧对照：随机数据与真实模板的分数
max|Δ| ≤ 1e-6，分类一致率 2400/2400（100%）。任何后端只要偏差超阈值
（分数 max|Δ| > 1e-3，或出现非「贴近阈值」的分类分歧）就不会被自动选中。

## UI 说明（新增「性能 / GPU」页）

* **硬件能力**：厂商/型号、实测可用的解码变体、匹配后端、编码器列表、ffmpeg 路径。
  全部**实测**得出；点「重新探测硬件」可重测。
* **解码加速**：自动（硬件优先）/ 不加速(OpenCV) / FFmpeg 软件(A_PT) / 各硬件变体。
  「自动」= 有实测可用的硬件解码就用，否则完全等价于改造前的 OpenCV 行为。
* **匹配计算**：自动测速 / cv2 直算 / CPU 批量 / CPU 多进程池 / OpenCL / CUDA。
  不可用的后端会标注「（不可用）」；「重新测速」实测各后端并选出最快的。
* **静止帧跳过**：与上一帧像素完全相同时复用上一帧判定（判定必然相同，只是省算力）。
* **CUDA 匹配加速包**（可选，约 2.5GB）：检测到 NVIDIA 且没有可用 CUDA 时会询问；
  也可手动「下载并启用」，或换成离线加速包。下载约 2.75GB；装好后点一次
  「重新测速」即可把匹配切到 CUDA。加速包只有 torch，可放任意可写目录复用。
* **运行事件区**：所有降级/回退原因都会出现在这里（打包成 windowed exe 后没有
  控制台，以前这些信息是看不到的），可点「查看完整日志」。
* **导出页**：编码器（实测可用的排前面）、编码速度预设（随编码器切换）、
  导出并发数、极速无损模式、**真实进度 + 剩余时间 + 取消导出**。

## uv 安装

```bash
uv sync
uv run arknight-auto-editing
```

如果只想按依赖文件安装，也可以使用：

```bash
uv pip install -r requirements.txt
```

### 可选：CUDA 匹配加速（NVIDIA）

不装也能跑（匹配自动回退 OpenCL / CPU）：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

或直接用程序「性能/GPU」页的「下载并启用」，或 `packaging/gpu_pack.ps1` 构建离线加速包。

## 打包发布

```powershell
powershell -File packaging\build.ps1
```

产物（`dist\`）：

* `v<版本>-win.zip` 主包：`剪暂停<版本>.exe` + 8 个模板/源图目录，顶层形态与
  `v26.7.23-win.zip` 一致；exe 为 onefile/windowed（本机实测 80MB）。
* `v<版本>-win-full.zip` 完整包：主包再内置 `ffmpeg.exe`/`ffprobe.exe`（约 217MB）
  与 `uv.exe`，开箱即用硬件解码/编码与程序内下载 CUDA 加速包。

> `build.ps1` 与 `gpu_pack.ps1` 刻意保持纯 ASCII：Windows PowerShell 5.1 会把
> 无 BOM 的 UTF-8 脚本按 ANSI 解析，中文会破坏语法。
>
> 默认不执行 `uv sync`（它会清理锁文件之外的包）；需要时加 `-Sync`。

自检（打包后可直接验证依赖/模板/ffmpeg，以及跑一次真实分析）：

```powershell
dist\剪暂停26927.exe --check --video tools\fixtures\ref40s.mp4
```

结果写在 exe 同目录的 `check-report.txt`。

## 测试与基准

```powershell
powershell -File tools\make_fixture.ps1      # 切出 40s 测试片段（其余脚本都依赖它）
uv run python tools\selftest.py              # 等价性自检（NCC 公式/热点函数/一致率/UI 接线）
uv run python tools\bench.py all             # 基准：匹配 / 解码 / 导出 / 编码器
uv run python tools\bench.py match           # 只跑某一项
```

`tools\bench.py` 的结果写在 `tools\out\*.json`；素材与产物都在 .gitignore 里。

> `packaging\*.ps1` 与 `tools\make_fixture.ps1` 刻意保持纯 ASCII：Windows
> PowerShell 5.1 会把无 BOM 的 UTF-8 脚本按 ANSI 解析，中文会破坏语法。
>
> `build.ps1` 默认不执行 `uv sync`（它会清理锁文件之外的包）；需要时加 `-Sync`。

自检（打包后可直接验证依赖/模板/ffmpeg，以及跑一次真实分析）：

```powershell
dist\剪暂停26106.exe --check --video tools\fixtures\ref40s.mp4
```

结果写在 exe 同目录的 `check-report.txt`。

## 代码结构

| 模块 | 职责 |
|---|---|
| `analyzer.py` | 模板加载 + 段落提取 + 删除掩码（并 re-export 其余模块的入口） |
| `pipeline.py` | 解码后端（含硬件解码）+ 读帧/批处理流水线 + 帧差向量化 |
| `matcher.py` | 掩码 NCC 多后端（CUDA/OpenCL/CPU）+ 自动测速选路 |
| `gpu_caps.py` | GPU/FFmpeg 能力实测、编码参数、解码变体、可选 CUDA 加速包 |
| `exporter.py` | 导出阶梯：无损直通 / 分块并行 / 滤镜编码 / 逐帧兜底 |
| `app_core.py` | 配置与缓存持久化 + 运行事件出口 |
| `video_io.py` / `preview_player.py` / `timeline_widget.py` / `settings_panel.py` | 预览、时间轴与界面 |

```
链接: https://pan.baidu.com/s/1_LF18ARW5CLo62MeSYMVpQ?pwd=2333
提取码: 2333
```
