# arknight-auto-editing
![alt text](https://github.com/liemark/arknight-auto-editing/blob/main/README.png)  
明日方舟可视化剪暂停与变速工具  
根据文件夹中的模板图片识别视频中每一帧的状态  
并根据暂停事件前后的DCT差异与Hist相似度判断暂停事件如何保留  
对于1倍速事件与0.2倍速事件可倍速播放  
默认参数已经有一定的剪辑效果  
还提供了时间轴用于暂停事件的精细化调整与视频预览  
目前是纯python版本，瓶颈在H.264解码
""
cap.read
  视频解码 (BGR 原始尺寸)                                3.62 ms/帧  (276 fps)  
Resize + ColorConvert  
  读帧线程内串行预处理                                  10.11 ms/帧  (99 fps)  
  4线程并行预处理 (ThreadPool)                        3.73 ms/帧  (268 fps)  
_classify_gray  
  单核分类耗时                                           0.26 ms/帧  (3895 fps)  
进程间通信 (IPC) 负载模拟  
  Pickle 灰度图 (400x225): 0.009 ms
  Pickle 原始帧 (2560x1440): 2.130 ms
端到端速度
  4.25 ms/帧  (235 fps)
""
