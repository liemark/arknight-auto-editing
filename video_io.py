# video_io.py — 视频 IO 线程
# 整个程序唯一持有 cv2.VideoCapture 的地方，
# 通过命令队列接收指令，把解码好的帧放入 frame_q。
#
# 主线程  ──cmd_q──▶  _VideoIOThread（唯一cap）──frame_q──▶  渲染循环

import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty

from frame_types import FRAME_TYPE_1X, FRAME_TYPE_0_2X

# ---------- 命令类型常量（导出供外部使用）----------
CMD_SEEK        = 'seek'         # 精确 seek，播放时应用裁剪区跳过
CMD_SEEK_LATEST = 'seek_latest'  # 节流 seek：丢弃积压旧命令，允许停在裁剪区内
CMD_PLAY        = 'play'
CMD_STOP        = 'stop'
CMD_QUIT        = 'quit'


class VideoIOThread(threading.Thread):
    """
    单一 cap 的视频 IO 线程。
    所有 cv2.VideoCapture 操作都在本线程内串行执行，
    彻底避免 FFmpeg async_lock 崩溃。
    """

    _GRAB_SEEK_THRESHOLD = 30  # 跨度超过此值改用 seek，避免长跨度 grab 阻塞

    def __init__(self, path: str, frame_q: Queue):
        super().__init__(daemon=True)
        self.path    = path
        self.frame_q = frame_q
        self.cmd_q: Queue = Queue()

        self._cap  = cv2.VideoCapture(path)
        self.fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur  = 0

        self._playing     = False
        self._play_params: dict = {}

    # ------------------------------------------------------------------
    # cap 操作（仅本线程调用）
    # ------------------------------------------------------------------

    def _seek_cap(self, frame_idx: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._cur = frame_idx

    def _grab_n(self, n: int):
        """跳过 n 帧：小跨度 grab（不解码），大跨度直接 seek（O(1)）"""
        target = min(self._cur + n, self.total - 1)
        if n > self._GRAB_SEEK_THRESHOLD:
            self._seek_cap(target)
        else:
            for _ in range(n):
                if not self._cap.grab():
                    break
                self._cur += 1

    def _read_one(self):
        ret, frame = self._cap.read()
        if ret:
            self._cur += 1
        return ret, frame

    # ------------------------------------------------------------------
    # 静态辅助
    # ------------------------------------------------------------------

    @staticmethod
    def jump_pause(cur: int, pause_segs: list) -> int:
        """
        循环跳过所有删除区，直到落点不在任何区间内。
        处理相邻/连续删除区（如暂停→普通→暂停紧挨着）：
        单次跳完第一个区间后落点可能仍在第二个区间，继续跳直到稳定。
        """
        while True:
            jumped = cur
            for ti, to in pause_segs:
                if to > ti and ti <= cur < to:
                    jumped = to
                    break
            if jumped == cur:
                return cur
            cur = jumped

    @staticmethod
    def speed_step(cur: int, speed_segs: list,
                   speedup_1x: bool, speedup_02: bool, factor_02: int) -> int:
        for s, e, t in speed_segs:
            if s <= cur <= e:
                if t == FRAME_TYPE_1X and speedup_1x:
                    return 2
                if t == FRAME_TYPE_0_2X and speedup_02:
                    return max(2, factor_02)
        return 1

    # ------------------------------------------------------------------
    # 队列辅助
    # ------------------------------------------------------------------

    def _flush_frame_q(self):
        while True:
            try:
                self.frame_q.get_nowait()
            except Empty:
                break

    def _push_frame(self, cur_idx: int, frame: np.ndarray, canvas_wh: tuple):
        """缩放到画布尺寸，转 RGB，推入帧队列（满则丢旧）"""
        cw, ch = canvas_wh
        fh, fw = frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw     = max(1, int(fw * scale))
        nh     = max(1, int(fh * scale))
        small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
            except Empty:
                pass
        self.frame_q.put((cur_idx, rgb))

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        base_frame_dur = 1.0 / self.fps

        while True:
            if not self._playing:
                # 阻塞等命令
                cmd = self.cmd_q.get()
                self._handle_cmd(cmd)
                continue

            # 播放中：优先响应命令
            try:
                cmd = self.cmd_q.get_nowait()
                self._handle_cmd(cmd)
                continue
            except Empty:
                pass

            if not self._playing:
                continue

            t0 = time.monotonic()
            pp           = self._play_params
            pause_segs   = pp['pause_segs']
            skip_trim    = pp['skip_trimmed']
            preview_step = pp['preview_step']
            # speed_multiplier > 1 → 慢速（帧间隔放大）；= 1 → 正常/加速（靠step）
            speed_mult   = pp.get('speed_multiplier', 1.0)
            frame_dur    = base_frame_dur * speed_mult
            canvas_wh    = pp['canvas_wh']

            # 1. 跳过暂停裁剪区
            if skip_trim:
                jumped = self.jump_pause(self._cur, pause_segs)
                if jumped != self._cur:
                    self._grab_n(jumped - self._cur)
                    self._flush_frame_q()
                    try:
                        cmd = self.cmd_q.get_nowait()
                        self._handle_cmd(cmd)
                        continue
                    except Empty:
                        pass

            if self._cur >= self.total:
                self._playing = False
                continue

            # 2. 步幅
            s_step     = self.speed_step(self._cur, pp['speed_segs'],
                                         pp['speedup_1x'], pp['speedup_02'],
                                         pp['speedup_02_factor'])
            total_step = preview_step * s_step

            # 3. 读帧
            ret, frame = self._read_one()
            if not ret:
                self._playing = False
                continue

            read_pos = self._cur

            # 4. 变速 / 预览跳帧
            skip = total_step - 1
            if skip > 0:
                if skip_trim:
                    landed = self.jump_pause(self._cur + skip, pause_segs)
                    if landed != self._cur + skip:
                        self._grab_n(landed - self._cur)
                        self._flush_frame_q()
                    else:
                        self._grab_n(skip)
                else:
                    self._grab_n(skip)

            # 5. 推帧
            self._push_frame(read_pos, frame, canvas_wh)

            # 6. 帧率控制
            elapsed = time.monotonic() - t0
            rem = frame_dur - elapsed
            if rem > 0.0005:
                time.sleep(rem)

        self._cap.release()

    # ------------------------------------------------------------------
    # 命令处理
    # ------------------------------------------------------------------

    def _handle_cmd(self, cmd: dict):
        t = cmd['type']

        if t == CMD_QUIT:
            self._playing = False
            self._cap.release()
            raise SystemExit

        elif t == CMD_STOP:
            self._playing = False
            self._flush_frame_q()

        elif t in (CMD_SEEK, CMD_SEEK_LATEST):
            self._playing = False
            self._flush_frame_q()
            frame_idx  = cmd['frame']
            canvas_wh  = cmd['canvas_wh']
            pause_segs = cmd.get('pause_segs', [])
            skip_trim  = cmd.get('skip_trimmed', True)

            # CMD_SEEK（播放 seek）才跳过裁剪区
            # CMD_SEEK_LATEST（拖动/点击）允许停在裁剪区内
            if skip_trim and t == CMD_SEEK:
                frame_idx = self.jump_pause(frame_idx, pause_segs)

            self._seek_cap(frame_idx)
            ret, frame = self._read_one()
            if ret:
                self._push_frame(frame_idx, frame, canvas_wh)

        elif t == CMD_PLAY:
            self._flush_frame_q()
            self._play_params = cmd['params']
            self._seek_cap(cmd['params']['start_frame'])
            self._playing = True

    # ------------------------------------------------------------------
    # 外部接口（线程安全）
    # ------------------------------------------------------------------

    def send(self, cmd: dict):
        """发送命令。CMD_SEEK_LATEST 会丢弃积压的旧同类命令（节流）"""
        if cmd['type'] == CMD_SEEK_LATEST:
            kept = []
            while True:
                try:
                    old = self.cmd_q.get_nowait()
                    if old['type'] != CMD_SEEK_LATEST:
                        kept.append(old)
                except Empty:
                    break
            for c in kept:
                self.cmd_q.put(c)
        self.cmd_q.put(cmd)

    def stop_and_quit(self):
        """优雅退出（程序关闭时调用）"""
        try:
            self.cmd_q.put({'type': CMD_QUIT})
        except Exception:
            pass