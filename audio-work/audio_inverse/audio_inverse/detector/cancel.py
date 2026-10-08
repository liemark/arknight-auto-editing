"""消除率（反相抵消）评测：**直接用"能消掉多少"判断模型对不对**。

为什么这才是主判据：
  acc / 识别率都是代理指标。用户最终要的是"把匹配到的音效反相放回原视频 -> 近乎静音"，
  所以模型"对不对"就等于"残差降了多少 dB"。而且它顺带覆盖了 acc 覆盖不到的东西：
  时延的亚采样精度、增益/频响/EQ 的估计质量 —— 反相是相干运算，
  残余 ≈ |1 − g·e^{−jωδ}|²，参数差一点点，消除率就掉下去。

三个数（每次评测都一起出，缺一个就无法解读）：
  PSR_truth  用**真值参数**渲染反相轨的消除率  = **理论上界**
  coh        真值反相轨 与 (mix − dry) 的相干度 = **渲染口径自检**
  PSR_model  用**模型预测**渲染反相轨的消除率  = 实际成绩
  PSR_null   什么都不消（= 0 dB，作为基线锚点）

`coh` 是诚实性检验：如果它远小于 1，说明 `synth/mixer.py` 的混合链与
`models/synthesize.render_events` 之间还有口径没对齐，此时**PSR_truth 不代表模型上限**，
必须先修渲染链，而不是去调模型。文档记录的相干度只有 ~0.1（`docs/PROJECT.md §6.2`），
本模块就是用来复现并跟踪那个问题的。

不可抵消的部分（残差地板）：背景 noise、干扰、底噪 floor、编解码、限幅、硬削波 ——
它们本来就不在"原子干声"里，任何反相轨都消不掉。所以 PSR_truth 天然不会是无穷大。
"""
from __future__ import annotations

import numpy as np
import torch

from ..audio import SR
from ..models.synthesize import Event, render_events
from ..synth.mixer import MixResult


# ------------------------------------------------------------------ 工具
def psr_db(mix: np.ndarray, cancel: np.ndarray, eps: float = 1e-12) -> float:
    """原/残 功率比 (dB)。cancel 是反相轨，mix + cancel = 残差。"""
    res = np.asarray(mix, dtype=np.float64) + np.asarray(cancel, dtype=np.float64)
    p_m = float((np.asarray(mix, dtype=np.float64) ** 2).sum()) + eps
    p_r = float((res ** 2).sum()) + eps
    return 10.0 * np.log10(p_m / p_r)


def coherence(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """零均值余弦相干度，用于"两条轨是不是同一件事"的自检。

    注意：这里**只做自检**，不进入训练路径。项目已弃用 NCC 做质量判据
    （见 `docs/NCC_RETIREMENT.md`），但"两条已知同一来源的轨是否相干"是它的正当用途。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n < 32:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


# ------------------------------------------------------------------ 真值事件
def truth_events(res: MixResult, *, atom_rms_db: float = -34.0) -> list[Event]:
    """把 MixResult 的实例转成渲染用的事件（含总线主增益 = 绝对增益）。"""
    evs: list[Event] = []
    for ins in res.instances:
        p = ins.p
        evs.append(Event(
            atom_id=int(ins.atom_id), delay=float(ins.delay),
            gain_db=float(ins.gain_db) + float(res.master_gain_db),
            r=float(p.r), tilt=float(p.tilt), pitch_semi=float(p.pitch_semi),
            eq_db=None if p.eq_db is None else np.asarray(p.eq_db, dtype=np.float32),
            wet=float(p.wet), drive=float(p.drive),
            rir=None if p.rir is None else np.asarray(p.rir, dtype=np.float32),
            source="truth"))
    return evs


# ------------------------------------------------------------------ 模型事件
def events_from_model(out: dict, labels_param_np, *, dev, sr: int = SR,
                      score_thr: float = 0.5, top_k: int = 1) -> list[Event]:
    """模型输出 -> 事件列表（用于渲染反相轨）。

    做法：帧级 `event_logit` 过阈值取峰 -> 每峰的 top-k 候选 -> 该位置的
    `params`（推理时就是 top-1 候选那条路径）-> Event。
    峰之间做不应期抑制（默认 0.5 帧），与旧包 `infer.py` 的 peak-picking 同思路。

    labels_param_np: 仅用于取 T（样本数）与 n_frames，可为 None。
    """
    logit = torch.sigmoid(out["event_logit"].float()).cpu().numpy()      # [B,N]
    params = out["params"].float().cpu().numpy()                         # [B,N,P]
    idl = out["id_logit"].float().cpu().numpy()                          # [B,N,K]
    cand = out["cand_row"].cpu().numpy()                                 # [B,N,K]
    pos_off = out["pos_off"].float().cpu().numpy()                       # [B,N]
    n_frames = logit.shape[1]
    frame_rate = 25.0
    evs: list[Event] = []
    for b in range(logit.shape[0]):
        p = logit[b]
        order = np.argsort(-p)
        taken = np.zeros(n_frames, dtype=bool)
        for i in order:
            if p[i] < score_thr:
                break
            lo, hi = max(0, i - 1), min(n_frames, i + 2)
            if taken[lo:hi].any():
                continue
            taken[lo:hi] = True
            row = idl[b, i]
            top = np.argsort(-row)[:top_k] if top_k > 1 else [int(np.argmax(row))]
            for k in top:
                v = params[b, i]
                t = (i + float(pos_off[b, i]) * frame_rate) / frame_rate
                t = float(t) + 0.0
                evs.append(Event(
                    atom_id=int(cand[b, i, k]), delay=float(t),
                    gain_db=float(v[1]), r=1.0 + float(v[2]), tilt=float(v[3]),
                    pitch_semi=float(v[4]),
                    eq_db=np.asarray(v[5:21], dtype=np.float32),
                    wet=float(v[21]), drive=float(v[22]), score=float(row[k]),
                    source="model"))
    return evs


# ------------------------------------------------------------------ 主入口
def cancel_report(lib, res: MixResult, *, model_events: list[Event] | None = None,
                  atom_rms_db: float = -34.0, sr: int = SR) -> dict:
    """一次完整评测：理论上界 + 渲染自检 + 模型消除率。"""
    n = int(res.mix.size)
    mix = np.asarray(res.mix, dtype=np.float32)

    # ---- 1) 理论上界（真值参数）----
    ev_truth = truth_events(res, atom_rms_db=atom_rms_db)
    cancel_truth = render_events(lib, ev_truth, n, sr=sr, invert=True,
                                 atom_rms_db=atom_rms_db)
    out = {"psr_truth": psr_db(mix, cancel_truth), "n_truth": len(ev_truth)}

    # ---- 2) 渲染口径自检 ----
    # 原子干声的"应然"值 = mix − background − interference（主增益已在 res.mix 里，
    # 而 background/interference 是在主增益之前加的，所以这里要按同一主增益缩放）。
    g = 10 ** (float(res.master_gain_db) / 20.0)
    dry_expected = mix - (np.asarray(res.background, dtype=np.float32)
                          + np.asarray(res.interference, dtype=np.float32)) * g
    dry_rendered = -cancel_truth                      # render_events(invert=True) 的负
    out["coh"] = coherence(dry_rendered, dry_expected)
    out["dry_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(dry_rendered ** 2)) + 1e-12))
    out["resid_floor_db"] = float(20 * np.log10(
        np.sqrt(np.mean((mix + cancel_truth) ** 2)) + 1e-12))
    out["mix_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(mix ** 2)) + 1e-12))

    # ---- 3) 模型消除率 ----
    if model_events is not None:
        cancel_model = render_events(lib, model_events, n, sr=sr, invert=True,
                                     atom_rms_db=atom_rms_db)
        out["psr_model"] = psr_db(mix, cancel_model)
        out["n_model"] = len(model_events)
        # 模型反相轨与真值反相轨的一致性：衡量"参数估计得像不像"
        out["coh_model"] = coherence(cancel_model, cancel_truth)
        # 拿模型事件数当分母做公平对比：漏检/多检都会直接压低 PSR
        out["psr_gap"] = out["psr_truth"] - out["psr_model"]
    return out


def format_report(r: dict) -> str:
    lines = [
        "  消除率（PSR，越大越好）",
        "    理论上界（真值参数）   %+7.2f dB" % r.get("psr_truth", float("nan")),
    ]
    if "psr_model" in r:
        lines.append("    模型（预测参数）       %+7.2f dB   差距 %+.2f dB"
                     % (r["psr_model"], -r.get("psr_gap", float("nan"))))
        lines.append("    模型反相轨 vs 真值反相  相干 %.3f" % r.get("coh_model", float("nan")))
    lines += [
        "  渲染口径自检（相干度应接近 1）",
        "    dry(渲染) vs dry(应然)  %.3f   <- 远小于 1 = 渲染链有口径未对齐"
        % r.get("coh", float("nan")),
        "    mix %.1f dBFS  理想残差底 %.1f dBFS"
        % (r.get("mix_rms_db", float("nan")), r.get("resid_floor_db", float("nan"))),
    ]
    return "\n".join(lines)
