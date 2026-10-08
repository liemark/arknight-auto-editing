"""两个辅助损失的不变量测试 (纯 CPU, 秒级).

  1. set_ce  : --tir-conf 0 时必须与 F.cross_entropy 逐位相同 (逐位相同)
  2. onset_margin : onset 比后续持续段高出 margin 时损失为 0, 否则线性增长;
                    窗口里别人的真值 onset 必须被排除
"""
import os, sys
import torch
import torch.nn.functional as F
ML = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ML)
from train import set_ce, onset_margin

print("=== set_ce ===")
torch.manual_seed(0)
K, M, n = 50, 8, 64
nbr = torch.randint(0, K, (K, M))
target = torch.randint(0, K, (n,))
logits = torch.randn(n, K)
val_lo = torch.rand(K, M) * 0.5
val_hi = torch.rand(K, M) * 0.5 + 0.5

a = set_ce(logits, target, nbr, val_lo, 0.99, M)
b = F.cross_entropy(logits, target)
print("1) conf 高于全部邻居 -> 退化成 CE : %.9f vs %.9f  diff %.2e" % (a.item(), b.item(), abs(a.item() - b.item())))
assert abs(a.item() - b.item()) < 1e-6

c = set_ce(logits, target, nbr, val_hi, 0.0, M)
logp = F.log_softmax(logits, dim=-1)
nb = nbr[target]
keep = nb != target[:, None]
sel = torch.where(keep, logp.gather(1, nb), torch.full_like(logp[:, :M], float("-inf")))
expect = -torch.logsumexp(torch.cat([logp.gather(1, target[:, None]), sel], 1), 1).mean()
print("2) conf=0 -> 并入全部邻居       : %.9f vs hand %.9f  diff %.2e" % (c.item(), expect.item(), abs(c.item() - expect.item())))
assert abs(c.item() - expect.item()) < 1e-6
print("3) 集合目标不会比 CE 更严       : %.6f <= %.6f" % (c.item(), b.item()))
assert c.item() <= b.item() + 1e-9

print("\n=== focal_prob (arXiv 2601.04178 的 OOL) ===")
from train import focal_prob
torch.manual_seed(1)
p = torch.rand(2000) * 0.98 + 0.01
y = (torch.rand(2000) < 0.1).float()
# 1) gamma=0 必须逐位等于 BCE(p, y)
f0 = focal_prob(p, y, 0.0)
bc = F.binary_cross_entropy(p, y)
print("1) gamma=0 -> BCE            : %.9f vs %.9f  diff %.2e" % (f0.item(), bc.item(), abs(f0.item() - bc.item())))
assert abs(f0.item() - bc.item()) < 1e-6
# 2) y in {0,1} 时必须等于论文 Eq.4 的原式
pe = p.clamp(1e-6, 1 - 1e-6)
pos = (1 - pe) ** 2 * (-torch.log(pe))
neg = pe ** 2 * (-torch.log(1 - pe))
ref = (y * pos + (1 - y) * neg).mean()
f2 = focal_prob(p, y, 2.0)
print("2) y in {0,1} -> 论文 Eq.4     : %.9f vs %.9f  diff %.2e" % (f2.item(), ref.item(), abs(f2.item() - ref.item())))
assert abs(f2.item() - ref.item()) < 1e-6
# 3) focal 必须比 BCE 更不惩罚【难样本】之外的容易样本, 即对高置信正样本的损失更小
ph = torch.tensor([0.9, 0.99]); yh = torch.tensor([1.0, 1.0])
print("3) 高置信正样本: BCE %.6f -> focal %.6f (应当更小)" % (
    F.binary_cross_entropy(ph, yh).item(), focal_prob(ph, yh, 2.0).item()))
assert focal_prob(ph, yh, 2.0).item() < F.binary_cross_entropy(ph, yh).item()
# 4) 软标签(我们的脉冲目标)也要能算, 且有限
ys = torch.full((100,), 0.5)
assert torch.isfinite(focal_prob(torch.full((100,), 0.5), ys, 2.0))
print("4) 软标签脉冲目标可算        : OK")

print("\n=== onset_margin ===")
B, Tm, G = 1, 40, 3
ev = torch.zeros(B, G, 2, dtype=torch.long)
ev[0, 0] = torch.tensor([10, 70])                 # onset -> 输出帧 5, 事件到帧 35
valid = torch.tensor([[True, False, False]])
lg = torch.zeros(B, Tm); lg[0, 5] = 5.0

lg[0, 20] = 3.0
r = onset_margin(lg, ev, valid, 25, 2.0)
print("1) 后段最高 3, 相差恰好 2 -> loss %.4f" % r.item()); assert abs(r.item()) < 1e-6
lg[0, 20] = 5.0
r = onset_margin(lg, ev, valid, 25, 2.0)
print("2) 后段与 onset 齐平      -> loss %.4f (应 = margin 2.0)" % r.item()); assert abs(r.item() - 2.0) < 1e-6
lg[0, 20] = 6.0
r = onset_margin(lg, ev, valid, 25, 2.0)
print("3) 后段反超 onset 1       -> loss %.4f" % r.item()); assert abs(r.item() - 3.0) < 1e-6
lg[0, 20] = 0.0; lg[0, 36] = 9.0
r = onset_margin(lg, ev, valid, 25, 2.0)
print("4) 窗口外的高峰不算       -> loss %.4f" % r.item()); assert abs(r.item()) < 1e-6

ev2 = ev.clone(); ev2[0, 1] = torch.tensor([40, 70])
valid2 = torch.tensor([[True, True, False]])
lg[0, 20] = 9.0                                    # 就压在另一个真值 onset 上 (输出帧 20)
r = onset_margin(lg, ev2, valid2, 25, 2.0)
print("5) 别人的真值 onset 被排除 -> loss %.4f" % r.item()); assert abs(r.item()) < 1e-6

# 7) onset 在窗口外的事件 (ev[...,0] < 0, 合成器里"声音已经在响") 不能要求在帧 0 出峰
lg[0, 20] = 0.0; lg[0, 36] = 0.0; lg[0, 5] = 0.0   # 清掉前面用例留下的峰
ev3 = torch.zeros(B, G, 2, dtype=torch.long)
ev3[0, 0] = torch.tensor([-3, 40])                 # onset 在窗口左边之外, span 到输出帧 20
valid3 = torch.tensor([[True, False, False]])
lg[0, 0] = -9.0                                    # 帧 0 的最低分: 若被要求出峰, loss 会爆
r = onset_margin(lg, ev3, valid3, 25, 2.0)
print("7) onset 在窗口外的事件被跳过 -> loss %.4f" % r.item())
assert float(r) != float(r)                        # 空集合 -> nan (调用处用 von.any() 挡住)
ev4 = ev3.clone(); ev4[0, 1] = torch.tensor([2, 60]); valid4 = torch.tensor([[True, True, False]])
lg[0, 1] = 5.0                                     # 帧 1 才是真 onset (ev=2)
r = onset_margin(lg, ev4, valid4, 25, 2.0)
print("8) 混一个窗口内的真 onset -> loss %.4f  (被切掉的那个如果没排除, 这里会是 %.1f)"
      % (r.item(), 5.0 - (-9.0) + 2.0))
assert torch.isfinite(r) and float(r) < 1e-6

lg2 = lg.clone().requires_grad_(True)
r = onset_margin(lg2, ev, valid, 25, 2.0); r.backward()
print("6) 反传                    -> loss %.4f  梯度有限 %s" % (r.item(), bool(torch.isfinite(lg2.grad).all())))
assert torch.isfinite(lg2.grad).all() and lg2.grad.abs().sum() > 0

print("\n全部通过")
