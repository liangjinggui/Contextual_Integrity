"""CtxHyperNet 单测(小维度, CPU)。

闸门:形状 / 恒等初始化(U_k=0 -> Δ=0)/ 上下文条件性(不同 c 出不同 U)/
梯度逃逸(双零是鞍点——U_q 侧播种后, 几步优化内两半边都要动起来, naive Task2 教训)。"""

import torch

from methods.hyper_ctx.hypernet import CtxHyperNet

# 小维度:hidden 32, 3 层, 2 头, head_dim 8, rank 2
HID, L, H, D, R = 32, 3, 2, 8, 2


def make():
    torch.manual_seed(0)
    return CtxHyperNet(hidden_size=HID, n_layers=L, n_heads=H, head_dim=D, rank=R)


def test_shapes():
    g = make()
    c = torch.randn(1, HID)
    U_q, U_k = g(c, layer_idx=0)
    assert U_q.shape == (1, H, D, R)
    assert U_k.shape == (1, H, D, R)


def test_identity_at_init():
    """初始 U_k 必须为 0(=> Δ=0, identity 闸门);U_q 有播种非零(逃逸鞍点)。"""
    g = make()
    c = torch.randn(4, HID)
    for l in range(L):
        U_q, U_k = g(c, layer_idx=l)
        assert torch.all(U_k == 0)
        assert U_q.abs().sum() > 0


def test_context_conditioning():
    """训练后(模拟:head 权重非零)不同 c 必须给出不同 U——这就是与静态 adapter 的区别。"""
    g = make()
    with torch.no_grad():
        g.head.weight.normal_(0, 0.05)
    c1, c2 = torch.randn(1, HID), torch.randn(1, HID)
    Uq1, Uk1 = g(c1, layer_idx=1)
    Uq2, Uk2 = g(c2, layer_idx=1)
    assert not torch.allclose(Uq1, Uq2)
    assert not torch.allclose(Uk1, Uk2)


def test_gradient_escapes_saddle():
    """玩具目标下几步优化 loss 要降, 且 U_k 从 0 动起来(播种 U_q 保证 ∂Δ/∂U_k≠0)。
    注: Δ 对参数是四次的, lr 太大会过冲爆炸(1e-2 实测 3 步炸), 用温和 lr + 裁剪。"""
    g = make()
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    torch.manual_seed(1)
    c = torch.randn(1, HID)
    q = torch.randn(1, H, 5, D)               # 5 个 query token
    k = torch.randn(1, H, 5, D)
    losses = []
    for _ in range(5):
        U_q, U_k = g(c, layer_idx=0)
        delta = (q @ U_q) @ (k @ U_k).transpose(-1, -2)     # (1,H,5,5)
        loss = (delta - 1.0).pow(2).mean()                  # 逼 Δ 离开 0
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    U_q, U_k = g(c, layer_idx=0)
    assert U_k.abs().sum() > 0                              # U_k 已离开 0
