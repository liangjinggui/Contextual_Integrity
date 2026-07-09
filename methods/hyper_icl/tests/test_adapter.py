"""HyperICLAdapter 单测：形状 / 恒等初始化 / 数值对照公式（小维度, CPU）。"""

import math
import torch
import torch.nn.functional as F

from methods.hyper_icl.adapter import HyperICLAdapter

# 小维度：2 层、3 头、head_dim 8、rank 2
L, H, D, R = 2, 3, 8, 2
B, T = 1, 5


def make():
    torch.manual_seed(0)
    adapter = HyperICLAdapter(n_layers=L, n_heads=H, head_dim=D, rank=R)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    return adapter, q, k


def test_shapes():
    adapter, q, k = make()
    delta, g = adapter(layer=0, q=q, k=k)
    assert delta.shape == (B, H, T, T)
    assert g.shape == (B, H, T)


def test_identity_at_init():
    """U_k 零初始化 -> Delta 恒为 0；w,b 零初始化 -> g 恒为 0.5。"""
    adapter, q, k = make()
    delta, g = adapter(layer=1, q=q, k=k)
    assert torch.all(delta == 0)
    assert torch.allclose(g, torch.full_like(g, 0.5))


def test_matches_formula():
    """随机 U_q/U_k/w/b 下，逐头对照论文式(6-7)与式(9)手算。"""
    adapter, q, k = make()
    with torch.no_grad():
        adapter.U_k.normal_(0, 0.1)
        adapter.w.normal_(0, 0.1)
        adapter.b.normal_(0, 0.1)
    layer = 0
    delta, g = adapter(layer=layer, q=q, k=k)

    for h in range(H):
        q_h, k_h = q[0, h], k[0, h]                       # (T, D)
        a = q_h @ adapter.U_q[layer, h]                   # (T, R)
        c = k_h @ adapter.U_k[layer, h]                   # (T, R)
        want_delta = a @ c.T / math.sqrt(R)               # 式(7)
        assert torch.allclose(delta[0, h], want_delta, atol=1e-6)

        qn = F.layer_norm(q_h, (D,))
        want_g = torch.sigmoid(qn @ adapter.w[layer, h] + adapter.b[layer, h])  # 式(9)
        assert torch.allclose(g[0, h], want_g, atol=1e-6)


def test_only_adapter_params():
    """参数量 = L*H*(2*D*R + D + 1)，全部可训练。"""
    adapter, _, _ = make()
    n = sum(p.numel() for p in adapter.parameters())
    assert n == L * H * (2 * D * R + D + 1)
