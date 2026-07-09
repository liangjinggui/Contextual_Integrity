"""losses 单测（CPU）：Lorentz 几何性质 + anchor/sup 损失行为。"""

import torch

from methods.hyper_icl.losses import lorentz_expmap, lorentz_dist, l_h_anchor, l_sup

KAPPA = 0.1


def test_expmap_lands_on_hyperboloid():
    """exp_o(u) 必须落在双曲面上：<p,p>_L = -1/kappa。"""
    torch.manual_seed(0)
    u = torch.randn(5, 8)
    p = lorentz_expmap(u, KAPPA)                                   # (5, 9)
    inner = -p[..., 0] ** 2 + (p[..., 1:] ** 2).sum(-1)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / KAPPA), atol=1e-4)


def test_dist_basic_properties():
    """d(x,x)=0；对称；不同点距离 > 0。"""
    torch.manual_seed(0)
    p = lorentz_expmap(torch.randn(4, 8), KAPPA)
    q = lorentz_expmap(torch.randn(4, 8), KAPPA)
    assert torch.allclose(lorentz_dist(p, p, KAPPA), torch.zeros(4), atol=1e-3)
    assert torch.allclose(lorentz_dist(p, q, KAPPA), lorentz_dist(q, p, KAPPA), atol=1e-5)
    assert (lorentz_dist(p, q, KAPPA) > 0).all()


def test_expmap_is_radial_isometry():
    """基点 o = exp_o(0)；测地距离 d(exp_o(u), o) 应等于 ||u||。"""
    u = torch.zeros(3, 8)
    u[0, 0], u[1, 1], u[2, :2] = 2.0, 0.5, torch.tensor([3.0, 4.0])  # 范数 2, 0.5, 5
    o = lorentz_expmap(torch.zeros(1, 8), KAPPA)
    d = lorentz_dist(lorentz_expmap(u, KAPPA), o, KAPPA)
    assert torch.allclose(d, torch.tensor([2.0, 0.5, 5.0]), atol=1e-3)


def test_l_h_anchor():
    """teacher==student -> 0；不同 -> >0；梯度只回 student 侧。"""
    torch.manual_seed(0)
    teacher = [torch.randn(5, 8) for _ in range(3)]                # 3 层、5 token、d=8
    same = [t.clone() for t in teacher]
    # 重合点处 acosh(1+fp32误差)^2 ~ 2e-6,是边界固有浮点行为(梯度有限),非bug
    assert l_h_anchor(same, teacher, KAPPA).item() < 1e-4

    student = [torch.randn(5, 8).requires_grad_(True) for _ in range(3)]
    loss = l_h_anchor(student, teacher, KAPPA)
    assert loss.item() > 0
    loss.backward()
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in student)


def test_l_sup_masking():
    """-100 位置不计损失：全 -100 时为 0 附近无 nan；有标签时 > 0。"""
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 32)
    labels = torch.full((1, 6), -100)
    labels[0, 3:] = torch.tensor([1, 2, 3])
    assert l_sup(logits, labels).item() > 0
