import torch
from methods.naive.hypernet import Gtheta


def test_identity_at_init():
    g = Gtheta(hidden_size=4096, head_dim=128, rank=8, n_layers=12)
    c = torch.randn(5, 4096)
    e = torch.randn(5, 4096)
    Pq, Pk = g(c, e, layer_idx=0)
    I = torch.eye(128).expand(5, 128, 128)
    assert torch.allclose(Pq, I, atol=1e-6)
    assert torch.allclose(Pk, I, atol=1e-6)


def test_trainable_after_step():
    g = Gtheta(hidden_size=4096, head_dim=128, rank=8, n_layers=12)
    c = torch.randn(3, 4096)
    e = torch.randn(3, 4096)
    Pq, Pk = g(c, e, 3)
    (Pq.sum() + Pk.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
