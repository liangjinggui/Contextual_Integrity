import torch
import torch.nn.functional as F
from methods.naive.losses import l_behav, l_attn


def test_l_behav_matches_cross_entropy():
    B, S, V = 2, 5, 7
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[:, :2] = -100  # prompt part ignored
    out = l_behav(logits, labels)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.isfinite(out) and torch.allclose(out, ref)


def test_l_attn_zero_when_clean():
    # private mass = 0 and share mass >= vanilla baseline -> nothing to penalize
    masses = {24: torch.tensor([0.0, 0.0, 0.5, 0.5]), 25: torch.tensor([0.0, 0.0, 0.6, 0.6])}
    base = {24: torch.tensor([0.1, 0.1, 0.4, 0.4]), 25: torch.tensor([0.1, 0.1, 0.5, 0.5])}
    out = l_attn(masses, base, share_idx=torch.tensor([2, 3]), priv_idx=torch.tensor([0, 1]), lam=1.0)
    assert torch.allclose(out, torch.tensor(0.0))


def test_l_attn_positive_when_private_mass():
    masses = {24: torch.tensor([0.3, 0.2, 0.5, 0.5])}  # private attrs still attended
    base = {24: torch.tensor([0.1, 0.1, 0.4, 0.4])}
    out = l_attn(masses, base, share_idx=torch.tensor([2, 3]), priv_idx=torch.tensor([0, 1]), lam=1.0)
    assert out > 0
