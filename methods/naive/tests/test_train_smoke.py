import pytest
import torch
from transformers import AutoTokenizer
from methods.naive.intervention import InterventionModel
from methods.naive.data import TargetDataset
from methods.naive.train_naive import train_step

DEV = "cuda:0"
TARGETS = "outputs/cimemories/self_distill_targets/Qwen3-8B/20260705_225937/train/targets.jsonl"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


@pytest.mark.slow
def test_train_smoke():
    """3 steps on one context: l_behav goes down, and ONLY Gtheta receives gradients."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    im = InterventionModel("Qwen/Qwen3-8B", band=range(24, 36), device=DEV, attn_implementation="sdpa")
    ds = TargetDataset(TARGETS, tok, PROMPTS, num_profiles=1)
    opt = torch.optim.Adam(im.gtheta.parameters(), lr=2e-4)

    item = ds[0]
    behavs = []
    for _ in range(3):
        lb, la, loss = train_step(im, item, DEV, beta=0.1, lam=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(im.gtheta.parameters(), 1.0)
        opt.step()
        behavs.append(lb.item())
        # base model gets NO gradients; Gtheta gets nonzero gradients
        assert all(p.grad is None for p in im.model.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in im.gtheta.parameters())

    assert behavs[-1] < behavs[0], behavs  # teacher-forcing loss decreased
