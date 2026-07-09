"""训练冒烟（slow, GPU）：单 context 3 步——loss 下降、梯度只落在 adapter 上。"""

import pytest
import torch
from transformers import AutoTokenizer

from methods.hyper_icl.data import HyperICLDataset
from methods.hyper_icl.train_hyper_icl import load_patched_model, train_step

DEV = "cuda:0"
TARGETS = "outputs/cimemories/self_distill_targets/Qwen3-8B/20260705_225937/train/targets.jsonl"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


@pytest.mark.slow
def test_train_smoke():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model, adapter, in_dev = load_patched_model("Qwen/Qwen3-8B", rank=4, device_map=DEV)
    ds = HyperICLDataset(TARGETS, tok, PROMPTS, num_profiles=1)
    opt = torch.optim.AdamW(adapter.parameters(), lr=5e-3)

    item = ds[0]
    losses = []
    for _ in range(3):
        anchor, sup, loss = train_step(model, adapter, item, in_dev, lam=0.5, kappa=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        # base 冻结无梯度；adapter 有非零梯度
        assert all(p.grad is None for p in model.parameters() if p.requires_grad is False)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())

    assert losses[-1] < losses[0], losses
