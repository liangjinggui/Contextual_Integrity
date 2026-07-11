"""hyper_ctx 训练冒烟(slow, GPU):单 context 3 步——loss 下降、梯度只落在 CtxAdapter 上。
顺带验证 data 里新加的 instr_tok_span 在真实数据上有效(非空、位于 prompt 内)。"""

import pytest
import torch
from transformers import AutoTokenizer

from methods.hyper_icl.data import HyperICLDataset
from methods.hyper_ctx.train_hyper_ctx import load_patched_model, train_step

DEV = "cuda:0"
TARGETS = "outputs/cimemories/self_distill_targets/Qwen3-8B/20260705_225937/train/targets.jsonl"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


@pytest.mark.slow
def test_train_smoke():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ds = HyperICLDataset(TARGETS, tok, PROMPTS, num_profiles=1)
    item = ds[0]
    # instr span 健康:非空、在 prompt 区间内(信件之前)
    i0, i1 = item["instr_tok_span"]
    n_prompt = len(item["student_ids"]) - item["n_letter"]
    assert 0 < i0 < i1 <= n_prompt

    model, adapter, in_dev = load_patched_model("Qwen/Qwen3-8B", rank=4, device_map=DEV)
    opt = torch.optim.AdamW(adapter.parameters(), lr=2e-4)   # Gθ 路径四次型敏感, 5e-3 三步即炸
    losses = []
    for _ in range(3):
        anchor, sup, loss = train_step(model, adapter, item, in_dev, lam=0.5, kappa=0.1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        assert all(p.grad is None for p in model.parameters() if p.requires_grad is False)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())

    assert losses[-1] < losses[0], losses
