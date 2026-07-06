import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.naive.intervention import InterventionModel

DEV = "cuda:0"  # run with CUDA_VISIBLE_DEVICES=<one free card>


@pytest.mark.slow
def test_identity_init_matches_base_logits():
    """At init (Gtheta -> P=I) the patched forward must equal the base forward, even with
    the full intervention path active (pool c/e -> Gtheta -> projected-score write-back)."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ids = tok("The person's address is 12 Oak Street. Please help me write a short polite message now.",
              return_tensors="pt").input_ids.to(DEV)
    S = ids.shape[1]
    qpos = torch.zeros(S, dtype=torch.bool)
    qpos[-3:] = True  # last 3 tokens act as the "generated" query rows -> exercises the write-back

    im = InterventionModel("Qwen/Qwen3-8B", band=range(24, 36), device=DEV)
    im.set_context(mem_tok_spans_per_attr={"a0": (0, 4), "a1": (4, 8)},
                   instr_tok_span=(8, S - 3), query_positions=qpos)
    with torch.no_grad():
        patched = im.forward(ids, attention_mask=torch.ones_like(ids)).logits

    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, attn_implementation="eager").to(DEV)
    with torch.no_grad():
        ref = base(ids).logits

    assert torch.allclose(patched, ref, atol=1e-3), (patched - ref).abs().max().item()
