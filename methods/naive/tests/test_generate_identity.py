"""P=I generation == vanilla eager generation, token for token. The correctness gate for the
patched decode path: prefill-caches P, reuses it every decode step, yet with P=I it must add
nothing. Both sides use eager everywhere so the only difference under test is the patch itself."""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluation.cim_eval.eval import load_prompts, load_profiles
from methods.naive.intervention import InterventionModel
from methods.utils.spans import build_intervention_prompt, char_spans, token_spans

DEV = "cuda:0"
MODEL = "Qwen/Qwen3-8B"
TARGETS = "outputs/cimemories/self_distill_targets/Qwen3-8B/20260705_225937/train/targets.jsonl"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


@pytest.mark.slow
def test_generate_identity():
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = load_prompts(PROMPTS, 1)
    prof = load_profiles(TARGETS, 1)[0]
    ctx = (prof.get("contexts") or [])[0]

    prompt, mem_lines = build_intervention_prompt(P, prof, ctx)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ts = token_spans(tok, chat, *char_spans(chat, mem_lines))
    ids = torch.tensor([tok(chat, add_special_tokens=False).input_ids], device=DEV)
    am = torch.ones_like(ids)
    gkw = dict(max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)

    # two eager 8B models don't fit together -> generate ours, free it, then load the reference.
    im = InterventionModel(MODEL, band=range(24, 36), device=DEV, attn_implementation="eager")
    im._force_identity = True                                  # P=I
    im.prepare_for_generation(ts["mem_tok_spans"], ts["instr_tok_span"])
    with torch.no_grad():
        ours = im.model.generate(ids, attention_mask=am, **gkw)[0, ids.shape[1]:].cpu().tolist()
    del im
    torch.cuda.empty_cache()

    ref = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
    with torch.no_grad():
        vanilla = ref.generate(ids, attention_mask=am, **gkw)[0, ids.shape[1]:].cpu().tolist()

    assert ours == vanilla, f"diverged:\n ours    {ours}\n vanilla {vanilla}"
