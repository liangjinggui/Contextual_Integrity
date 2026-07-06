import json
from transformers import AutoTokenizer
from evaluation.cim_eval.eval import load_prompts, load_profiles
from methods.naive.spans import build_intervention_prompt, char_spans, token_spans

DATA = "data/CIMemories/data/data_openai_gpt-oss-120b_gold_labelled_personas_gemini-3-flash-preview_10profiles_combined_train7.json"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


def test_spans_decode_to_statements():
    P = load_prompts(PROMPTS, 1)
    prof = load_profiles(DATA, 1)[0]
    ctx = prof["contexts"][0]
    prompt, mem_lines = build_intervention_prompt(P, prof, ctx)
    assert len(mem_lines) >= 100  # ~147 attrs
    mcs, ics = char_spans(prompt, mem_lines)
    # every memory statement's char span slices back to that statement
    for key, (s, e) in mcs.items():
        stmt = dict(mem_lines)[key]
        assert prompt[s:e] == stmt
    # instruction span contains the task verb
    assert "Write a complete message" in prompt[ics[0]:ics[1]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ts = token_spans(tok, prompt, mcs, ics)
    # a memory token span, decoded, contains its statement's first word
    any_key = next(iter(ts["mem_tok_spans"]))
    t0, t1 = ts["mem_tok_spans"][any_key]
    dec = tok.decode(tok(prompt).input_ids[t0:t1])
    assert dict(mem_lines)[any_key].split()[0] in dec
