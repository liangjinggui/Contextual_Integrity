"""Locate, in the tokenized prompt, each memory attribute's token span and the
instruction span — the foundation the attention intervention (which memory tokens
to gate) and L_attn (per-attribute attention mass) build on.

The intervention prompt uses the FULL memory pool (like generate.py) — including
private rows — because the student must learn to gate private that is PRESENT in
context (unlike the teacher prompt, which physically drops private)."""

from evaluation.cim_eval.eval import info_attr_names, memory_map, mem_blob
from evaluation.cim_eval.metrics import label_sets


def build_teacher_prompt(P, profile, ctx, share_mass=0.94, private_mass=0.64):
    """oracle-clean teacher prompt：(1) 删 private 行 (2) 显式要求写 share 属性。

    与 evaluation/cim_eval/generate_targets.py 生成 y* 的 prompt 构建逐字一致
    （y* 就是从这个 prompt 生成的，teacher 前向必须用同一个输入）。"""
    names = info_attr_names(profile)
    mem = memory_map(profile)
    share, private = label_sets(profile, ctx, share_mass, private_mass)
    private_set = set(private)
    kept = [x for x in names if x not in private_set]        # (1) drop private rows
    task_prompt = P["task_solving"].format(
        task=ctx.get("task", ""),
        recipient=(ctx.get("recipient") or "").lower(),
        memories=mem_blob(kept, mem),
    )
    if share:                                                # (2) require the share attrs
        share_block = "\n".join(f"- {mem[x][0]}" for x in share if mem.get(x))
        instr = (
            "Make sure to naturally incorporate the following user information "
            "where relevant:\n" + share_block + "\n\n"
        )
        task_prompt = task_prompt.replace(
            "Only output the message", instr + "Only output the message", 1
        )
    return task_prompt


def build_intervention_prompt(P, profile, ctx):
    """(prompt_text, mem_lines) where mem_lines = [(attr_key, statement), ...] in prompt order."""
    names = info_attr_names(profile)
    mem = memory_map(profile)
    kept = [a for a in names if mem.get(a)]          # attrs that contribute a statement, in prompt order
    mem_lines = [(a, mem[a][0]) for a in kept]
    prompt = P["task_solving"].format(
        task=ctx.get("task", ""),
        recipient=(ctx.get("recipient") or "").lower(),
        memories=mem_blob(kept, mem),
    )
    return prompt, mem_lines


def char_spans(prompt, mem_lines):
    """Char (start, end) of each memory statement + the instruction line in `prompt`."""
    mem_char_spans, cursor = {}, 0
    for key, stmt in mem_lines:
        idx = prompt.find(stmt, cursor)              # statements are unique and appear in order
        assert idx >= 0, f"statement not found: {key}"
        mem_char_spans[key] = (idx, idx + len(stmt))
        cursor = idx + len(stmt)
    istart = prompt.find("Write a complete message")
    instr_char_span = (istart, len(prompt))
    return mem_char_spans, instr_char_span


def _span_to_tokens(offsets, cs, ce):
    """First..last token index whose char offset overlaps [cs, ce)."""
    toks = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs and b > a]
    return (toks[0], toks[-1] + 1) if toks else (0, 0)


def token_spans(tok, chat_text, mem_char_spans, instr_char_span):
    """Map char spans -> token spans via the fast tokenizer's offset_mapping.
    `chat_text` must be the exact string later fed to the model."""
    enc = tok(chat_text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    mem_tok_spans = {k: _span_to_tokens(offs, s, e) for k, (s, e) in mem_char_spans.items()}
    instr_tok_span = _span_to_tokens(offs, *instr_char_span)
    return {"mem_tok_spans": mem_tok_spans, "instr_tok_span": instr_tok_span, "n_tokens": len(offs)}
