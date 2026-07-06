"""TargetDataset: turn a self-distill targets.jsonl into training items.

Each item = the FULL-memory intervention prompt (chat-wrapped) followed by the
target letter. labels are -100 over the prompt and the target token ids over the
letter (teacher forcing); query_positions marks the letter tokens (the rows the
intervention gates); share_idx/priv_idx index into the per-attribute ordering
(= the keys of mem_tok_spans) for L_attn."""

import torch

from evaluation.cim_eval.eval import load_prompts, load_profiles
from evaluation.cim_eval.metrics import label_sets
from methods.naive.spans import build_intervention_prompt, char_spans, token_spans


class TargetDataset:
    def __init__(self, targets_jsonl, tok, prompts_file, num_profiles=7,
                 target_index=0, share_mass=0.94, private_mass=0.64):
        self.tok = tok
        P = load_prompts(prompts_file, 1)
        self.items = []
        for prof in load_profiles(targets_jsonl, num_profiles):
            for ctx in prof.get("contexts") or []:
                tgts = ctx.get("target_solutions")
                if not tgts:
                    continue
                self.items.append(self._build(P, prof, ctx, tgts[target_index], share_mass, private_mass))

    def _build(self, P, prof, ctx, target_letter, share_mass, private_mass):
        prompt, mem_lines = build_intervention_prompt(P, prof, ctx)
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        mcs, ics = char_spans(chat, mem_lines)                 # spans in the chat-wrapped text
        ts = token_spans(self.tok, chat, mcs, ics)
        prompt_ids = self.tok(chat, add_special_tokens=False).input_ids
        letter_ids = self.tok(target_letter, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        input_ids = prompt_ids + letter_ids
        labels = [-100] * len(prompt_ids) + letter_ids
        query_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        query_positions[len(prompt_ids):] = True               # the letter tokens are the query rows

        keys = list(ts["mem_tok_spans"].keys())                # attribute order for share_idx/priv_idx
        share, priv = label_sets(prof, ctx, share_mass, private_mass)
        share_s, priv_s = set(share), set(priv)
        share_idx = torch.tensor([i for i, k in enumerate(keys) if k in share_s], dtype=torch.long)
        priv_idx = torch.tensor([i for i, k in enumerate(keys) if k in priv_s], dtype=torch.long)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "mem_tok_spans": ts["mem_tok_spans"],
            "instr_tok_span": ts["instr_tok_span"],
            "query_positions": query_positions,
            "share_idx": share_idx,
            "priv_idx": priv_idx,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
