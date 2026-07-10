"""HyperICLDataset：每个 context 出一条 teacher/student 配对样本。

student = 完整 memory prompt（含 private）+ y*   -> 施加 adapter、算 L_sup
teacher = 干净 prompt（删 private + share 指令）+ y*  -> 冻结前向、供 L_H-anchor 对齐
两边追加同一份 y* token；信件 = 各自序列的最后 n_letter 个位置（anchor 按此对齐）。

hyper-icl 是全行干预，不需要 memory/指令的 token span（比 naive 的数据简单很多）。"""

import torch

from evaluation.cim_eval.eval import load_prompts, load_profiles
from methods.utils.spans import build_intervention_prompt, build_teacher_prompt


class HyperICLDataset:
    def __init__(self, targets_jsonl, tok, prompts_file, num_profiles=7, target_index=0):
        self.items = []
        P = load_prompts(prompts_file, 1)

        def encode_chat(prompt_text):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            return tok(chat, add_special_tokens=False).input_ids

        for prof in load_profiles(targets_jsonl, num_profiles):
            for ctx in prof.get("contexts") or []:
                targets = ctx.get("target_solutions")
                if not targets:
                    continue
                letter_ids = tok(targets[target_index], add_special_tokens=False).input_ids
                letter_ids = letter_ids + [tok.eos_token_id]

                student_prompt, _ = build_intervention_prompt(P, prof, ctx)
                # v1 teacher：无 share 指令块 -> teacher/student 只差 private 行
                # （变体探针显示 anchor 可学空间比 v0 大一倍；y* 不变，仅表示参照变了）
                teacher_prompt = build_teacher_prompt(P, prof, ctx, share_instruction=False)
                student_prompt_ids = encode_chat(student_prompt)
                teacher_prompt_ids = encode_chat(teacher_prompt)

                student_ids = student_prompt_ids + letter_ids
                labels = [-100] * len(student_prompt_ids) + letter_ids
                self.items.append({
                    "student_ids": torch.tensor(student_ids),
                    "student_labels": torch.tensor(labels),
                    "teacher_ids": torch.tensor(teacher_prompt_ids + letter_ids),
                    "n_letter": len(letter_ids),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
