"""HyperICLDataset 单测（CPU，只用 tokenizer）。

teacher = 干净 prompt（删 private + share 指令，与生成 y* 的 prompt 逐字一致）+ y*；
student = 完整 memory prompt + y*。两边追加的 y* token 必须一模一样（anchor 对齐的前提）。"""

import torch
from transformers import AutoTokenizer

from methods.hyper_icl.data import HyperICLDataset

TARGETS = "outputs/cimemories/self_distill_targets/Qwen3-8B/20260705_225937/train/targets.jsonl"
PROMPTS = "data/CIMemories/eval/prompts.yaml"


def make():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ds = HyperICLDataset(TARGETS, tok, PROMPTS, num_profiles=1)
    return tok, ds


def test_basic_shapes_and_alignment():
    tok, ds = make()
    assert len(ds) > 0
    item = ds[0]
    n = item["n_letter"]
    assert n > 0
    # 信件尾部：student 和 teacher 的最后 n 个 token 必须一致（同一份 y*）
    assert torch.equal(item["student_ids"][-n:], item["teacher_ids"][-n:])
    # teacher prompt 删了 private 行 -> 一定更短
    assert len(item["teacher_ids"]) < len(item["student_ids"])
    # labels：prompt 区 -100，信件区等于 student_ids 的对应位置
    labels = item["student_labels"]
    assert (labels[:-n] == -100).all()
    assert torch.equal(labels[-n:], item["student_ids"][-n:])


def test_letter_ends_with_eos():
    tok, ds = make()
    item = ds[0]
    assert item["student_ids"][-1].item() == tok.eos_token_id
    assert item["teacher_ids"][-1].item() == tok.eos_token_id
