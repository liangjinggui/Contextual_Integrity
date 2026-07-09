"""intervention 单测（tiny 随机 Qwen3, CPU）。

闸门：adapter 恒等初始化（U_k=0）时，打补丁的模型必须与原始模型等价——
前向 logits allclose（补丁走 SDPA 融合内核，与 eager 数学相等、非逐比特），
带 KV cache 的贪心生成逐 token 一致；U_k 随机化后输出必须改变（非空）。"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from methods.hyper_icl.intervention import attach_adapter

TINY = Qwen3Config(
    vocab_size=128, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=256,
)


def make_pair():
    """同一份权重的两个 tiny 模型：base 原样，patched 挂 adapter。都用 eager。"""
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(TINY).eval()
    base.config._attn_implementation = "eager"
    patched = Qwen3ForCausalLM(TINY).eval()
    patched.config._attn_implementation = "eager"
    patched.load_state_dict(base.state_dict())
    adapter = attach_adapter(patched, rank=2)
    return base, patched, adapter


def test_forward_identity_at_init():
    base, patched, _ = make_pair()
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_base = base(ids).logits
        out_patched = patched(ids).logits
    assert torch.allclose(out_base, out_patched, atol=1e-4, rtol=1e-4)


def test_generate_identity_at_init():
    """带 KV cache 的贪心生成也逐 token 一致（预验 decode 路径）。"""
    base, patched, _ = make_pair()
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        gen_base = base.generate(ids, max_new_tokens=8, do_sample=False)
        gen_patched = patched.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(gen_base, gen_patched)


def test_random_Uk_changes_output():
    base, patched, adapter = make_pair()
    with torch.no_grad():
        adapter.U_k.normal_(0, 0.5)
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_base = base(ids).logits
        out_patched = patched(ids).logits
    assert not torch.equal(out_base, out_patched)


def test_identity_with_default_attn_impl():
    """回归（真实翻车场景）：patched 模型不手动设 eager（默认 sdpa 时 HF 给
    attention_mask=None -> 补丁里没有因果掩码 -> 双向注意力、信件 token 看到未来）。
    attach_adapter 内部必须强制 eager，保证任何加载方式下 identity 都成立。"""
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(TINY).eval()
    base.config._attn_implementation = "eager"
    patched = Qwen3ForCausalLM(TINY).eval()          # 不设 _attn_implementation，走默认
    patched.load_state_dict(base.state_dict())
    attach_adapter(patched, rank=2)
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        assert torch.allclose(base(ids).logits, patched(ids).logits, atol=1e-4, rtol=1e-4)
        gen_b = base.generate(ids, max_new_tokens=8, do_sample=False)
        gen_p = patched.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(gen_b, gen_p)


def test_enabled_flag():
    """enabled=False 时即使 U_k 随机也精确回到 base（硬开关）。"""
    base, patched, adapter = make_pair()
    with torch.no_grad():
        adapter.U_k.normal_(0, 0.5)
    adapter.enabled = False
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_base = base(ids).logits
        out_patched = patched(ids).logits
    assert torch.allclose(out_base, out_patched, atol=1e-4, rtol=1e-4)
