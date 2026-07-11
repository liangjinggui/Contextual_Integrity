"""hyper_ctx intervention 单测(tiny 随机 Qwen3, CPU)。

闸门:恒等初始化下打补丁==裸模型(前向 allclose + 贪心生成逐 token 相等);
上下文条件性穿透到 logits;U 的 prefill 缓存行为(prefill 算一次, decode 只复用);
enabled/默认配置回归(继承 hyper_icl 的教训)。"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from methods.hyper_ctx.intervention import attach_ctx_adapter

TINY = Qwen3Config(
    vocab_size=128, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=256,
)
INSTR = (2, 6)      # 玩具指令 span:位置 2..6


def make_pair():
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(TINY).eval()
    base.config._attn_implementation = "eager"
    patched = Qwen3ForCausalLM(TINY).eval()          # 不设 eager,靠 attach 内部兜底(回归)
    patched.load_state_dict(base.state_dict())
    adapter = attach_ctx_adapter(patched, rank=2)
    return base, patched, adapter


def test_forward_identity_at_init():
    base, patched, adapter = make_pair()
    ids = torch.randint(0, 128, (1, 10))
    adapter.set_context(INSTR)
    with torch.no_grad():
        out_b = base(ids).logits
        out_p = patched(ids).logits
    assert torch.allclose(out_b, out_p, atol=1e-4, rtol=1e-4)


def test_generate_identity_at_init():
    """贪心生成逐 token 相等(带 KV cache + U 的 prefill 缓存路径)。"""
    base, patched, adapter = make_pair()
    ids = torch.randint(0, 128, (1, 10))
    adapter.prepare_for_generation(INSTR)
    with torch.no_grad():
        gen_b = base.generate(ids, max_new_tokens=8, do_sample=False)
        gen_p = patched.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(gen_b, gen_p)


def test_context_conditioning_reaches_logits():
    """head 权重非零后:同一输入、不同指令 span(不同 c)必须给出不同 logits。"""
    _, patched, adapter = make_pair()
    with torch.no_grad():
        adapter.gtheta.head.weight.normal_(0, 0.05)
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        adapter.set_context((2, 6)); out1 = patched(ids).logits
        adapter.set_context((6, 9)); out2 = patched(ids).logits
    assert not torch.allclose(out1, out2)


def test_prefill_caches_U_decode_reuses():
    """生成模式:prefill 每层调一次 Gθ 并缓存;decode 步不再调 Gθ。"""
    _, patched, adapter = make_pair()
    calls = {"n": 0}
    orig = adapter.gtheta.forward
    adapter.gtheta.forward = lambda c, l: (calls.__setitem__("n", calls["n"] + 1), orig(c, l))[1]
    ids = torch.randint(0, 128, (1, 10))
    adapter.prepare_for_generation(INSTR)
    with torch.no_grad():
        patched.generate(ids, max_new_tokens=6, do_sample=False)
    assert calls["n"] == TINY.num_hidden_layers        # 只在 prefill 每层一次
    assert len(adapter._U_cache) == TINY.num_hidden_layers


def test_enabled_flag():
    """enabled=False 时即使 head 权重随机也回到 base(硬开关)。"""
    base, patched, adapter = make_pair()
    with torch.no_grad():
        adapter.gtheta.head.weight.normal_(0, 0.5)
    adapter.enabled = False
    ids = torch.randint(0, 128, (1, 10))
    adapter.set_context(INSTR)
    with torch.no_grad():
        assert torch.allclose(base(ids).logits, patched(ids).logits, atol=1e-4, rtol=1e-4)
