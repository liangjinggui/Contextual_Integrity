import torch
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
from methods.naive.intervention import naive_eager_attention


class _Mod:  # minimal stand-in for the attention module attrs eager uses
    num_key_value_groups = 4
    training = False


def _qkv(B=1, Hq=8, Hkv=2, S=6, D=4):
    return (torch.randn(B, Hq, S, D), torch.randn(B, Hkv, S, D), torch.randn(B, Hkv, S, D))


def test_identity_ctx_matches_vanilla():
    q, k, v = _qkv(); m = _Mod(); mask = None
    o0, w0 = eager_attention_forward(m, q, k, v, mask, scaling=0.5)
    o1, w1 = naive_eager_attention(m, q, k, v, mask, scaling=0.5, ctx=None)
    assert torch.allclose(o0, o1, atol=1e-6) and torch.allclose(w0, w1, atol=1e-6)


def test_P_identity_matches_vanilla():
    q, k, v = _qkv(); m = _Mod(); D = q.shape[-1]
    ctx = {"P_Q": torch.eye(D).expand(2, D, D), "P_K": torch.eye(D).expand(2, D, D),
           "mem_key_positions": torch.tensor([1, 2]), "mem_key_group": torch.tensor([0, 1]),
           "query_positions": torch.tensor([False, False, False, True, True, True])}
    o0, _ = eager_attention_forward(m, q, k, v, None, scaling=0.5)
    o1, _ = naive_eager_attention(m, q, k, v, None, scaling=0.5, ctx=ctx)
    assert torch.allclose(o0, o1, atol=1e-5)


def test_negative_score_suppresses_memory():
    # Suppression comes from a very NEGATIVE projected score, not a zero one (score 0 is a
    # neutral logit, not suppression). With q=k=1 (q·k>0), a large-negative P_K drives the
    # memory score hugely negative -> softmax weight -> 0.
    m = _Mod(); D = 4
    q = torch.ones(1, 8, 6, D)      # query = +1
    k = torch.ones(1, 2, 6, D)      # keys  = +1 -> base q·k = D > 0
    v = torch.randn(1, 2, 6, D)
    ctx = {"P_Q": torch.eye(D).unsqueeze(0), "P_K": (-1e4 * torch.eye(D)).unsqueeze(0),
           "mem_key_positions": torch.tensor([1]), "mem_key_group": torch.tensor([0]),
           "query_positions": torch.tensor([False, False, False, True, True, True])}
    _, w = naive_eager_attention(m, q, k, v, None, scaling=0.5, ctx=ctx)
    assert w[:, :, 3:, 1].max() < 1e-3   # generated rows give ~0 weight to memory token 1
