"""方案2 attention intervention: replace the generated-token -> memory-token score
with the projected bilinear (P_Q·q_base)·(P_K·k_base). Everything else (query->query,
query->non-memory) stays vanilla; ctx=None or P_Q=None => byte-identical to
transformers' eager_attention_forward (the identity/off switch)."""

import torch
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, eager_attention_forward


def naive_eager_attention(module, query, key, value, attention_mask, scaling, ctx=None, dropout=0.0, **kw):
    if ctx is None or ctx.get("P_Q") is None:
        return eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling      # (B,H,S,S) base logits

    P_Q, P_K = ctx["P_Q"].to(query.dtype), ctx["P_K"].to(query.dtype)     # (M,D,D)
    pos = ctx["mem_key_positions"]                                        # (nmem,) key positions
    grp = ctx["mem_key_group"]                                           # (nmem,) attribute idx in [0,M)
    qmask = ctx["query_positions"]                                        # (S,) bool
    k_mem = key_states[:, :, pos, :]                                      # (B,H,nmem,D)
    PQ, PK = P_Q[grp], P_K[grp]                                           # (nmem,D,D) per memory token
    qP = torch.einsum("bhsd,nde->bhnse", query, PQ)                       # (B,H,nmem,S,D)
    kP = torch.einsum("bhnd,nde->bhne", k_mem, PK)                        # (B,H,nmem,D)
    proj = torch.einsum("bhnse,bhne->bhsn", qP, kP) * scaling             # (B,H,S,nmem) projected score

    new_attn = attn.clone()
    rows = qmask.nonzero(as_tuple=True)[0]                                # generated/query rows to modify
    new_attn[:, :, rows[:, None], pos[None, :]] = proj[:, :, rows, :]     # write back on query x memory
    if attention_mask is not None:
        new_attn = new_attn + attention_mask[:, :, :, : key_states.shape[-2]]
    w = torch.softmax(new_attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w
