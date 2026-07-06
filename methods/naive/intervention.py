"""方案2 attention intervention: replace the generated-token -> memory-token score
with the projected bilinear (P_Q·q_base)·(P_K·k_base). Everything else (query->query,
query->non-memory) stays vanilla; ctx=None or P_Q=None => byte-identical to
transformers' eager_attention_forward (the identity/off switch)."""

import types

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, eager_attention_forward, apply_rotary_pos_emb

from methods.naive.hypernet import Gtheta


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
    # proj[s,n] = (P_Q q_s)·(P_K k_n) = q_sᵀ (P_Qᵀ P_K) k_n. Fold into one bilinear per
    # group and contract k first -> avoids the (nmem,S,D) blowup.
    B_n = torch.matmul(P_Q.transpose(-1, -2), P_K)[grp]                   # (nmem,D,D): P_Qᵀ P_K per token
    kB = torch.einsum("nde,bhne->bhnd", B_n, k_mem)                       # (B,H,nmem,D)
    proj = torch.einsum("bhsd,bhnd->bhsn", query, kB) * scaling           # (B,H,S,nmem) projected score

    new_attn = attn.clone()
    rows = qmask.nonzero(as_tuple=True)[0]                                # generated/query rows to modify
    new_attn[:, :, rows[:, None], pos[None, :]] = proj[:, :, rows, :]     # write back on query x memory
    if attention_mask is not None:
        new_attn = new_attn + attention_mask[:, :, :, : key_states.shape[-2]]
    w = torch.softmax(new_attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w


def _patched_attn_forward(im, layer_idx):
    """Bound-method replacement for a band layer's Qwen3Attention.forward: same q/k/v
    (q_norm->RoPE) as base, but builds the intervention ctx from the pooled layer-input
    hidden states and routes through naive_eager_attention."""

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        ctx = im._build_ctx(hidden_states, layer_idx)
        attn_output, attn_weights = naive_eager_attention(
            self, query_states, key_states, value_states, attention_mask,
            scaling=self.scaling, ctx=ctx,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        if ctx is not None:
            im._record_masses(layer_idx, attn_weights, ctx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


class InterventionModel(nn.Module):
    """Frozen Qwen3 base + trainable Gtheta. Band layers' attention is monkeypatched to
    project base q/k into a Gtheta-emitted subspace on generated->memory pairs. At init
    (Gtheta -> P=I) the whole forward equals the base forward (off switch / warm start)."""

    def __init__(self, model_path, band=range(24, 36), rank=8, dtype="bfloat16", device="cuda:0",
                 attn_implementation="eager"):
        super().__init__()
        self.device = device
        td = getattr(torch, dtype)
        # non-band layers use `attn_implementation` (sdpa saves memory in training); band layers
        # are always monkeypatched to naive_eager_attention (needs the scores).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=td, attn_implementation=attn_implementation
        ).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        cfg = self.model.config
        self.band = list(band)
        self._band_pos = {ell: i for i, ell in enumerate(self.band)}   # layer idx -> Gtheta layer_idx
        self.gtheta = Gtheta(hidden_size=cfg.hidden_size, head_dim=cfg.head_dim,
                             rank=rank, n_layers=len(self.band)).to(device).to(td)
        self._ctx_spans = None
        self._force_identity = False   # if True, _build_ctx uses P=I (for vanilla/base masses)
        self.last_masses = {}
        for ell in self.band:
            attn = self.model.model.layers[ell].self_attn
            attn.forward = types.MethodType(_patched_attn_forward(self, ell), attn)

    def set_context(self, mem_tok_spans_per_attr, instr_tok_span, query_positions):
        self._ctx_spans = {"mem": mem_tok_spans_per_attr, "instr": instr_tok_span,
                           "qpos": query_positions.to(self.device)}

    def clear_context(self):
        self._ctx_spans = None

    def _build_ctx(self, hidden_states, layer_idx):
        sp = self._ctx_spans
        if sp is None or int(sp["qpos"].sum()) == 0 or not sp["mem"]:
            return None                                   # no query rows / no memory -> vanilla (exact)
        keys = list(sp["mem"].keys())
        gd = self.gtheta.head.weight.dtype
        i0, i1 = sp["instr"]
        if self._force_identity:                          # vanilla masses: P=I, no gtheta
            D = self.model.config.head_dim
            I = torch.eye(D, device=hidden_states.device, dtype=hidden_states.dtype)
            P_Q = P_K = I.expand(len(keys), D, D)
        else:
            c = hidden_states[:, i0:i1, :].mean(1)        # (B, hidden), B=1
            es = [hidden_states[:, s:e, :].mean(1) for (s, e) in (sp["mem"][k] for k in keys)]
            e = torch.cat(es, 0)                          # (M, hidden)
            P_Q, P_K = self.gtheta(c.expand(len(keys), -1).to(gd), e.to(gd), self._band_pos[layer_idx])
        positions, groups = [], []
        for m, k in enumerate(keys):
            t0, t1 = sp["mem"][k]
            positions += list(range(t0, t1)); groups += [m] * (t1 - t0)
        dev = hidden_states.device
        return {"P_Q": P_Q, "P_K": P_K,
                "mem_key_positions": torch.tensor(positions, device=dev),
                "mem_key_group": torch.tensor(groups, device=dev),
                "query_positions": sp["qpos"].to(dev)}

    def _record_masses(self, layer_idx, w, ctx):
        qrows = ctx["query_positions"].nonzero(as_tuple=True)[0]
        if len(qrows) == 0:
            return
        pos, grp = ctx["mem_key_positions"], ctx["mem_key_group"]
        wq = w[:, :, qrows][:, :, :, pos].float()         # (B,H,nq,nmem)
        M = ctx["P_Q"].shape[0]
        masses = torch.stack([wq[:, :, :, grp == m].sum(-1).mean() for m in range(M)])
        self.last_masses[layer_idx] = masses

    def forward(self, input_ids, attention_mask=None):
        self.last_masses = {}
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
