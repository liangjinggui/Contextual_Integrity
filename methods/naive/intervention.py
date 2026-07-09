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
        if ctx is not None and not im._gen_mode:          # masses are a training-only signal
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
        self._gen_mode = False         # generation: compute P once at prefill, cache, reuse at decode
        self._P_cache = {}             # layer_idx -> (P_Q, P_K, mem_key_positions, mem_key_group)
        self.last_masses = {}
        for ell in self.band:
            attn = self.model.model.layers[ell].self_attn
            attn.forward = types.MethodType(_patched_attn_forward(self, ell), attn)

    def set_context(self, mem_tok_spans_per_attr, instr_tok_span, query_positions):
        """Training: query rows (letter tokens) are present in the single teacher-forced forward."""
        self._ctx_spans = {"mem": mem_tok_spans_per_attr, "instr": instr_tok_span,
                           "qpos": query_positions.to(self.device)}
        self._gen_mode = False

    def prepare_for_generation(self, mem_tok_spans_per_attr, instr_tok_span):
        """Generation: call once per context before model.generate(). P is context-fixed, so it is
        computed from the prompt at PREFILL and cached per band layer; every DECODE step reuses it
        (query rows aren't known ahead of time -> no qpos here)."""
        self._ctx_spans = {"mem": mem_tok_spans_per_attr, "instr": instr_tok_span, "qpos": None}
        self._gen_mode = True
        self._P_cache = {}

    def clear_context(self):
        self._ctx_spans = None
        self._gen_mode = False
        self._P_cache = {}

    def _emit_P(self, hidden_states, keys, layer_idx):
        """Pool instruction + per-memory hidden states -> Gtheta -> (P_Q, P_K), each (M,D,D).
        hidden_states is one row (B=1); _force_identity short-circuits to P=I (vanilla masses)."""
        if self._force_identity:
            D = self.model.config.head_dim
            I = torch.eye(D, device=hidden_states.device, dtype=hidden_states.dtype)
            return I.expand(len(keys), D, D), I.expand(len(keys), D, D)
        sp = self._ctx_spans
        gd = self.gtheta.head.weight.dtype
        i0, i1 = sp["instr"]
        c = hidden_states[:, i0:i1, :].mean(1)            # (1, hidden) instruction repr c^l
        es = [hidden_states[:, s:e, :].mean(1) for (s, e) in (sp["mem"][k] for k in keys)]
        e = torch.cat(es, 0)                              # (M, hidden) per-memory repr e_m^l
        return self.gtheta(c.expand(len(keys), -1).to(gd), e.to(gd), self._band_pos[layer_idx])

    def _flatten_spans(self, keys, dev):
        """Per-attribute spans -> flat per-token (positions, group idx). group[j] = which of the M
        attributes memory-token j belongs to, so naive_eager_attention can index P by memory token."""
        positions, groups = [], []
        for m, k in enumerate(keys):
            t0, t1 = self._ctx_spans["mem"][k]
            positions += list(range(t0, t1)); groups += [m] * (t1 - t0)
        return (torch.tensor(positions, device=dev), torch.tensor(groups, device=dev))

    def _build_ctx(self, hidden_states, layer_idx):
        sp = self._ctx_spans
        if sp is None or not sp["mem"]:
            return None
        if self._gen_mode:
            return self._build_ctx_gen(hidden_states, layer_idx)
        if int(sp["qpos"].sum()) == 0:                    # training: no query rows -> vanilla (exact)
            return None
        keys = list(sp["mem"].keys())
        P_Q, P_K = self._emit_P(hidden_states, keys, layer_idx)
        pos, grp = self._flatten_spans(keys, hidden_states.device)
        return {"P_Q": P_Q, "P_K": P_K, "mem_key_positions": pos, "mem_key_group": grp,
                "query_positions": sp["qpos"].to(hidden_states.device)}

    def _build_ctx_gen(self, hidden_states, layer_idx):
        """PREFILL (S>1): compute P from the prompt row, cache it per band layer, return None so the
        prompt attends vanilla (prompt tokens aren't generated tokens). DECODE (S==1): reuse cached P
        with the single new token as the sole query row. Memory key positions are absolute prompt
        positions and stay valid across decode steps (the KV cache preserves order)."""
        keys = list(self._ctx_spans["mem"].keys())
        dev = hidden_states.device
        if hidden_states.shape[1] > 1:
            P_Q, P_K = self._emit_P(hidden_states[0:1], keys, layer_idx)   # replicas identical -> row 0
            pos, grp = self._flatten_spans(keys, dev)
            self._P_cache[layer_idx] = (P_Q, P_K, pos, grp)
            return None
        if layer_idx not in self._P_cache:
            return None
        P_Q, P_K, pos, grp = self._P_cache[layer_idx]
        return {"P_Q": P_Q, "P_K": P_K, "mem_key_positions": pos, "mem_key_group": grp,
                "query_positions": torch.ones(1, dtype=torch.bool, device=dev)}

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
