"""把 CtxHyperNet 接进冻结的 Qwen3:机制与 hyper_icl 完全相同(S̃=S+Diag(g)Δ,
SDPA 融合内核,全行干预),唯一区别是 U_q/U_k 来自 Gθ(c^ℓ) 而非静态参数。

c^ℓ 的来源分两种模式(由 CtxAdapter 的状态切换):
  训练  set_context(instr_span):每层 forward 就地从本层输入隐状态池化 c^ℓ -> Gθ -> U。
  生成  prepare_for_generation(instr_span):prefill(T>1)时算一次并缓存每层的 U;
        decode(T==1)只复用缓存(c 是 context 固定的, 缓存=精确复用)。门 g 每步现算。

安全纪律(全部继承 hyper_icl 的教训):attach 内部强制 eager 配置(sdpa 配置下
HF 给 attention_mask=None -> 静默双向);forward 里 assert mask 非空。
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb

from methods.hyper_ctx.hypernet import CtxHyperNet


class CtxAdapter(nn.Module):
    """Gθ + 静态门参数 + 模式状态。可训练参数 = gtheta + w + b;base 冻结。"""

    def __init__(self, hidden_size, n_layers, n_heads, head_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.head_dim = head_dim
        self.gtheta = CtxHyperNet(hidden_size, n_layers, n_heads, head_dim, rank)
        self.w = nn.Parameter(torch.zeros(n_layers, n_heads, head_dim))   # 门:零初始化 -> g=0.5
        self.b = nn.Parameter(torch.zeros(n_layers, n_heads))
        self.enabled = True
        self._instr_span = None        # (i0, i1) 指令 token 区间
        self._gen_mode = False
        self._U_cache = {}             # layer_idx -> (U_q, U_k)  仅生成模式用

    def set_context(self, instr_span):
        """训练:每次前向都从当层隐状态现算 c^ℓ -> U。"""
        self._instr_span = instr_span
        self._gen_mode = False
        self._U_cache = {}

    def clear_context(self):
        self._instr_span = None
        self._gen_mode = False
        self._U_cache = {}

    def prepare_for_generation(self, instr_span):
        """生成:prefill 算 U 并缓存, decode 复用。每个 context 调一次。"""
        self._instr_span = instr_span
        self._gen_mode = True
        self._U_cache = {}

    def get_U(self, hidden_states, layer_idx):
        """返回该层的 (U_q, U_k), 各 (B, H, D, r);无 span 时返回 None(走纯 base)。"""
        if self._instr_span is None:
            return None
        gw = self.gtheta.head.weight
        gd, gdev = gw.dtype, gw.device            # 多卡切分时后半段层的 c 在别的卡, 搬到 Gθ 的卡
        if self._gen_mode:
            T = hidden_states.shape[1]
            if T > 1:                                          # prefill:算一次并缓存
                i0, i1 = self._instr_span
                c = hidden_states[:, i0:i1, :].mean(1)         # (B, hidden)
                self._U_cache[layer_idx] = self.gtheta(c.to(device=gdev, dtype=gd), layer_idx)
                return self._U_cache[layer_idx]
            return self._U_cache.get(layer_idx)                # decode:复用(未缓存则 None)
        i0, i1 = self._instr_span                              # 训练:就地现算
        c = hidden_states[:, i0:i1, :].mean(1)
        return self.gtheta(c.to(device=gdev, dtype=gd), layer_idx)

    def gate(self, q, layer_idx):
        """g = sigmoid(LN(q)·w + b): (B, H, T)。w/b 是静态参数(与 hyper_icl 相同)。"""
        dev = q.device
        q_norm = F.layer_norm(q, (self.head_dim,))
        w = self.w[layer_idx].to(dev).unsqueeze(1)             # (H, 1, D)
        b = self.b[layer_idx].to(dev).unsqueeze(-1)            # (H, 1)
        return torch.sigmoid((q_norm * w).sum(-1) + b)


def _patched_forward(adapter, layer_idx):
    """构造第 layer_idx 层 self_attn 的替换 forward(与 hyper_icl 版同构)。"""

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        B, T, _ = hidden_states.shape

        # ---- q/k/v:与 base 完全相同(QK-norm -> RoPE -> KV cache)----
        shape = (B, T, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)   # (B, H, T, D)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        k_all = repeat_kv(k, self.num_key_value_groups)                            # (B, H, Tk, D)
        v_all = repeat_kv(v, self.num_key_value_groups)

        # ---- bias = 因果 mask + Diag(g)·Δ,Δ 的 U 来自 Gθ(c^ℓ) ----
        assert attention_mask is not None, "expected an explicit 4D causal mask (eager config)"
        bias = attention_mask[:, :, :, : k_all.shape[-2]]                          # (B, 1, T, Tk)
        U = adapter.get_U(hidden_states, layer_idx) if adapter.enabled else None
        if U is not None:
            U_q, U_k = U                                       # 各 (B, H, D, r)
            U_q = U_q.to(device=q.device, dtype=q.dtype)       # Gθ 在首卡, U 搬回本层所在卡
            U_k = U_k.to(device=q.device, dtype=q.dtype)
            q_low = q @ U_q                                    # (B, H, T,  r)
            k_low = k_all @ U_k                                # (B, H, Tk, r)
            delta = q_low @ k_low.transpose(-1, -2) / math.sqrt(adapter.rank)      # (B, H, T, Tk)
            g = adapter.gate(q, layer_idx)                     # (B, H, T)
            bias = bias + g.unsqueeze(-1) * delta

        if bias.requires_grad:
            # 训练:efficient 后端对带梯度的 attn_mask 反传有 LSE 对齐 bug -> 用 math 后端
            # (数学等价的朴素矩阵实现;生成 no_grad 不进这支,照旧走高效内核)
            with sdpa_kernel([SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias, scale=self.scaling)
        else:
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias, scale=self.scaling)
        out = out.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(out), None

    return forward


def attach_ctx_adapter(model, rank=4):
    """冻结 base、每层挂上下文条件 adapter,返回 CtxAdapter(唯一可训练模块)。"""
    cfg = model.config
    # 强制 eager 配置:保证 HF 总是准备显式 4D 因果 mask(hyper_icl 三跑教训)
    cfg._attn_implementation = "eager"
    layers = model.model.layers
    adapter = CtxAdapter(hidden_size=cfg.hidden_size, n_layers=len(layers),
                         n_heads=cfg.num_attention_heads, head_dim=cfg.head_dim, rank=rank)
    param = next(model.parameters())
    adapter.to(device=param.device, dtype=param.dtype)
    model.requires_grad_(False)
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        attn.forward = types.MethodType(_patched_forward(adapter, layer_idx), attn)
    return adapter
