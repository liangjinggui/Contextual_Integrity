"""把 HyperICLAdapter 接进冻结的 Qwen3：S~ = S + Diag(g)·Delta（式5，全层全头全行）。

attach_adapter(model, rank) 给模型每一层的 self_attn 换上带偏置的 forward：
q/k/v 的计算与 base 完全一样（QK-norm -> RoPE -> KV cache），打分与 softmax 交给
F.scaled_dot_product_attention（SDPA）一次融合内核算完——式(5) 的 Diag(g)·Delta
连同因果 mask 一起作为 additive attn_mask 传入，数学上与手写 eager 完全等价：
    SDPA(q,k,v, attn_mask=B) == softmax(qk^T*scale + B) v
（手写 eager 版生成时 36 层 x 每 token 十几个小算子，单 context ~8 分钟，不可用；
SDPA 版一次内核，快一个量级。浮点求和顺序不同 -> 与 eager 数学相等但非逐比特。）

adapter 恒等初始化（U_k=0）时 bias 里只剩因果 mask，前向等价于 base；
adapter.enabled=False 是硬开关。生成天然兼容 KV cache：decode 步 q 只有
1 个新 token，其 g 和 Delta 行现算，key 侧用 cache 里的全部 key。
"""

import types

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb

from methods.hyper_icl.adapter import HyperICLAdapter


def _patched_forward(adapter, layer_idx):
    """构造第 layer_idx 层 self_attn 的替换 forward（绑定到该层模块上）。"""

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        B, T, _ = hidden_states.shape

        # ---- q/k/v：与 base 完全相同（QK-norm -> RoPE -> KV cache）----
        shape = (B, T, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)   # (B, H, T, D)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)   # (B, Hkv, T, D)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)                # (B, Hkv, T, D)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        k_all = repeat_kv(k, self.num_key_value_groups)                            # (B, H, Tk, D)
        v_all = repeat_kv(v, self.num_key_value_groups)                            # (B, H, Tk, D)

        # ---- 组装加性 bias = 因果 mask + Diag(g)·Delta（式5）----
        # attach_adapter 强制 eager 配置，HF 必给显式 4D float mask；若为 None 宁可
        # 炸也不能静默双向（三跑教训）。
        assert attention_mask is not None, "expected an explicit 4D causal mask (eager config)"
        bias = attention_mask[:, :, :, : k_all.shape[-2]]                          # (B, 1, T, Tk)
        if adapter.enabled:
            delta, g = adapter(layer_idx, q, k_all)                # (B,H,T,Tk), (B,H,T)
            bias = bias + g.unsqueeze(-1) * delta                  # Diag(g)·Delta：第 i 行乘 g_i

        # ---- 一次 SDPA 融合内核：softmax(qk^T*scale + bias) v ----
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias, scale=self.scaling)
        out = out.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(out), None

    return forward


def attach_adapter(model, rank=4):
    """给 model（Qwen3ForCausalLM）每一层挂上 adapter，返回 HyperICLAdapter。

    base 权重全部冻结；只有返回的 adapter 含可训练参数。"""
    cfg = model.config
    # 必须强制 eager：sdpa 配置下 HF 对无 padding 输入给 attention_mask=None（SDPA 内部
    # 用 is_causal 标志），我们的 forward 就没有因果掩码 -> 双向注意力（信件 token 能看
    # 到未来，teacher-forcing 变抄答案）。eager 配置保证 HF 总是准备显式 4D 因果掩码。
    cfg._attn_implementation = "eager"
    layers = model.model.layers
    adapter = HyperICLAdapter(
        n_layers=len(layers), n_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim, rank=rank,
    )
    param = next(model.parameters())
    adapter.to(device=param.device, dtype=param.dtype)
    adapter.enabled = True

    model.requires_grad_(False)
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        attn.forward = types.MethodType(_patched_forward(adapter, layer_idx), attn)
    return adapter
