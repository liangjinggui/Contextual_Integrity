"""Hyper-ICL 静态 adapter（arXiv 2606.04434 式 5-9）。

每个 (层 l, 头 h) 持有低秩因子 U_q, U_k 和门参数 w, b：
    Delta^{l,h} = (Q U_q)(K U_k)^T / sqrt(r)      # (Tq,Tk) 注意力 logit 偏置，式(6-7)
    g^{l,h}     = sigmoid( LN(Q)·w + b )          # (Tq,)  逐 query-token 强度门，式(9)
干预后的 logits: S~ = S + Diag(g)·Delta           # 式(5)，由 intervention.py 施加

初始化（论文未规定，按 LoRA 纪律）：U_k = 0 -> Delta 恒为 0 -> 打补丁后逐比特等于
base（identity 闸门）；w = b = 0 -> g 恒为 0.5，乘在零 Delta 上无影响。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperICLAdapter(nn.Module):
    def __init__(self, n_layers, n_heads, head_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.head_dim = head_dim
        # 所有层/头的参数各放一个大张量里，用 [layer] 索引出该层的 (H, ...) 切片
        self.U_q = nn.Parameter(torch.empty(n_layers, n_heads, head_dim, rank))
        self.U_k = nn.Parameter(torch.zeros(n_layers, n_heads, head_dim, rank))  # 零初始化 -> Delta=0
        self.w = nn.Parameter(torch.zeros(n_layers, n_heads, head_dim))          # 零初始化 -> g=0.5
        self.b = nn.Parameter(torch.zeros(n_layers, n_heads))
        nn.init.normal_(self.U_q, std=0.02)

    def forward(self, layer, q, k):
        """计算第 layer 层的 (Delta, g)。

        q: (B, H, Tq, D)  该层各头的 query（与算 S 用的同一份）
        k: (B, H, Tk, D)  该层各头的 key（GQA 需先 repeat_kv 到 H 个头）
        返回 delta: (B, H, Tq, Tk)，g: (B, H, Tq)
        """
        # 多卡 device_map 下各层分居不同卡：把当层参数搬到 q 所在的卡
        # （每层仅 ~几十 KB；同卡时 .to 是 no-op 零开销）
        dev = q.device
        # ---- Delta = (q U_q)(k U_k)^T / sqrt(r)，式(6-7) ----
        q_low = q @ self.U_q[layer].to(dev)                    # (B, H, Tq, r)
        k_low = k @ self.U_k[layer].to(dev)                    # (B, H, Tk, r)
        delta = q_low @ k_low.transpose(-1, -2)                # (B, H, Tq, Tk)
        delta = delta / math.sqrt(self.rank)

        # ---- g = sigmoid( LN(q)·w + b )，式(9) ----
        q_norm = F.layer_norm(q, (self.head_dim,))             # LN 只归一化，无可学参数
        w = self.w[layer].to(dev).unsqueeze(1)                 # (H, 1, D)，与 (B,H,Tq,D) 广播
        b = self.b[layer].to(dev).unsqueeze(-1)                # (H, 1)
        g = torch.sigmoid((q_norm * w).sum(-1) + b)            # (B, H, Tq)
        return delta, g
