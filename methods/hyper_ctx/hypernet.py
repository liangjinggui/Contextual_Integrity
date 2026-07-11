"""CtxHyperNet:按 context 现场生成 Hyper-ICL 的低秩因子 U_q/U_k。

与静态版(methods/hyper_icl/adapter.py)的唯一区别:U_q/U_k 不再是固定参数,
而是 Gθ(c^ℓ, 层嵌入) 的输出——每个 context 一套、每层现算。机制其余不变:
    Δ = (Q·U_q)(K·U_k)ᵀ/√r,  S̃ = S + Diag(g)·Δ  (门 g 的 w,b 仍是静态参数, 在接线处)

输入 c^ℓ = 该层指令 span 的池化隐状态(4096 维)——编码 task+recipient,
即决定隐私规范的上下文;memory 内容不进 Gθ,它在打分时经 k_j 进来(内容瞄准)。

两条从 naive 学来的纪律:
  1) 恒等初始化:head 的 U_k 半边全零 -> 初始 Δ≡0(identity 闸门);
  2) 别双零:U_q 半边 bias 播种 N(0, 0.02)——双零是鞍点(∂Δ/∂U_q ∝ U_k=0 梯度死),
     播种 U_q 后 ∂Δ/∂U_k ∝ Q·U_q ≠ 0, U_k 先动、U_q 随后跟上。
"""

import torch
import torch.nn as nn


class CtxHyperNet(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads, head_dim, rank=4,
                 trunk_hidden=512, layer_emb_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rank = rank
        self.half = n_heads * head_dim * rank            # U_q(或 U_k)展平后的长度

        self.layer_emb = nn.Embedding(n_layers, layer_emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size + layer_emb_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
        )
        self.head = nn.Linear(trunk_hidden, 2 * self.half)
        # 纪律1+2:weight 全零(输出只由 bias 决定, 上下文依赖靠训练长出来);
        # bias 的 U_q 半边播种、U_k 半边保持零 -> Δ=0 且梯度有逃逸口
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[: self.half].normal_(0, 0.02)   # U_q 半边

    def forward(self, c, layer_idx):
        """c: (B, hidden) 该层指令池化表征;layer_idx: int。
        返回 U_q, U_k: 各 (B, H, D, r)。"""
        B = c.shape[0]
        emb = self.layer_emb.weight[layer_idx].expand(B, -1)          # (B, emb)
        feats = self.trunk(torch.cat([c, emb], dim=-1))               # (B, trunk)
        out = self.head(feats)                                        # (B, 2*half)
        U_q = out[:, : self.half].view(B, self.n_heads, self.head_dim, self.rank)
        U_k = out[:, self.half:].view(B, self.n_heads, self.head_dim, self.rank)
        return U_q, U_k
