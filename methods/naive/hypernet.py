"""Gtheta — the ONLY trainable module. Per (intervention layer, memory attribute)
it reads the pooled instruction repr `c` and memory repr `e` and emits low-rank
factors assembled into the projections P_Q = I + U_Q V_Qᵀ, P_K = I + U_K V_Kᵀ.

The output head is zero-initialized, so at init U=V=0 -> P=I: the intervention is
an exact no-op (vanilla warm start); training moves P away from identity."""

import torch
import torch.nn as nn


class Gtheta(nn.Module):
    def __init__(self, hidden_size, head_dim=128, rank=8, n_layers=12, hidden=512):
        super().__init__()
        self.head_dim, self.rank = head_dim, rank
        self.layer_emb = nn.Embedding(n_layers, hidden)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_size + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, 4 * head_dim * rank)  # U_Q, V_Q, U_K, V_K
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Identity init WITHOUT a dead gradient: keep U=0 (so U·Vᵀ=0 -> P=I) but seed V
        # with a small bias. Then ∂P/∂U = V ≠ 0 at init, so gradient flows (zeroing BOTH
        # factors is a saddle with ∂P/∂U=∂P/∂V=0 -> training can't leave identity).
        hr = head_dim * rank
        with torch.no_grad():
            self.head.bias[hr:2 * hr].normal_(0, 0.02)        # V_Q slice
            self.head.bias[3 * hr:4 * hr].normal_(0, 0.02)    # V_K slice

    def forward(self, c, e, layer_idx):
        """c, e: (N, hidden_size) pooled reprs; returns P_Q, P_K: (N, head_dim, head_dim)."""
        N = c.shape[0]
        le = self.layer_emb(torch.full((N,), layer_idx, dtype=torch.long, device=c.device))
        x = self.head(self.trunk(torch.cat([c, e, le], dim=-1)))
        U_Q, V_Q, U_K, V_K = x.view(N, 4, self.head_dim, self.rank).unbind(1)
        I = torch.eye(self.head_dim, device=c.device, dtype=c.dtype)
        P_Q = I + U_Q @ V_Q.transpose(-1, -2)
        P_K = I + U_K @ V_K.transpose(-1, -2)
        return P_Q, P_K
