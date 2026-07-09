"""Hyper-ICL 损失（arXiv 2606.04434 式 10-17）。

L = L_H-anchor + lambda * L_sup
  L_H-anchor：逐层把 student / teacher 的隐状态（LN 后）经 Lorentz exp 映射到
              双曲面 H^d_kappa，最小化两者测地距离的平方（式 16）。
  L_sup     ：student 输出对 y* 的 teacher-forcing 交叉熵。

约定：曲率 kappa > 0（论文最优 0.1）；基点 o = (sqrt(1/kappa), 0, ..., 0)；
LN(h) 是无参数 layer norm（式 12）。内部统一用 float32 算（bf16 下 cosh 会溢出）。
"""

import torch
import torch.nn.functional as F

_EPS = 1e-7


def lorentz_expmap(u, kappa):
    """切空间向量 u: (..., d) -> 双曲面上的点 p: (..., d+1)（式 13）。

    p = cosh(sqrt(k)*||u||) * o + sinh(sqrt(k)*||u||) / (sqrt(k)*||u||) * (0, u)
    其中 o = (sqrt(1/k), 0, ..., 0)。||u||=0 时系数取 1（论文约定）。
    """
    u = u.float()
    sqrt_k = kappa ** 0.5
    norm = u.norm(dim=-1, keepdim=True).clamp_min(_EPS)            # (..., 1)
    first = torch.cosh(sqrt_k * norm) / sqrt_k                     # (..., 1) 首坐标
    rest = torch.sinh(sqrt_k * norm) / (sqrt_k * norm) * u         # (..., d)
    return torch.cat([first, rest], dim=-1)                        # (..., d+1)


def lorentz_dist(p, q, kappa):
    """双曲面上两点的测地距离（式 15）：d = arcosh(-kappa * <p,q>_L) / sqrt(kappa)。

    Lorentz 内积 <p,q>_L = -p0*q0 + sum(p_t*q_t)（式 10）。对合法点 -kappa*<p,q>_L >= 1，
    数值上 clamp 到 1 以防浮点误差落出定义域。
    """
    inner = -p[..., 0] * q[..., 0] + (p[..., 1:] * q[..., 1:]).sum(-1)   # (...,)
    arg = (-kappa * inner).clamp_min(1.0)
    return torch.acosh(arg) / kappa ** 0.5


def lorentz_dist_tangent(u, v, kappa):
    """d_L(exp_o(u), exp_o(v))：直接从切向量算，数值稳定版。u, v: (..., d)。

    经双曲面坐标算内积（lorentz_dist）在大范数下会灾难性相消：d=4096、LN 后
    ||u||~64 时 p0*q0 ~ 9e17，真值 -1/kappa=-10 是两个 9e17 巨数之差，float32
    只有 7 位有效数字 -> 误差 ~1e10，重合点算出 dist~73（实测 self-anchor=0.5），
    梯度全是噪声（这正是首两次训练 anchor 平死的根因）。等价闭式全为正项乘积、
    无相消：
        -kappa*<exp_o(u),exp_o(v)>_L = cosh(a-b) + sinh(a)*sinh(b)*||u^-v^||^2/2
    其中 a=sqrt(k)||u||, b=sqrt(k)||v||，u^,v^ 是单位向量。重合点精确得 dist=0。
    """
    u, v = u.float(), v.float()
    sqrt_k = kappa ** 0.5
    nu = u.norm(dim=-1, keepdim=True).clamp_min(_EPS)              # (..., 1)
    nv = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    a, b = sqrt_k * nu, sqrt_k * nv
    gap_sq = ((u / nu - v / nv) ** 2).sum(-1, keepdim=True)        # ||u^-v^||^2
    arg = torch.cosh(a - b) + torch.sinh(a) * torch.sinh(b) * gap_sq / 2
    return (torch.acosh(arg.clamp_min(1.0)) / sqrt_k).squeeze(-1)  # (...,)


def l_h_anchor(student_layers, teacher_layers, kappa=0.1):
    """式 16：跨层、跨 token 的双曲锚蒸馏损失。

    student_layers / teacher_layers：等长列表，每项是对齐好的 (T, d) 隐状态
    （同一批 y* token 在该层的表示）。返回标量。
    """
    d = student_layers[0].shape[-1]
    per_layer = []
    for h_s, h_t in zip(student_layers, teacher_layers):
        u = F.layer_norm(h_s.float(), (d,))
        v = F.layer_norm(h_t.float(), (d,))
        dist_sq = lorentz_dist_tangent(u, v, kappa) ** 2               # (T,)
        # 场景适配（披露）：dist^2 按 hidden_dim 归一。原论文 teacher/student 只差几个
        # demo、表示很近；我们 teacher/student 的 prompt 内容不同（student 多 private 行），
        # 4096 维距离^2 ~1e4 会把 λ·L_sup(~1) 淹没 4 个量级。除以 d 后与 sup 同量级。
        per_layer.append(dist_sq.mean() / d)
    # device_map 多卡下各层隐状态分居不同卡 -> 逐层标量先搬到同一张卡再 stack
    dev0 = per_layer[0].device
    return torch.stack([p.to(dev0) for p in per_layer]).mean()


def l_sup(logits, labels):
    """teacher-forcing 交叉熵：labels 为 -100 的位置不计。logits (B,T,V), labels (B,T)。"""
    vocab = logits.size(-1)
    shift_logits = logits[:, :-1].reshape(-1, vocab)
    shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
    return F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
