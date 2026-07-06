"""L = L_behav + β·L_attn.

L_behav: teacher-forced causal-LM CE toward the self-distill target letter (labels
are -100 over the prompt, real token ids over the letter).

L_attn: two-sided attention-mass supervision, averaged over band layers — push the
attention mass on PRIVATE attributes toward 0, and keep SHARE attributes' mass from
dropping below their vanilla baseline (relu hinge). "other" attrs are unsupervised."""

import torch
import torch.nn.functional as F


def l_behav(logits, labels):
    """logits: (B,S,V); labels: (B,S) with -100 on positions to ignore."""
    V = logits.size(-1)
    shift_logits = logits[:, :-1].reshape(-1, V)
    shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def l_attn(masses, base_masses, share_idx, priv_idx, lam=1.0):
    """masses, base_masses: {layer -> (M,) per-attribute attention mass}. share_idx/priv_idx:
    Long index tensors into the M attributes. Returns a scalar averaged over layers."""
    terms = []
    for ell, m in masses.items():
        priv = m[priv_idx].mean() if len(priv_idx) else m.new_zeros(())
        share = torch.relu(base_masses[ell][share_idx] - m[share_idx]).mean() if len(share_idx) else m.new_zeros(())
        terms.append(priv + lam * share)
    return torch.stack(terms).mean() if terms else torch.zeros(())
