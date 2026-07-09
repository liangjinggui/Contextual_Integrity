"""Train Gtheta (base frozen) with L = l_behav + beta * l_attn.

Each step: run the frozen model once with P=I to cache the vanilla per-attribute
attention mass (base_masses), then run it with Gtheta's P to get logits + current
masses, and step Adam on Gtheta only."""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from methods.naive.intervention import InterventionModel
from methods.utils.data import TargetDataset
from methods.naive.losses import l_behav, l_attn


def parse_band(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def train_step(im, item, device, beta, lam):
    """Returns (l_behav, l_attn, total_loss). Caller does backward + opt.step()."""
    ids = item["input_ids"].unsqueeze(0).to(device)
    labels = item["labels"].unsqueeze(0).to(device)
    attn = torch.ones_like(ids)
    spans = (item["mem_tok_spans"], item["instr_tok_span"], item["query_positions"])

    im.set_context(*spans)
    im._force_identity = True                      # vanilla masses (P=I)
    with torch.no_grad():
        im.forward(ids, attention_mask=attn)
    base_masses = {k: v.detach() for k, v in im.last_masses.items()}
    im._force_identity = False

    im.set_context(*spans)                         # intervened forward (Gtheta's P), with grad
    out = im.forward(ids, attention_mask=attn)
    lb = l_behav(out.logits, labels)
    la = l_attn(im.last_masses, base_masses,
                item["share_idx"].to(device), item["priv_idx"].to(device), lam)
    return lb, la, lb + beta * la


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--targets_jsonl", required=True)
    ap.add_argument("--prompts_file", default="data/CIMemories/eval/prompts.yaml")
    ap.add_argument("--band", default="24-35")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--num_profiles", type=int, default=7)
    ap.add_argument("--target_index", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="outputs/cimemories/ckpt/naive/naive_8b.pt")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    im = InterventionModel(a.model_path, band=parse_band(a.band), rank=a.rank,
                           device=a.device, attn_implementation="sdpa")
    ds = TargetDataset(a.targets_jsonl, tok, a.prompts_file, a.num_profiles, a.target_index)
    opt = torch.optim.Adam(im.gtheta.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in im.gtheta.parameters())
    N, total = len(ds), a.epochs * len(ds)
    print(f"[train] {N} contexts x {a.epochs} epochs = {total} steps | band {a.band} r{a.rank} | "
          f"Gtheta {nparam/1e6:.1f}M params | lr {a.lr} beta {a.beta} lam {a.lam}", flush=True)

    def hms(s):
        s = int(s); return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}" if s >= 3600 else f"{s//60:d}:{s%60:02d}"

    t0 = time.time()
    ema = None                                    # smoothed loss (0.9) so the trend is readable
    step = 0
    for ep in range(a.epochs):
        run_lb = run_la = 0.0                      # per-epoch running means
        for i, item in enumerate(ds, 1):
            lb, la, loss = train_step(im, item, a.device, a.beta, a.lam)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(im.gtheta.parameters(), 1.0)  # P is sensitive -> clip
            opt.step()
            step += 1
            lv, lbv, lav = loss.item(), lb.item(), la.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            run_lb += lbv; run_la += lav
            if step % a.log_every == 0 or step == total:
                el = time.time() - t0
                sit = el / step
                eta = sit * (total - step)
                print(f"[train] ep{ep+1}/{a.epochs} {i:>3}/{N} | step {step:>4}/{total} "
                      f"({100*step/total:4.1f}%) | loss {lv:.4f} ema {ema:.4f} "
                      f"(lb {lbv:.4f} la {lav:.4f}) | gnorm {gnorm:.2f} | "
                      f"{sit:.2f}s/it | {hms(el)} elapsed | ETA {hms(eta)}", flush=True)
        print(f"[train] == epoch {ep+1}/{a.epochs} done | mean lb {run_lb/N:.4f} "
              f"la {run_la/N:.4f} | {hms(time.time()-t0)} elapsed ==", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(im.gtheta.state_dict(), a.out)
    print(f"[train] saved {a.out} | total {hms(time.time()-t0)}", flush=True)


if __name__ == "__main__":
    main()
