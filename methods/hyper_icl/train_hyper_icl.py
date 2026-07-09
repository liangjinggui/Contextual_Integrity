"""训练 Hyper-ICL adapter（base 冻结）：L = L_H-anchor + lambda * L_sup（式 17）。

每步两趟前向（同一个模型，靠 adapter.enabled 切换）：
  teacher：adapter 关，喂干净 prompt + y*（无梯度）-> 信件 token 各层隐状态
  student：adapter 开，喂完整 memory prompt + y*   -> 隐状态 + logits
anchor 把 student 的信件表示拉向"没看过 private 的 teacher"，sup 是对 y* 的 CE。

全 36 层 eager 注意力带梯度会超显存，训练开 gradient checkpointing。
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from methods.hyper_icl.intervention import attach_adapter
from methods.hyper_icl.data import HyperICLDataset
from methods.hyper_icl.losses import l_h_anchor, l_sup


def load_patched_model(model_path, rank=4, device_map="cuda:0", dtype=torch.bfloat16):
    """冻结 base + 挂 adapter；开 gradient checkpointing（全层 eager 反传的显存开销大）。

    device_map: 'cuda:0' 整模型放可见的第一张卡（8B 单卡装得下，训练最快，默认）；
    'auto' 按层切分铺满所有可见卡（仅模型单卡装不下时用，流水线式、更慢）。
    返回 (model, adapter, in_dev)，in_dev = 输入张量该放的卡（嵌入层所在卡）。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="eager", device_map=device_map)
    adapter = attach_adapter(model, rank=rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # HF 的 checkpointing 只在 training 模式下生效（eval 会存满 36 层 eager 权重矩阵 -> OOM）。
    # Qwen3 全是 RMSNorm 且 dropout=0，train 模式数值上与 eval 无差别。
    model.train()
    in_dev = model.get_input_embeddings().weight.device
    return model, adapter, in_dev


def train_step(model, adapter, item, device, lam, kappa):
    """返回 (anchor, sup, total)。调用方负责 backward + step。"""
    teacher_ids = item["teacher_ids"].unsqueeze(0).to(device)
    student_ids = item["student_ids"].unsqueeze(0).to(device)
    labels = item["student_labels"].unsqueeze(0).to(device)
    n = item["n_letter"]

    adapter.enabled = False                      # teacher = 精确 base（identity 测试保证）
    with torch.no_grad():
        t_out = model(teacher_ids, output_hidden_states=True, use_cache=False)
    teacher_hs = [h[0, -n:] for h in t_out.hidden_states[1:]]     # 36 层 x (n_letter, d)

    adapter.enabled = True                       # student = base + adapter
    s_out = model(student_ids, output_hidden_states=True, use_cache=False)
    student_hs = [h[0, -n:] for h in s_out.hidden_states[1:]]

    anchor = l_h_anchor(student_hs, teacher_hs, kappa)
    sup = l_sup(s_out.logits, labels)
    return anchor, sup, anchor + lam * sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--targets_jsonl", required=True)
    ap.add_argument("--prompts_file", default="data/CIMemories/eval/prompts.yaml")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--num_profiles", type=int, default=7)
    ap.add_argument("--target_index", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.5, help="L_sup 权重（论文最优 0.5）")
    ap.add_argument("--kappa", type=float, default=0.1, help="双曲曲率（论文最优 0.1）")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--device_map", default="cuda:0",
                    help="'cuda:0' 单卡（默认，最快）；'auto' 模型装不下时按层铺满所有可见卡")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="outputs/cimemories/ckpt/hyper_icl/hyper_icl_8b.pt")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    model, adapter, in_dev = load_patched_model(a.model_path, rank=a.rank, device_map=a.device_map)
    ds = HyperICLDataset(a.targets_jsonl, tok, a.prompts_file, a.num_profiles, a.target_index)
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr)
    total = a.epochs * len(ds)
    sched = get_cosine_schedule_with_warmup(opt, int(total * a.warmup_frac), total)
    n_param = sum(p.numel() for p in adapter.parameters())
    print(f"[train] {len(ds)} contexts x {a.epochs} epochs = {total} steps | rank {a.rank} | "
          f"adapter {n_param/1e6:.2f}M params | lr {a.lr} lam {a.lam} kappa {a.kappa}", flush=True)

    def hms(sec):
        sec = int(sec)
        return f"{sec//3600:d}:{(sec%3600)//60:02d}:{sec%60:02d}" if sec >= 3600 else f"{sec//60:d}:{sec%60:02d}"

    t0 = time.time()
    ema = None
    step = 0
    for ep in range(a.epochs):
        sum_anchor = sum_sup = 0.0
        for i, item in enumerate(ds, 1):
            anchor, sup, loss = train_step(model, adapter, item, in_dev, a.lam, a.kappa)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            sum_anchor += anchor.item()
            sum_sup += sup.item()
            if step % a.log_every == 0 or step == total:
                elapsed = time.time() - t0
                per_step = elapsed / step
                eta = per_step * (total - step)
                print(f"[train] ep{ep+1}/{a.epochs} {i:>3}/{len(ds)} | step {step:>4}/{total} "
                      f"({100*step/total:4.1f}%) | loss {loss.item():.4f} ema {ema:.4f} "
                      f"(anchor {anchor.item():.4f} sup {sup.item():.4f}) | gnorm {grad_norm:.2f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {per_step:.2f}s/it | "
                      f"{hms(elapsed)} elapsed | ETA {hms(eta)}", flush=True)
        print(f"[train] == epoch {ep+1}/{a.epochs} done | mean anchor {sum_anchor/len(ds):.4f} "
              f"sup {sum_sup/len(ds):.4f} | {hms(time.time()-t0)} elapsed ==", flush=True)
        # 论文按 best-performing epoch 报告 -> 每个 epoch 都存，评测时可选
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        ep_path = a.out.replace(".pt", f"_ep{ep+1}.pt")
        torch.save(adapter.state_dict(), ep_path)
        print(f"[train] saved {ep_path}", flush=True)

    torch.save(adapter.state_dict(), a.out)
    print(f"[train] saved {a.out} | total {hms(time.time()-t0)}", flush=True)


if __name__ == "__main__":
    main()
