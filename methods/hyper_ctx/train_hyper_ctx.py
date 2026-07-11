"""训练 CtxAdapter(base 冻结):与 hyper_icl 完全同配置的单变量对照。

L = L_H-anchor + lambda·L_sup(损失/数据(v1 teacher)/超参全部复用 hyper_icl),
唯一区别:U_q/U_k 由 Gθ(c^ℓ) 现场生成——student 前向前先 set_context(指令 span)。
teacher 前向 adapter 关(=裸 base)。反传只更新 CtxAdapter(Gθ + 门 w,b)。
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from methods.hyper_ctx.intervention import attach_ctx_adapter
from methods.hyper_icl.data import HyperICLDataset
from methods.hyper_icl.losses import l_h_anchor, l_sup


def load_patched_model(model_path, rank=4, device_map="cuda:0", dtype=torch.bfloat16):
    """冻结 base + 挂上下文条件 adapter;开 gradient checkpointing(全层反传显存大)。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="eager", device_map=device_map)
    adapter = attach_ctx_adapter(model, rank=rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # HF checkpointing 只在 training 模式生效;Qwen3 全 RMSNorm 且 dropout=0, train 模式数值同 eval
    model.train()
    in_dev = model.get_input_embeddings().weight.device
    return model, adapter, in_dev


def train_step(model, adapter, item, device, lam, kappa):
    """返回 (anchor, sup, total)。调用方负责 backward + step。"""
    teacher_ids = item["teacher_ids"].unsqueeze(0).to(device)
    student_ids = item["student_ids"].unsqueeze(0).to(device)
    labels = item["student_labels"].unsqueeze(0).to(device)
    n = item["n_letter"]

    adapter.enabled = False                      # teacher = 裸 base
    with torch.no_grad():
        t_out = model(teacher_ids, output_hidden_states=True, use_cache=False)
    teacher_hs = [h[0, -n:] for h in t_out.hidden_states[1:]]

    adapter.enabled = True                       # student = base + Gθ(c) 生成的 U
    adapter.set_context(item["instr_tok_span"])
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
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Gθ 路径对 lr 敏感(Δ 四次型): 5e-3/1e-3 实测发散, 2e-4 稳(与静态版对照的唯一超参偏离)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--device_map", default="cuda:0",
                    help="'cuda:0' 单卡(默认;本机多卡通路不可靠, 见 exp_records 平台案卷)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="outputs/cimemories/ckpt/hyper_ctx/hyper_ctx_8b.pt")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    model, adapter, in_dev = load_patched_model(a.model_path, rank=a.rank, device_map=a.device_map)
    ds = HyperICLDataset(a.targets_jsonl, tok, a.prompts_file, a.num_profiles, a.target_index)
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr)
    total = a.epochs * len(ds)
    sched = get_cosine_schedule_with_warmup(opt, int(total * a.warmup_frac), total)
    n_param = sum(p.numel() for p in adapter.parameters())
    print(f"[train] {len(ds)} contexts x {a.epochs} epochs = {total} steps | rank {a.rank} | "
          f"CtxAdapter {n_param/1e6:.1f}M params | lr {a.lr} lam {a.lam} kappa {a.kappa}", flush=True)

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
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)  # Δ 对参数四次, 必须裁剪
            opt.step()
            sched.step()
            step += 1
            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            sum_anchor += anchor.item()
            sum_sup += sup.item()
            if step % a.log_every == 0 or step == total:
                elapsed = time.time() - t0
                eta = elapsed / step * (total - step)
                print(f"[train] ep{ep+1}/{a.epochs} {i:>3}/{len(ds)} | step {step:>4}/{total} "
                      f"({100*step/total:4.1f}%) | loss {loss.item():.4f} ema {ema:.4f} "
                      f"(anchor {anchor.item():.4f} sup {sup.item():.4f}) | gnorm {grad_norm:.2f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {elapsed/step:.2f}s/it | "
                      f"{hms(elapsed)} elapsed | ETA {hms(eta)}", flush=True)
        print(f"[train] == epoch {ep+1}/{a.epochs} done | mean anchor {sum_anchor/len(ds):.4f} "
              f"sup {sum_sup/len(ds):.4f} | {hms(time.time()-t0)} elapsed ==", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        ep_path = a.out.replace(".pt", f"_ep{ep+1}.pt")
        torch.save(adapter.state_dict(), ep_path)
        print(f"[train] saved {ep_path}", flush=True)

    torch.save(adapter.state_dict(), a.out)
    print(f"[train] saved {a.out} | total {hms(time.time()-t0)}", flush=True)


if __name__ == "__main__":
    main()
