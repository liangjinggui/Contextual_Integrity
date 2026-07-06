"""
Self-distillation TARGET generation for the Naive Solution (L_behav teacher).
For each (profile, context) it builds a TARGET prompt that (1) drops the private
memory rows and (2) explicitly requires the share attributes, then generates the
target letter(s) y* with the frozen base model (greedy by default; --do_sample +
--num_trials>1 for best-of-N). Writes targets.jsonl (profiles as-is + labels),
each context gaining a `target_solutions` list. Run once per split (train7/test3).

Multi-GPU = HF device_map model sharding (accelerate, pipeline style; no model
code changes). This SCALES with model size on the same command: an 8B lands
whole on one visible GPU (effectively single-card, no cross-device copies),
while a 32B auto-spreads across all visible GPUs. Give it enough free GPUs.
Optional --max_memory_per_gpu forces a spread / caps usage when GPUs are shared
with co-tenant jobs.

Prompt/data helpers are imported from eval.py so the prompt is byte-identical.
"""

import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, LogitsProcessor, LogitsProcessorList

from evaluation.cim_eval.eval import load_prompts, load_profiles, info_attr_names, memory_map, mem_blob
from evaluation.cim_eval.metrics import label_sets


class NanGuardLogitsProcessor(LogitsProcessor):
    """Sanitize inf/nan in the (post-warper) scores so multinomial can't assert on a
    bad probability tensor. No-op on finite logits (the overwhelming majority): only
    the rare step that would crash gets nan/-inf -> very negative (effectively filtered)
    and +inf -> large finite. Masks the symptom of the bf16 forward instability so a
    single bad step no longer kills the whole run."""

    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--num_profiles", type=int, default=10)
    ap.add_argument("--num_trials", type=int, default=1, help="# targets per context (1=single; >1 with --do_sample for best-of-N)")
    ap.add_argument("--out_dir", required=True, help="explicit output dir, e.g. .../self_distill_targets/<model>/<ts>/train")
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--privacy_prompts_level", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8, help="# prompts per batch (each yields num_trials samples)")
    ap.add_argument("--device_map", type=str, default="sequential",
                    help="'sequential' PACKS one GPU before spilling (8B stays single-card, 32B spills) — "
                         "the default. 'auto'/'balanced' spreads even a small model across all GPUs "
                         "(needless pipeline latency). 'cuda:0' pins to one GPU.")
    ap.add_argument("--max_memory_per_gpu", type=str, default=None,
                    help="e.g. '5GiB' — cap weight placement per visible GPU to FORCE a spread / avoid "
                         "colliding with co-tenant jobs. Unset = plain device_map='auto' (packs GPU0 first, "
                         "spilling to the next only when it doesn't fit — right for a free-GPU run).")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="model load dtype. NOTE: the multinomial 'inf/nan' crash was the 4-card device_map, "
                         "not precision — keep CUDA_VISIBLE_DEVICES to <=2 cards for this model.")
    ap.add_argument("--do_sample", action="store_true", help="sample instead of greedy (needed for best-of-N); default greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--checkpoint_every", type=int, default=20, help="flush targets.jsonl to disk every N batches (crash-safety)")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="a prior run's targets.jsonl (or its dir); already-generated contexts are skipped "
                         "and generation continues. Output still goes to --out_dir.")
    ap.add_argument("--share_mass", type=float, default=0.94, help="share (necessary) mass threshold for label_sets")
    ap.add_argument("--private_mass", type=float, default=0.64, help="private mass threshold for label_sets")
    ap.add_argument("--seed", type=int, default=42, help="fix RNG so sampling is reproducible (deterministic targets)")
    a = ap.parse_args()

    set_seed(a.seed)  # deterministic sampling: reproducible targets, and a stable path re: the multinomial nan

    tok = AutoTokenizer.from_pretrained(a.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = dict(dtype=getattr(torch, a.dtype), device_map=a.device_map)
    if a.max_memory_per_gpu:
        load_kw["max_memory"] = {i: a.max_memory_per_gpu for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(a.model_path, **load_kw)
    model.eval()
    print("[gen] device_map:", getattr(model, "hf_device_map", None))
    in_dev = model.get_input_embeddings().weight.device  # where inputs must live

    P = load_prompts(a.prompts_file, a.privacy_prompts_level)
    # resume from a prior (possibly partial) run: it carries the full profile + labels +
    # any target_solutions already generated, so load it in place of the raw data_file.
    src = a.resume_from
    if src and os.path.isdir(src):
        src = os.path.join(src, "targets.jsonl")
    data = load_profiles(src or a.data_file, a.num_profiles)

    # build the self-distillation TARGET prompt per (profile, context):
    #   (1) drop private memory rows  (2) explicitly require the share attributes.
    # base then writes a letter that structurally can't leak private and is pushed to
    # cover share -> a high-completeness, zero-violation teacher for L_behav.
    jobs = []  # (ctx, num_attrs, chat_text)
    for prof in data:
        names = info_attr_names(prof)
        mem = memory_map(prof)
        for ctx in prof.get("contexts") or []:
            if ctx.get("target_solutions"):  # already generated (resume)
                continue
            share, private = label_sets(prof, ctx, a.share_mass, a.private_mass)
            private_set = set(private)
            kept = [x for x in names if x not in private_set]   # (1) drop private rows
            n = len(kept)
            task_prompt = P["task_solving"].format(
                task=ctx.get("task", ""),
                recipient=(ctx.get("recipient") or "").lower(),
                memories=mem_blob(kept, mem),
            )
            if share:                                           # (2) require the share attrs
                share_block = "\n".join(f"- {mem[x][0]}" for x in share if mem.get(x))
                instr = (
                    "Make sure to naturally incorporate the following user information "
                    "where relevant:\n" + share_block + "\n\n"
                )
                task_prompt = task_prompt.replace(
                    "Only output the message", instr + "Only output the message", 1
                )
            chat = tok.apply_chat_template(
                [{"role": "user", "content": task_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            jobs.append((ctx, n, chat))

    # sort by prompt length so each batch is length-uniform: minimal left-padding =>
    # lower peak KV and no OOM spike from one long prompt inflating the whole batch.
    jobs.sort(key=lambda j: len(tok(j[2]).input_ids))

    outdir = a.out_dir
    os.makedirs(outdir, exist_ok=True)
    respath = os.path.join(outdir, "targets.jsonl")
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    def checkpoint():  # atomic: write tmp then rename, so a crash never leaves a half file
        tmp = respath + ".tmp"
        with open(tmp, "w") as f:
            for prof in data:
                f.write(json.dumps(prof) + "\n")
        os.replace(tmp, respath)

    print(f"[gen] {len(jobs)} contexts x {a.num_trials} trials = {len(jobs) * a.num_trials} generations -> {outdir}", flush=True)
    t0 = time.time()
    done = 0
    for b, i in enumerate(range(0, len(jobs), a.batch_size)):
        batch = jobs[i : i + a.batch_size]
        enc = tok([c for (_, _, c) in batch], return_tensors="pt", padding=True).to(in_dev)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=a.max_new_tokens,
                do_sample=a.do_sample,
                temperature=a.temperature,
                top_p=a.top_p,
                top_k=a.top_k,
                num_return_sequences=a.num_trials,
                pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([NanGuardLogitsProcessor()]),
            )
        gen = out[:, enc.input_ids.shape[1] :]  # strip the (left-padded) prompt
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for bi, (ctx, _, _) in enumerate(batch):
            ctx["target_solutions"] = [dec[bi * a.num_trials + z].strip() for z in range(a.num_trials)]
        done += len(batch)
        el = time.time() - t0
        if (b + 1) % a.checkpoint_every == 0:
            checkpoint()
        print(f"[gen] {done}/{len(jobs)} contexts | {el:.0f}s | ~{el / max(done,1) * len(jobs):.0f}s total", flush=True)

    checkpoint()
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
