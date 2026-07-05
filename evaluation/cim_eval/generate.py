"""
HF batched generation for CIMemories (Stage A only), for white-box models that
can't be served by SGLang (e.g. an ASC-patched Qwen3). Writes an eval.py-style
results.jsonl (combination_* entries with model_solution.response_solution), so
it plugs straight into `eval.py --rejudge` for the REVEAL judge, then metrics.py.

vanilla baseline = run this as-is; ASC = same script with the attention patch
applied to `model`. Both use the identical generation path (no engine confound).

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.cim_eval.eval import load_prompts, load_profiles, info_attr_names, memory_map, mem_blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--num_profiles", type=int, default=10)
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--results_dir", required=True)
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
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--checkpoint_every", type=int, default=20, help="flush results.jsonl to disk every N batches (crash-safety)")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="a prior run's results.jsonl (or its dir); already-generated contexts are skipped "
                         "and generation continues. Output still goes to a fresh timestamped dir.")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = dict(dtype=torch.bfloat16, device_map=a.device_map)
    if a.max_memory_per_gpu:
        load_kw["max_memory"] = {i: a.max_memory_per_gpu for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(a.model_path, **load_kw)
    model.eval()
    print("[gen] device_map:", getattr(model, "hf_device_map", None))
    in_dev = model.get_input_embeddings().weight.device  # where inputs must live

    P = load_prompts(a.prompts_file, a.privacy_prompts_level)
    # resume from a prior (possibly partial) run: it carries the full profile + labels +
    # any combination_* already generated, so load it in place of the raw data_file.
    src = a.resume_from
    if src and os.path.isdir(src):
        src = os.path.join(src, "results.jsonl")
    data = load_profiles(src or a.data_file, a.num_profiles)

    # build one chat prompt per (profile, context); full memory pool as memories
    jobs = []  # (ctx, num_attrs, chat_text)
    for prof in data:
        names = info_attr_names(prof)
        mem = memory_map(prof)
        n = len(names)
        for ctx in prof.get("contexts") or []:
            if any(k.startswith("combination_") for k in ctx):  # already done (resume)
                continue
            task_prompt = P["task_solving"].format(
                task=ctx.get("task", ""),
                recipient=(ctx.get("recipient") or "").lower(),
                memories=mem_blob(names, mem),
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

    now = time.strftime("%Y_%B_%d_%I:%M:%S_%p")
    tag = os.path.basename(a.model_path.rstrip("/")).replace("/", "_")
    outdir = os.path.join(a.results_dir, tag, now)
    os.makedirs(outdir, exist_ok=True)
    respath = os.path.join(outdir, "results.jsonl")
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
                do_sample=True,
                temperature=a.temperature,
                top_p=a.top_p,
                top_k=a.top_k,
                num_return_sequences=a.num_trials,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc.input_ids.shape[1] :]  # strip the (left-padded) prompt
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for bi, (ctx, n, _) in enumerate(batch):
            for z in range(a.num_trials):
                letter = dec[bi * a.num_trials + z].strip()
                key = f"combination_num_to_share={n}_num_not_to_share=0_trial={z}"
                ctx.setdefault(key, {}).setdefault("model_solution", {})["response_solution"] = letter
        done += len(batch)
        el = time.time() - t0
        if (b + 1) % a.checkpoint_every == 0:
            checkpoint()
        print(f"[gen] {done}/{len(jobs)} contexts | {el:.0f}s | ~{el / max(done,1) * len(jobs):.0f}s total", flush=True)

    checkpoint()
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
