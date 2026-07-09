"""HF generation with the Gtheta attention intervention, for CIMemories Stage A eval.

Mirrors evaluation/cim_eval/generate.py (same results.jsonl format -> plugs straight into
`eval.py --rejudge` then metrics.py), but runs the patched InterventionModel instead of a
plain model. The intervention P is context-fixed, so we generate ONE context at a time:
prepare_for_generation() sets the spans, P is computed+cached at prefill, and all num_trials
samples of that context share it (they are the same prompt replicated by num_return_sequences).

vanilla-through-the-same-path = pass --force_identity (P=I -> byte-identical attention), which
isolates the intervention as the only variable vs the trained checkpoint.

Prompt/spans are built with the exact training helpers so the input is identical to training."""

import argparse, json, os, time
import torch
from transformers import AutoTokenizer, LogitsProcessorList

from evaluation.cim_eval.eval import load_prompts, load_profiles, info_attr_names
from evaluation.cim_eval.generate import NanGuardLogitsProcessor
from methods.naive.intervention import InterventionModel
from methods.utils.spans import build_intervention_prompt, char_spans, token_spans


def parse_band(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", default=None, help="Gtheta state_dict; omit (or --force_identity) for vanilla")
    ap.add_argument("--force_identity", action="store_true", help="P=I: vanilla through the patched path")
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--num_profiles", type=int, default=3)
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--privacy_prompts_level", type=int, default=1)
    ap.add_argument("--band", default="24-35")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--checkpoint_every", type=int, default=20, help="flush results.jsonl every N contexts")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # band layers run eager internally (need the scores); non-band use sdpa for speed.
    im = InterventionModel(a.model_path, band=parse_band(a.band), rank=a.rank,
                           device=a.device, attn_implementation="sdpa")
    if a.force_identity:
        im._force_identity = True
    elif a.ckpt:
        im.gtheta.load_state_dict(torch.load(a.ckpt, map_location=a.device))
    else:
        raise SystemExit("pass --ckpt <path> or --force_identity")

    P = load_prompts(a.prompts_file, a.privacy_prompts_level)
    data = load_profiles(a.data_file, a.num_profiles)

    now = time.strftime("%Y%m%d_%H%M%S")
    tag = os.path.basename(a.model_path.rstrip("/")).replace("/", "_")
    kind = "identity" if a.force_identity else os.path.basename(a.ckpt or "ckpt").replace(".pt", "")
    outdir = os.path.join(a.results_dir, tag, f"{now}_{kind}")
    os.makedirs(outdir, exist_ok=True)
    respath = os.path.join(outdir, "results.jsonl")
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    def checkpoint():  # atomic write so a crash never leaves a half file
        tmp = respath + ".tmp"
        with open(tmp, "w") as f:
            for prof in data:
                f.write(json.dumps(prof) + "\n")
        os.replace(tmp, respath)

    # one (profile, context) job at a time; each generates num_trials samples sharing one P
    jobs = []
    for prof in data:
        n = len(info_attr_names(prof))
        for ctx in prof.get("contexts") or []:
            jobs.append((prof, ctx, n))
    print(f"[gen-naive] {len(jobs)} contexts x {a.num_trials} trials = {len(jobs)*a.num_trials} "
          f"generations | {kind} -> {outdir}", flush=True)

    t0 = time.time()
    for j, (prof, ctx, n) in enumerate(jobs, 1):
        prompt, mem_lines = build_intervention_prompt(P, prof, ctx)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        mcs, ics = char_spans(chat, mem_lines)
        ts = token_spans(tok, chat, mcs, ics)
        im.prepare_for_generation(ts["mem_tok_spans"], ts["instr_tok_span"])   # P cached at prefill

        ids = torch.tensor([tok(chat, add_special_tokens=False).input_ids], device=a.device)
        with torch.no_grad():
            out = im.model.generate(
                ids, max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                num_return_sequences=a.num_trials, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([NanGuardLogitsProcessor()]),
            )
        dec = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        for z in range(a.num_trials):
            key = f"combination_num_to_share={n}_num_not_to_share=0_trial={z}"
            ctx.setdefault(key, {}).setdefault("model_solution", {})["response_solution"] = dec[z].strip()

        el = time.time() - t0
        if j % a.checkpoint_every == 0:
            checkpoint()
        print(f"[gen-naive] {j}/{len(jobs)} | {el:.0f}s | ~{el/j*len(jobs):.0f}s total", flush=True)

    checkpoint()
    print(f"[gen-naive] saved {respath}", flush=True)


if __name__ == "__main__":
    main()
