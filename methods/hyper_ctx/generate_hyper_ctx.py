"""带 CtxAdapter 的生成(CIMemories Stage A):每个 context 一次 generate。

与 generate_hyper_icl 的区别:U 是上下文相关的——每个 context 先定位指令 span,
prepare_for_generation() 后 prefill 时各层算一次 Gθ 并缓存 U,decode 复用;
因此不跨 context 批量(一次 generate = 1 个 context 的 5 个采样 trial,它们是
同 prompt 副本,共享同一套 U)。prompt 用与训练完全相同的构建(build_intervention_prompt)。

results.jsonl 与 eval.py 兼容,直接接 --rejudge + metrics。
"""

import argparse, json, os, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from evaluation.cim_eval.eval import load_prompts, load_profiles, info_attr_names
from evaluation.cim_eval.generate import NanGuardLogitsProcessor
from methods.hyper_ctx.intervention import attach_ctx_adapter
from methods.utils.spans import build_intervention_prompt, char_spans, token_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", required=True, help="CtxAdapter state_dict")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--num_profiles", type=int, default=3)
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--privacy_prompts_level", type=int, default=1)
    ap.add_argument("--device_map", default="auto",
                    help="'auto' 铺满可见卡;想单卡就把 CUDA_VISIBLE_DEVICES 限成一张")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--checkpoint_every", type=int, default=20)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=torch.bfloat16, attn_implementation="eager", device_map=a.device_map)
    model.eval()
    print("[gen-ctx] device_map:", getattr(model, "hf_device_map", None), flush=True)
    in_dev = model.get_input_embeddings().weight.device
    adapter = attach_ctx_adapter(model, rank=a.rank)
    adapter.load_state_dict(torch.load(a.ckpt, map_location="cpu"))

    P = load_prompts(a.prompts_file, a.privacy_prompts_level)
    data = load_profiles(a.data_file, a.num_profiles)

    jobs = []
    for prof in data:
        n = len(info_attr_names(prof))
        for ctx in prof.get("contexts") or []:
            jobs.append((prof, ctx, n))

    now = time.strftime("%Y%m%d_%H%M%S")
    tag = os.path.basename(a.model_path.rstrip("/")).replace("/", "_")
    kind = os.path.basename(a.ckpt).replace(".pt", "")
    outdir = os.path.join(a.results_dir, tag, f"{now}_{kind}")
    os.makedirs(outdir, exist_ok=True)
    respath = os.path.join(outdir, "results.jsonl")
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    def checkpoint():
        tmp = respath + ".tmp"
        with open(tmp, "w") as f:
            for prof in data:
                f.write(json.dumps(prof) + "\n")
        os.replace(tmp, respath)

    print(f"[gen-ctx] {len(jobs)} contexts x {a.num_trials} trials | {kind} -> {outdir}", flush=True)
    t0 = time.time()
    for j, (prof, ctx, n) in enumerate(jobs, 1):
        # 与训练完全一致的 prompt 构建 + 指令 span 定位
        prompt, mem_lines = build_intervention_prompt(P, prof, ctx)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        instr_span = token_spans(tok, chat, *char_spans(chat, mem_lines))["instr_tok_span"]
        adapter.prepare_for_generation(instr_span)      # prefill 算 U 并缓存, decode 复用

        ids = torch.tensor([tok(chat, add_special_tokens=False).input_ids], device=in_dev)
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                num_return_sequences=a.num_trials, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([NanGuardLogitsProcessor()]),
            )
        dec = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        for z in range(a.num_trials):
            key = f"combination_num_to_share={n}_num_not_to_share=0_trial={z}"
            ctx.setdefault(key, {}).setdefault("model_solution", {})["response_solution"] = dec[z].strip()

        if j % a.checkpoint_every == 0:
            checkpoint()
        el = time.time() - t0
        print(f"[gen-ctx] {j}/{len(jobs)} | {el:.0f}s | ~{el/j*len(jobs):.0f}s total", flush=True)

    checkpoint()
    print(f"[gen-ctx] saved {respath}", flush=True)


if __name__ == "__main__":
    main()
