"""带 Hyper-ICL adapter 的 HF 批量生成（CIMemories Stage A）。

流程镜像 evaluation/cim_eval/generate.py（同样的 prompt 构建、length-sorted 批处理、
results.jsonl 格式 -> 直接接 eval.py --rejudge + metrics.py）。区别只有一处：模型
挂了训练好的静态 adapter。adapter 与 context 无关，因此可以照常跨 context 批量。

--disable_adapter 时 adapter.enabled=False，走同一条代码路径生成 vanilla 基线
（排除生成路径本身的混淆）。注意力全层 eager（补丁内实现），显存比 sdpa 高，
batch_size 默认取小。
"""

import argparse, json, os, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from evaluation.cim_eval.eval import load_prompts, load_profiles, info_attr_names, memory_map, mem_blob
from evaluation.cim_eval.generate import NanGuardLogitsProcessor
from methods.hyper_icl.intervention import attach_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", default=None, help="adapter state_dict；--disable_adapter 时可省")
    ap.add_argument("--disable_adapter", action="store_true", help="adapter 关：同路径生成 vanilla 基线")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--num_profiles", type=int, default=3)
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--privacy_prompts_level", type=int, default=1)
    ap.add_argument("--device_map", default="auto",
                    help="'auto' 把模型铺满所有可见 GPU；'sequential' 先填满第一张再溢出")
    ap.add_argument("--batch_size", type=int, default=4, help="全层 eager 注意力显存高，默认取小")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--checkpoint_every", type=int, default=20)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=torch.bfloat16, attn_implementation="eager", device_map=a.device_map)
    model.eval()
    print("[gen-hyper] device_map:", getattr(model, "hf_device_map", None), flush=True)
    in_dev = model.get_input_embeddings().weight.device      # 输入要放的卡
    adapter = attach_adapter(model, rank=a.rank)             # forward 内部会把参数搬到各层所在卡
    if a.disable_adapter:
        adapter.enabled = False
    elif a.ckpt:
        adapter.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    else:
        raise SystemExit("pass --ckpt <path> or --disable_adapter")

    P = load_prompts(a.prompts_file, a.privacy_prompts_level)
    data = load_profiles(a.data_file, a.num_profiles)

    # 每个 (profile, context) 一条 job：完整 memory 的标准 prompt（与 generate.py 逐字一致）
    jobs = []
    for prof in data:
        names = info_attr_names(prof)
        mem = memory_map(prof)
        n = len(names)
        for ctx in prof.get("contexts") or []:
            task_prompt = P["task_solving"].format(
                task=ctx.get("task", ""),
                recipient=(ctx.get("recipient") or "").lower(),
                memories=mem_blob(names, mem),
            )
            chat = tok.apply_chat_template(
                [{"role": "user", "content": task_prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            jobs.append((ctx, n, chat))
    jobs.sort(key=lambda j: len(tok(j[2]).input_ids))     # 长度排序，减小 padding 浪费

    now = time.strftime("%Y%m%d_%H%M%S")
    tag = os.path.basename(a.model_path.rstrip("/")).replace("/", "_")
    kind = "vanilla" if a.disable_adapter else os.path.basename(a.ckpt).replace(".pt", "")
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

    print(f"[gen-hyper] {len(jobs)} contexts x {a.num_trials} trials | {kind} -> {outdir}", flush=True)
    t0 = time.time()
    done = 0
    for b, i in enumerate(range(0, len(jobs), a.batch_size)):
        batch = jobs[i:i + a.batch_size]
        enc = tok([chat for (_, _, chat) in batch], return_tensors="pt", padding=True).to(in_dev)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                num_return_sequences=a.num_trials, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([NanGuardLogitsProcessor()]),
            )
        dec = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for bi, (ctx, n, _) in enumerate(batch):
            for z in range(a.num_trials):
                key = f"combination_num_to_share={n}_num_not_to_share=0_trial={z}"
                ctx.setdefault(key, {}).setdefault("model_solution", {})["response_solution"] = dec[bi * a.num_trials + z].strip()
        done += len(batch)
        if (b + 1) % a.checkpoint_every == 0:
            checkpoint()
        el = time.time() - t0
        print(f"[gen-hyper] {done}/{len(jobs)} | {el:.0f}s | ~{el/done*len(jobs):.0f}s total", flush=True)

    checkpoint()
    print(f"[gen-hyper] saved {respath}", flush=True)


if __name__ == "__main__":
    main()
