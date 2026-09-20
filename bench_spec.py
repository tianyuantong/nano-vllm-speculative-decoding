"""Throughput and latency of speculative decoding versus ordinary decoding on the 48-prompt panel.

One invocation = one configuration = one JSON result with a manifest, per-step timings,
per-request outputs and acceptance statistics. tools/summarize_bench.py turns results
into metrics. A warm-up generate() precedes the measured one.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams

GIB = 1 << 30
WARMUP_PROMPTS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--target-kv-gib", type=float, default=10)
    parser.add_argument("--draft-kv-gib", type=float, default=6)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--panel", default="benchmarks/inputs/panel-48.json")
    parser.add_argument("--commit", required=True, help="git commit of the synced tree (the sync step refuses a dirty tree)")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def manifest(args) -> dict:
    import flash_attn
    import triton
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(), "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "flash_attn": flash_attn.__version__, "triton": triton.__version__, "python": platform.python_version(),
            "commit": args.commit}


def main():
    args = parse_args()
    kwargs = dict(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, enable_prefix_cache=False,
                  kv_cache_memory_bytes=int(args.target_kv_gib * GIB), seed=args.seed, record_step_timings=True)
    if args.k:
        kwargs.update(draft_model=args.draft, num_speculative_tokens=args.k,
                      draft_kv_cache_memory_bytes=int(args.draft_kv_gib * GIB))
    llm = LLM(args.target, **kwargs)
    panel = json.loads(Path(args.panel).read_text())
    prompts = [entry["prompt_token_ids"] for entry in panel]
    params = SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, max_tokens=args.max_tokens)

    llm.generate(prompts[:WARMUP_PROMPTS], params, use_tqdm=False)          # warm-up: excluded
    if llm.speculative is not None:
        llm.speculative.stats.clear()
    wall_start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    wall_seconds = time.perf_counter() - wall_start
    result = {
        "manifest": manifest(args),
        "config": {"target": args.target, "draft": args.draft, "num_speculative_tokens": args.k,
                   "max_num_seqs": args.max_num_seqs, "temperature": args.temperature, "top_k": args.top_k,
                   "top_p": args.top_p, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
                   "seed": args.seed, "panel": args.panel, "num_prompts": len(prompts)},
        "wall_seconds": wall_seconds,
        "step_timings": llm.step_timings(),
        "phase_timings": llm.phase_timings(),
        "outputs": [{"id": entry["id"], "token_ids": output["token_ids"], "first_token_step": output["first_token_step"],
                     "finish_step": output["finish_step"]} for entry, output in zip(panel, outputs)],
        "speculative_stats": dict(llm.speculative.stats) if llm.speculative is not None else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    completion_tokens = sum(len(output["token_ids"]) for output in outputs)
    print(f"{args.output}: {completion_tokens} tokens in {wall_seconds:.1f}s wall")
    llm.exit()


if __name__ == "__main__":
    main()
