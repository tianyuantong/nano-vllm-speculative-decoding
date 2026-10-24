"""Smoke test: the speculative engine generates, k = 0 generates, both return the requested fields."""
import argparse
import json

from nanovllm import LLM, SamplingParams

GIB = 1 << 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def engine_kwargs(args, draft: bool) -> dict:
    kwargs = dict(max_model_len=4096, max_num_seqs=4, enable_prefix_cache=False,
                  kv_cache_memory_bytes=4 * GIB, seed=0, record_step_timings=True)
    if draft:
        kwargs.update(draft_model=args.draft, num_speculative_tokens=args.k, draft_kv_cache_memory_bytes=2 * GIB)
    return kwargs


def main():
    args = parse_args()
    prompts = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):"]
    report = {}
    for label, draft in (("speculative", True), ("ordinary", False)):
        llm = LLM(args.target, **engine_kwargs(args, draft))
        outputs = llm.generate(prompts, SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=48), use_tqdm=False)
        timings = llm.step_timings()
        report[label] = {
            "texts": [output["text"] for output in outputs],
            "steps": len(timings),
            "kinds": sorted({timing["kind"] for timing in timings}),
            "first_token_steps": [output["first_token_step"] for output in outputs],
            "finish_steps": [output["finish_step"] for output in outputs],
            "stats": dict(llm.speculative.stats) if draft else None,
        }
        llm.exit()
    assert report["speculative"]["kinds"] == ["prefill", "round"], report["speculative"]["kinds"]
    assert report["ordinary"]["kinds"] == ["decode", "prefill"], report["ordinary"]["kinds"]
    assert all(step >= 0 for step in report["speculative"]["first_token_steps"] + report["speculative"]["finish_steps"])
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
