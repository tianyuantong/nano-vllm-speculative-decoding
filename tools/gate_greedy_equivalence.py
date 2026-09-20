"""G3: greedy k = 0 versus greedy speculation on the 48-prompt panel.

`run` generates with one configuration and stores token ids; `compare` reports the match
rate and, for every divergence, the fp32 top-2 logit margin of the target at that position
(computed by prefilling prompt + common prefix), to be read against the G1 noise floor.
"""
import argparse
import json

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

GIB = 1 << 30
MAX_TOKENS = 512


def load_panel(path: str) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)


def run(args) -> None:
    kwargs = dict(max_model_len=4096, max_num_seqs=args.max_num_seqs, enable_prefix_cache=False,
                  kv_cache_memory_bytes=int(args.target_kv_gib * GIB), seed=0)
    if args.k:
        kwargs.update(draft_model=args.draft, num_speculative_tokens=args.k,
                      draft_kv_cache_memory_bytes=int(args.draft_kv_gib * GIB))
    llm = LLM(args.target, **kwargs)
    panel = load_panel(args.panel)
    outputs = llm.generate([entry["prompt_token_ids"] for entry in panel], SamplingParams(temperature=0, max_tokens=MAX_TOKENS), use_tqdm=False)
    with open(args.output, "w") as handle:
        json.dump({"k": args.k, "outputs": [{"id": entry["id"], "token_ids": output["token_ids"]}
                                            for entry, output in zip(panel, outputs)]}, handle)
    llm.exit()


def top2_margin(llm, token_ids: list[int]) -> float:
    """fp32 top-2 logit margin of the target after `token_ids`, through one eager prefill."""
    seq = Sequence(token_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    seq.num_scheduled_tokens = len(token_ids)
    input_ids, positions = llm.model_runner.prepare_prefill([seq])
    logits = llm.model_runner.run_model(input_ids, positions, True)[-1].float()
    reset_context()
    llm.scheduler.block_manager.deallocate(seq)
    return float(logits.topk(2).values.diff().abs())


def compare(args) -> None:
    ordinary, speculative = (json.load(open(path)) for path in (args.ordinary, args.speculative))
    panel = {entry["id"]: entry["prompt_token_ids"] for entry in load_panel(args.panel)}
    llm = LLM(args.target, max_model_len=4096, max_num_seqs=1, enable_prefix_cache=False, kv_cache_memory_bytes=4 * GIB)
    divergences = []
    matches = 0
    for left, right in zip(ordinary["outputs"], speculative["outputs"]):
        assert left["id"] == right["id"]
        if left["token_ids"] == right["token_ids"]:
            matches += 1
            continue
        index = next(i for i, (x, y) in enumerate(zip(left["token_ids"], right["token_ids"])) if x != y)
        margin = top2_margin(llm, panel[left["id"]] + left["token_ids"][:index])
        divergences.append({"id": left["id"], "index": index, "top2_margin": margin,
                            "ordinary_token": left["token_ids"][index], "speculative_token": right["token_ids"][index]})
    llm.exit()
    report = {"requests": len(ordinary["outputs"]), "matches": matches, "divergences": divergences,
              "margins_below_noise_floor": sum(entry["top2_margin"] < args.noise_floor for entry in divergences)}
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "divergences"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--target", required=True)
    run_parser.add_argument("--draft")
    run_parser.add_argument("--k", type=int, default=0)
    run_parser.add_argument("--max-num-seqs", type=int, default=4)
    run_parser.add_argument("--target-kv-gib", type=float, default=6)
    run_parser.add_argument("--draft-kv-gib", type=float, default=3)
    run_parser.add_argument("--panel", default="benchmarks/inputs/panel-48.json")
    run_parser.add_argument("--output", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--target", required=True)
    compare_parser.add_argument("--ordinary", required=True)
    compare_parser.add_argument("--speculative", required=True)
    compare_parser.add_argument("--panel", default="benchmarks/inputs/panel-48.json")
    compare_parser.add_argument("--noise-floor", type=float, required=True, help="G1 noise floor in logit units")
    compare_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
