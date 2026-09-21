"""G1: verification logits against step-by-step decode logits on the same prefix.

Sequence A decodes k+1 tokens greedily through the ordinary decode path and records the
logits of each step. Sequences B, C, D share A's prompt and only prefilled; B is verified
alone (graph batch 1) with A's tokens as drafts, and B, C, D together (padded to 4) to
check that padding rows leave B's logits unchanged. Reports max |delta| per position.
"""
import argparse
import json

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import reset_context

GIB = 1 << 30
NOISE_FLOOR_WARNING = 0.5     # bf16 kernels typically differ by 1e-2..1e-1; a larger gap signals a bug


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def decode_step(llm, seq) -> torch.Tensor:
    """One ordinary decode step for `seq` outside the scheduler; returns the fp32 logits row."""
    runner = llm.model_runner
    llm.scheduler.block_manager.reserve(seq, len(seq))
    input_ids, positions = runner.prepare_decode([seq])
    logits = runner.run_model(input_ids, positions, False)
    reset_context()
    seq.target_kv.num_cached_tokens = len(seq)
    seq.append_token(int(logits.argmax(dim=-1)))
    return logits[0].float()


def main():
    args = parse_args()
    llm = LLM(args.target, draft_model=args.draft, num_speculative_tokens=args.k, max_model_len=4096,
              max_num_seqs=4, enable_prefix_cache=False, kv_cache_memory_bytes=4 * GIB,
              draft_kv_cache_memory_bytes=2 * GIB, seed=0)
    params = SamplingParams(temperature=0, max_tokens=args.k + 2)
    for _ in range(4):
        llm.add_request(args.prompt, params)
    outputs, _ = llm.step()                                   # one prefill batch: A, B, C, D get the same first token
    assert not outputs
    a, b, c, d = list(llm.scheduler.running)
    assert a.last_token == b.last_token == c.last_token == d.last_token

    reference = torch.stack([decode_step(llm, a) for _ in range(args.k + 1)])     # [k+1, V]
    drafts = torch.tensor([a.completion_token_ids[1:args.k + 1]], device=llm.model_runner.device)

    for seq in (b, c, d):
        llm.scheduler.block_manager.reserve(seq, len(seq) + args.k)
    single = llm.speculative.verify_logits([b], drafts)[0].float()                      # [k+1, V]
    padded = llm.speculative.verify_logits([b, c, d], drafts.expand(3, -1))[0].float()

    per_position = []
    for position in range(args.k + 1):
        delta = (single[position] - reference[position]).abs().max().item()
        per_position.append({
            "position": position,
            "max_abs_delta": delta,
            "argmax_agrees": bool(single[position].argmax() == reference[position].argmax()),
            "top2_margin_reference": float(reference[position].topk(2).values.diff().abs()),
        })
    report = {
        "k": args.k,
        "per_position": per_position,
        "padding_max_abs_delta": (single - padded).abs().max().item(),
        "noise_floor": max(entry["max_abs_delta"] for entry in per_position),
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    llm.exit()
    assert report["noise_floor"] < NOISE_FLOOR_WARNING, "verification logits differ from decode beyond bf16 noise"
    assert report["padding_max_abs_delta"] < NOISE_FLOOR_WARNING, "padding rows changed a real row"


if __name__ == "__main__":
    main()
