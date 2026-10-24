"""G3 follow-up: at G3 divergence points, compare decode-path and verify-path logits on identical KV.

For each selected divergence: prefill prompt + ordinary tokens up to the position before the
divergence (both paths then share this KV), run one ordinary decode step and one verification
forward whose first query token is the same token, and report the max |delta| between the two
logit rows plus both argmaxes. A verification bug shows as a large decode-vs-verify delta;
kernel noise shows as deltas at the G1 level with argmax flips only at small margins.
"""
import argparse
import json

import torch

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

GIB = 1 << 30


def prefill(llm, token_ids: list[int]) -> tuple[Sequence, torch.Tensor]:
    """Prefill a fresh sequence; returns it (KV covers all tokens) and the last-position fp32 logits."""
    seq = Sequence(token_ids)
    llm.scheduler.block_manager.allocate(seq, 0)
    llm.scheduler.draft_block_manager.allocate(seq, 0)
    seq.num_scheduled_tokens = len(token_ids)
    input_ids, positions = llm.model_runner.prepare_prefill([seq])
    logits = llm.model_runner.run_model(input_ids, positions, True)[-1].float()
    reset_context()
    seq.num_cached_tokens = len(token_ids)
    return seq, logits


def probe(llm, prompt: list[int], ordinary: list[int], index: int, k: int) -> dict:
    prefix = prompt + ordinary[:index]                 # tokens before the divergence; the last one is the decode input
    seq, prefill_logits = prefill(llm, prefix[:-1])
    seq.append_token(prefix[-1])                       # state: KV covers len-1, last token is the next input
    # decode path
    llm.scheduler.block_manager.reserve(seq, len(seq))
    input_ids, positions = llm.model_runner.prepare_decode([seq])
    decode_logits = llm.model_runner.run_model(input_ids, positions, False)[0].float()
    reset_context()
    # verification path on the same KV: drafts are the ordinary continuation (any tokens would do)
    llm.scheduler.block_manager.reserve(seq, len(seq) + k)
    drafts = torch.tensor([ordinary[index:index + k]], device=llm.model_runner.device)
    verify_logits = llm.speculative.verify_logits([seq], drafts)[0, 0].float()
    llm.scheduler.block_manager.deallocate(seq)
    llm.scheduler.draft_block_manager.deallocate(seq)
    return {
        "index": index,
        "position_mod_block": (len(prefix) - 1) % llm.config.kvcache_block_size,
        "decode_vs_verify_max_delta": (decode_logits - verify_logits).abs().max().item(),
        "prefill_vs_decode_max_delta": (prefill_logits - decode_logits).abs().max().item(),
        "argmax_prefill": int(prefill_logits.argmax()),
        "argmax_decode": int(decode_logits.argmax()),
        "argmax_verify": int(verify_logits.argmax()),
        "decode_top2_margin": float(decode_logits.topk(2).values.diff().abs()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--g3", required=True)
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--panel", default="benchmarks/inputs/panel-48.json")
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.load(open(args.g3))
    ordinary = {entry["id"]: entry["token_ids"] for entry in json.load(open(args.ordinary))["outputs"]}
    panel = {entry["id"]: entry["prompt_token_ids"] for entry in json.load(open(args.panel))}
    selected = [d for d in report["divergences"] if d["top2_margin"] >= args.min_margin]
    selected += [d for d in report["divergences"] if d["top2_margin"] < args.min_margin][:4]     # a few near-ties as controls
    llm = LLM(args.target, draft_model=args.draft, num_speculative_tokens=args.k, max_model_len=4096, max_num_seqs=1,
              enable_prefix_cache=False, kv_cache_memory_bytes=4 * GIB, draft_kv_cache_memory_bytes=2 * GIB)
    rows = []
    for divergence in selected:
        row = probe(llm, panel[divergence["id"]], ordinary[divergence["id"]], divergence["index"], args.k)
        row.update(id=divergence["id"][:8], g3_margin=divergence["top2_margin"],
                   ordinary_token=divergence["ordinary_token"], speculative_token=divergence["speculative_token"])
        rows.append(row)
        print(json.dumps(row))
    llm.exit()
    with open(args.output, "w") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
