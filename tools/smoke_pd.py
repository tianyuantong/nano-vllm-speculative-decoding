"""GPU smoke test for `DisaggLLM` (plain decoding + DSpark speculative decoding + two-thread stream overlap)."""
from __future__ import annotations

import argparse
import atexit
import gc
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanovllm import LLM, SamplingParams
from nanovllm.disagg import DisaggLLM

PROMPTS = [
    "Explain how paged attention manages KV cache memory in large language model serving.",
    "Write a Python function that merges two sorted lists in linear time.",
    "What is the difference between tensor parallelism and pipeline parallelism?",
    "Compute the sum of the first 20 positive integers step by step.",
]


def first_decode_rounds(outputs: list[dict]) -> list[int]:
    """Per request, the decode round that delivered its first token, counted from the first round of this pass."""
    firsts = [out["d_deliveries"][0][0] for out in outputs]
    return [step - min(firsts) for step in firsts]


def first_diff(left: list[int], right: list[int]) -> int | None:
    """Index of the first token that differs, or None when the two sequences are equal."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def check_line(
    label: str,
    model_dir: str,
    draft_dir: str | None,
    kv_bytes: int,
    max_tokens: int,
    prefill_device: int = 0,
    decode_device: int = 0,
) -> dict:
    common = dict(
        max_num_seqs=8,
        max_model_len=1024,
        enable_prefix_cache=False,
        kv_cache_memory_bytes=kv_bytes,
        seed=0,
    )
    if draft_dir is not None:
        common.update(draft_model=draft_dir, draft_type="dspark")

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # 1. Reference single-engine LLM
    ref_llm = LLM(model_dir, **common)
    try:
        ref_out = ref_llm.generate(PROMPTS, params, use_tqdm=False)
    finally:
        ref_llm.exit()
        atexit.unregister(ref_llm.exit)
        ref_llm.speculative = None
    del ref_llm
    gc.collect()
    torch.cuda.empty_cache()

    # 2. DisaggLLM (two-thread pipeline + serial step check)
    disagg = DisaggLLM(model_dir, prefill_device=prefill_device, decode_device=decode_device, **common)
    try:
        p_total_blocks = len(disagg.prefill_engine.scheduler.block_manager.blocks)
        d_total_blocks = len(disagg.decode_engine.scheduler.block_manager.blocks)
        serial_out = disagg.generate(PROMPTS, params, threaded=False)
        threaded_out = disagg.generate(PROMPTS, params, threaded=True)
        p_free_after = len(disagg.prefill_engine.scheduler.block_manager.free_block_ids)
        d_free_after = len(disagg.decode_engine.scheduler.block_manager.free_block_ids)
    finally:
        disagg.exit()
    del disagg
    gc.collect()
    torch.cuda.empty_cache()

    assert p_free_after == p_total_blocks, f"P leaked blocks: {p_free_after}/{p_total_blocks}"
    assert d_free_after == d_total_blocks, f"D leaked blocks: {d_free_after}/{d_total_blocks}"

    ref_ids = [r["token_ids"] for r in ref_out]
    serial_ids = [r["token_ids"] for r in serial_out]
    threaded_ids = [r["token_ids"] for r in threaded_out]
    # The serial pass waits for every hand-off copy, so its sequences join decode in the same round, as in the
    # reference. The worker threads admit each sequence when its own copy lands: across two cards the first one can
    # decode a round alone, and a round at another batch size can flip a near-tie under greedy decoding.
    assert serial_ids == ref_ids, f"{label}: DisaggLLM vs reference LLM mismatch"
    threaded_rounds = first_decode_rounds(threaded_out)
    has_same_rounds = threaded_rounds == first_decode_rounds(serial_out)
    assert not has_same_rounds or threaded_ids == serial_ids, f"{label}: serial vs threaded mismatch on the same rounds"

    ttft_ms = [round(r["ttft_ms"], 2) for r in threaded_out]
    itl_p50_ms = [round(sorted(r["itl_ms"])[len(r["itl_ms"]) // 2], 2) for r in threaded_out if r["itl_ms"]]
    return {
        "line": label,
        "exact_match_ref": True,
        "exact_match_serial_vs_threaded": threaded_ids == serial_ids,
        "threaded_first_decode_round": threaded_rounds,
        "threaded_first_diff": [first_diff(t, s) for t, s in zip(threaded_ids, serial_ids)],
        "p_blocks_reclaimed": f"{p_free_after}/{p_total_blocks}",
        "d_blocks_reclaimed": f"{d_free_after}/{d_total_blocks}",
        "ttft_ms": ttft_ms,
        "itl_median_ms": itl_p50_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("/content/models"))
    parser.add_argument("--dspark-dir", type=str, default="dspark_qwen3_8b_block7-03326e504381")
    parser.add_argument("--kv-gb", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-device", type=int, default=0)
    parser.add_argument("--decode-device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_dir = str(args.models_root / "Qwen3-8B")
    draft_dir = str(args.models_root / args.dspark_dir)
    kv_bytes = args.kv_gb << 30

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "prefill_device": args.prefill_device,
        "decode_device": args.decode_device,
        "k0": check_line("k0", model_dir, None, kv_bytes, args.max_tokens, args.prefill_device, args.decode_device),
        "dspark_k7": check_line(
            "dspark_k7", model_dir, draft_dir, kv_bytes, args.max_tokens, args.prefill_device, args.decode_device
        ),
    }
    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
