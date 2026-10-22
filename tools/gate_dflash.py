"""GPU checks of the DFlash port against DeepSpec's reference run (tools/ref_dflash_deepspec.py), port design §5.2.

  draft   G1-draft: nano's DFlash block forward on the real paged FlashAttention path, fed each fixture's target
          aux features, against the block logits DeepSpec computed for that fixture. Pass: argmax equal on every
          row whose reference top-2 margin exceeds 0.5 (G1-wide's rule), and max |delta| / max |logit| <= 2e-2.
  rounds  tokens per round of the whole nano engine with the DFlash draft (eager, C=1, temperature-only) on the
          reference's prompts, against the reference's. Pass: |difference| <= 2 x the combined standard error.

  python tools/gate_dflash.py draft  --draft DIR --reference DIR --out FILE
  python tools/gate_dflash.py rounds --target DIR --draft DIR --reference DIR --panel FILE --out FILE
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

GIB = 1 << 30


def check_draft(draft_dir: str, reference_dir: Path) -> dict:
    import torch.distributed as dist
    from transformers import AutoConfig
    from nanovllm.models.dflash import DFlashDraftModel
    from nanovllm.utils.context import reset_context, set_context
    from nanovllm.utils.loader import load_model

    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
    config = AutoConfig.from_pretrained(draft_dir)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    model = DFlashDraftModel(config)
    load_model(model, draft_dir)
    block_size, heads, head_dim = 256, config.num_key_value_heads, config.head_dim
    rows = []
    for fixture_path in sorted((reference_dir / "fixtures").glob("*.pt")):
        fixture = torch.load(fixture_path)
        length = fixture["anchor_position"]
        blocks = math.ceil((length + model.block_size) / block_size)
        kv_cache = torch.zeros(2, config.num_hidden_layers, blocks, block_size, heads, head_dim)
        for layer_id, layer in enumerate(model.layers):
            layer.self_attn.attn.k_cache, layer.self_attn.attn.v_cache = kv_cache[0, layer_id], kv_cache[1, layer_id]
        with torch.inference_mode():
            context = torch.arange(length)
            model.store_context_kv(model.context_features(fixture["aux"].cuda()), context, context.to(torch.int32))
            block = torch.arange(length, length + model.block_size)
            set_context(False, slot_mapping=block.to(torch.int32),
                        context_lens=torch.tensor([length + model.block_size], dtype=torch.int32),
                        block_tables=torch.arange(blocks, dtype=torch.int32)[None],
                        query_len_per_seq=model.block_size, causal=False)
            input_ids = torch.tensor([fixture["anchor_token"]] + [model.mask_token_id] * (model.block_size - 1))
            logits = model.compute_logits(model(input_ids, block)).float().cpu()
            reset_context()
        expected = fixture["block_logits"].float()
        top2 = expected.topk(2, dim=-1).values
        decided = top2[:, 0] - top2[:, 1] > 0.5
        rows.append({"fixture": fixture_path.stem, "context": length,
                     "argmax_equal_where_decided": bool(torch.equal(logits.argmax(-1)[decided], expected.argmax(-1)[decided])),
                     "decided_rows": int(decided.sum()),
                     "relative_error": float((logits - expected).abs().max() / expected.abs().max())})
    passed = all(row["argmax_equal_where_decided"] and row["relative_error"] <= 2e-2 for row in rows)
    return {"gate": "G1-draft", "passed": passed, "threshold": "argmax where top-2 margin > 0.5; relative error <= 2e-2",
            "fixtures": rows}


def mean_and_error(histogram: dict[int, int]) -> tuple[float, float, int]:
    """Mean of tokens per round and its standard error, from {tokens per round: count}."""
    values = [value for value, count in histogram.items() for _ in range(count)]
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values)), len(values)


def check_rounds(target: str, draft: str, reference_dir: Path, panel: Path, seed: int) -> dict:
    from nanovllm import LLM, SamplingParams

    reference = json.loads((reference_dir / "summary.json").read_text())
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    by_id = {entry["id"]: entry["prompt_token_ids"] for entry in json.loads(panel.read_text())}
    prompts = [by_id[prompt_id] for prompt_id in manifest["prompt_ids"]]
    llm = LLM(target, draft_model=draft, draft_type="dflash", enforce_eager=True, max_num_seqs=1,
              enable_prefix_cache=False, kv_cache_memory_bytes=4 * GIB, seed=seed)
    rows = []
    for name, result in reference.items():
        histogram, per_prompt = Counter(), []
        for prompt in prompts:    # one at a time: generate() restarts the acceptance counts, so they are this prompt's
            llm.generate([prompt], SamplingParams(temperature=result["temperature"], max_tokens=manifest["max_new_tokens"]),
                         use_tqdm=False)
            counts = {count + 1: llm.speculative.stats[f"accepted_{count}"] for count in range(llm.speculative.num_drafts + 1)}
            histogram.update(counts)
            per_prompt.append(sum(tokens * rounds for tokens, rounds in counts.items()) / sum(counts.values()))
        nano_mean, nano_error, nano_rounds = mean_and_error({k: v for k, v in histogram.items() if v})
        ref_mean, ref_error, ref_rounds = mean_and_error({int(k): v for k, v in result["histogram"].items()})
        bound = 2 * math.hypot(nano_error, ref_error)
        rows.append({"sampling": name, "nano": nano_mean, "nano_rounds": nano_rounds, "deepspec": ref_mean,
                     "deepspec_rounds": ref_rounds, "difference": nano_mean - ref_mean, "bound": bound,
                     "passed": abs(nano_mean - ref_mean) <= bound, "nano_per_prompt": per_prompt,
                     "deepspec_per_prompt": [prompt["tokens_per_round"] for prompt in result["per_prompt"]]})
    llm.exit()
    return {"gate": "tokens-per-round alignment", "passed": all(row["passed"] for row in rows),
            "threshold": "|nano - deepspec| <= 2 x combined standard error", "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["draft", "rounds"])
    parser.add_argument("--target")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="output directory of ref_dflash_deepspec.py")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.check == "draft":
        result = check_draft(args.draft, args.reference)
    else:
        result = check_rounds(args.target, args.draft, args.reference, args.panel, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
