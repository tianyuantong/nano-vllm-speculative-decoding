"""Reference DFlash numbers from DeepSpec's own inference code (port plan Task 12).

For the first D48 prompts, measures tokens per round of a target + DeepSpec DFlash draft with DeepSpec's
generate_decoding_sample, and saves fixtures (target aux features -> the draft block's logits) for the
port's numerical checks. Batch 1 and temperature-only sampling: DeepSpec's evaluator has no top-k/top-p.

  python tools/ref_dflash_deepspec.py --target DIR --draft DIR --deepspec DIR --panel FILE --out DIR
"""
import argparse
import json
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch


def load_pair(target_dir: str, draft_dir: str, deepspec_dir: str, device: str, dtype: torch.dtype) -> tuple:
    """HF target (sdpa) + DeepSpec's Qwen3DSparkModel, as Qwen3DSparkEvaluator.build_models loads them."""
    sys.path.insert(0, deepspec_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspec.eval.base_evaluator import assert_no_final_target_layer
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    target = AutoModelForCausalLM.from_pretrained(target_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    draft = Qwen3DSparkModel.from_pretrained(draft_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    assert_no_final_target_layer(target, draft.target_layer_ids)
    return target, draft, AutoTokenizer.from_pretrained(target_dir)


@torch.inference_mode()
def dump_fixture(prompt_ids: list[int], target, draft, out_dir: Path, name: str) -> None:
    """The prompt as context, the target's greedy next token as the anchor, and the draft block's 7 logits rows."""
    from transformers import DynamicCache
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
    from deepspec.modeling.dspark.common import extract_context_feature

    device = next(target.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)
    output = target(input_ids=input_ids, output_hidden_states=True)
    aux = extract_context_feature(output.hidden_states, draft.target_layer_ids)          # [1, L, 5H]
    anchor_position = len(prompt_ids)
    anchor_token = int(output.logits[0, -1].argmax())
    block_ids = torch.full((1, draft.block_size), draft.mask_token_id, dtype=torch.long, device=device)
    block_ids[0, 0] = anchor_token
    position_ids = torch.arange(anchor_position + draft.block_size, device=device).unsqueeze(0)
    block_hidden = forward_dspark_draft_block(draft, draft_input_ids=block_ids, position_ids=position_ids,
                                              past_key_values_draft=DynamicCache(), target_hidden_states=aux,
                                              start=anchor_position, block_size=draft.block_size)
    logits = draft.compute_logits(block_hidden[:, :draft.block_size])                      # [1, 7, V]
    torch.save({"prompt_ids": prompt_ids, "aux": aux[0].cpu(), "anchor_token": anchor_token,
                "anchor_position": anchor_position, "block_logits": logits[0].cpu()}, out_dir / f"{name}.pt")


def measure_tokens_per_round(prompts: list[list[int]], target, draft, temperature: float, max_new_tokens: int,
                             seed: int, stop_token_ids: list[int] | None) -> dict:
    """DeepSpec's decoding loop with Qwen3DSparkEvaluator's _init_context/_propose/_update, prompt by prompt."""
    from transformers import DynamicCache
    from deepspec.eval.base_evaluator import generate_decoding_sample
    from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
    from deepspec.modeling.dspark.common import extract_context_feature

    def init_context(*, initial_output, **kwargs):
        return SimpleNamespace(past_key_values_draft=DynamicCache(),
                               target_hidden_states=extract_context_feature(initial_output.hidden_states,
                                                                            draft.target_layer_ids))

    def propose(*, context, output_ids, position_ids, start, stop_token_ids=None):
        block_ids = torch.full((1, draft.block_size), draft.mask_token_id, dtype=torch.long, device=output_ids.device)
        block_ids[:, 0] = output_ids[:, start]
        block_hidden = forward_dspark_draft_block(draft, draft_input_ids=block_ids, position_ids=position_ids,
                                                  past_key_values_draft=context.past_key_values_draft,
                                                  target_hidden_states=context.target_hidden_states,
                                                  start=start, block_size=draft.block_size)
        return build_dspark_proposal(model=draft, draft_input_ids=block_ids, block_hidden=block_hidden,
                                     block_size=draft.block_size, temperature=temperature, confidence_threshold=0.0)

    def update(context, verification):
        features = extract_context_feature(verification.target_output.hidden_states, draft.target_layer_ids)
        context.target_hidden_states = features[:, :verification.accepted_draft_tokens + 1, :]

    torch.manual_seed(seed)
    device = next(target.parameters()).device
    rounds, per_prompt = [], []
    for prompt_ids in prompts:
        sample = generate_decoding_sample(target_model=target, input_ids=torch.tensor([prompt_ids], device=device),
                                          max_new_tokens=max_new_tokens, max_proposal_tokens=draft.block_size,
                                          temperature=temperature, stop_token_ids=stop_token_ids,
                                          init_context=init_context, propose=propose, update=update)
        rounds += sample.acceptance_lengths
        per_prompt.append({"output_tokens": int(sample.num_output_tokens), "rounds": len(sample.acceptance_lengths),
                           "tokens_per_round": statistics.fmean(sample.acceptance_lengths)
                           if sample.acceptance_lengths else None})
    return {"temperature": temperature, "seed": seed, "rounds": len(rounds),
            "tokens_per_round": statistics.fmean(rounds), "histogram": dict(sorted(Counter(rounds).items())),
            "per_prompt": per_prompt}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--deepspec", required=True, help="DeepSpec source tree (the directory holding deepspec/)")
    parser.add_argument("--deepspec-commit", default="005e03b")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--fixture-context", type=int, default=256, help="prompt prefix length used for fixtures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    target, draft, tokenizer = load_pair(args.target, args.draft, args.deepspec, args.device, getattr(torch, args.dtype))
    from deepspec.eval.base_evaluator import resolve_stop_token_ids
    import transformers

    panel = json.loads(args.panel.read_text())[:args.prompts]
    prompts = [entry["prompt_token_ids"] for entry in panel]
    (args.out / "fixtures").mkdir(parents=True, exist_ok=True)
    for entry in panel:
        dump_fixture(entry["prompt_token_ids"][:args.fixture_context], target, draft, args.out / "fixtures", entry["id"])
    stop_token_ids = resolve_stop_token_ids(target, tokenizer)
    summary = {f"T={temperature}": measure_tokens_per_round(prompts, target, draft, temperature, args.max_new_tokens,
                                                           args.seed, stop_token_ids)
               for temperature in args.temperatures}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"target": args.target, "draft": args.draft, "deepspec_commit": args.deepspec_commit,
                "panel": str(args.panel), "prompt_ids": [entry["id"] for entry in panel],
                "max_new_tokens": args.max_new_tokens, "fixture_context": args.fixture_context,
                "stop_token_ids": stop_token_ids, "device": args.device, "dtype": args.dtype,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, result in summary.items():
        print(f"{name}: {result['tokens_per_round']:.3f} tokens per round over {result['rounds']} rounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
