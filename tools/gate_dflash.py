"""GPU checks of the block-draft port (DFlash, DSpark) against DeepSpec's reference run (tools/ref_dflash_deepspec.py),
port design §5.2.

  draft   G1-draft: nano's block forward on the real paged FlashAttention path, fed each fixture's target aux
          features, against the block logits DeepSpec computed for that fixture. Pass: argmax equal on every row
          whose reference top-2 margin exceeds 0.5 (G1-wide's rule), and max |delta| / max |logit| <= 2e-2. For
          DSpark the same rule also holds for the rows after their Markov bias, each row's previous token taken
          from the reference's greedy chain, so one flipped token cannot mismatch the rows after it.
  rounds  tokens per round of the whole nano engine with the block draft (eager, C=1, temperature-only) on the
          reference's prompts, against the reference's. Its per-run pass (|difference| <= 2 x the combined per-round
          standard error) is a diagnostic; the gate is `alignment` over several runs.
  graphs  the verify graph (it exports the aux features) and the draft block graph against their eager paths on the
          same rows. Pass: verify logits and aux, block logits and the block K/V stored in the draft's KV bit-identical
          at a captured batch size, a batch padded up to one reported only; at both, no other slot of the draft's KV
          written. For DSpark also its sampling graphs, under greedy, temperature-only and the protocol's sampling:
          drafts and q bit-identical to the eager rows from the same generator state at a captured batch size, and
          at both batch sizes the generator left where the eager rows leave it.
  alignment  the protocol's dflash_alignment rule over several `rounds` results (one per seed): per sampling, the mean
          of nano's runs against the mean of DeepSpec's. Pass: |difference| <= 2 x the standard error of that
          difference, computed from the per-run means.

  python tools/gate_dflash.py draft  --draft DIR --reference DIR --out FILE
  python tools/gate_dflash.py rounds --target DIR --draft DIR --reference DIR --panel FILE --out FILE
  python tools/gate_dflash.py graphs --target DIR --draft DIR --panel FILE --out FILE
  python tools/gate_dflash.py alignment --runs FILE [FILE ...] --out FILE
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import torch

GIB = 1 << 30
# (temperature, top_k, top_p) of the samplings DSpark's sampling graphs are checked under; rec is the protocol's.
GRAPH_SAMPLINGS = {"greedy": (0.0, 0, 1.0), "temperature": (1.0, 0, 1.0), "rec": (0.7, 20, 0.8)}


def block_draft_type(draft_dir: str) -> str:
    """dspark when the checkpoint has a Markov head, else dflash: the draft_type Config requires of it."""
    from transformers import AutoConfig
    return "dspark" if getattr(AutoConfig.from_pretrained(draft_dir), "markov_rank", 0) > 0 else "dflash"


def compare_logits(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """G1's rule on [rows, V] logits: argmax equal where the reference's top-2 margin exceeds 0.5, and the largest
    difference over the largest reference logit."""
    top2 = expected.topk(2, dim=-1).values
    decided = top2[:, 0] - top2[:, 1] > 0.5
    return {"argmax_equal_where_decided": bool(torch.equal(actual.argmax(-1)[decided], expected.argmax(-1)[decided])),
            "decided_rows": int(decided.sum()),
            "relative_error": float((actual - expected).abs().max() / expected.abs().max())}


def passes(comparison: dict) -> bool:
    return comparison["argmax_equal_where_decided"] and comparison["relative_error"] <= 2e-2


def markov_rows(logits: torch.Tensor, markov_head: Callable[[torch.Tensor], torch.Tensor], anchor_token: int,
                chain: list[int]) -> torch.Tensor:
    """Each block row plus the Markov bias of the token before it on the chain (the anchor for row 0), summed in the
    logits' dtype as the engine sums them."""
    prev = torch.tensor([anchor_token] + chain[:-1], device=logits.device)
    return logits + markov_head(prev)


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
            logits = model.compute_logits(model(input_ids, block))
            reset_context()
            row = {"fixture": fixture_path.stem, "context": length,
                   **compare_logits(logits.float().cpu(), fixture["block_logits"].float())}
            if model.markov_head is not None:
                markov = markov_rows(logits, model.markov_head, fixture["anchor_token"], fixture["markov_tokens"])
                row["markov"] = compare_logits(markov.float().cpu(), fixture["markov_logits"].float())
        rows.append(row)
    passed = all(passes(row) and ("markov" not in row or passes(row["markov"])) for row in rows)
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
    llm = LLM(target, draft_model=draft, draft_type=block_draft_type(draft), enforce_eager=True, max_num_seqs=1,
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


def check_sampling_graphs(drafter, seqs: list, generator: torch.Generator) -> list[dict]:
    """DSpark's sampling graph against its eager rows on the same block outputs, from the same generator state."""
    from types import SimpleNamespace
    from nanovllm.layers.spec_sampler import sampling_tensors

    k, rows = drafter.decoder.num_drafts, []
    for batch_size in (4, 3):
        batch = seqs[:batch_size]
        with torch.inference_mode():
            logits = drafter.model.compute_logits(drafter._forward_graph(batch)).view(batch_size, k, -1)
            for name, (temperature, top_k, top_p) in GRAPH_SAMPLINGS.items():
                params = SimpleNamespace(temperature=temperature, top_k=top_k, top_p=top_p)
                sampling = sampling_tensors([params] * batch_size, drafter.decoder.device)
                state = generator.get_state()
                eager_drafts, eager_probs = drafter._sample_markov(batch, logits, sampling)
                eager_state = generator.get_state()
                generator.set_state(state)
                graph_drafts, graph_probs = drafter._sample_markov_graph(batch_size, sampling)
                row = {"batch": batch_size, "captured_size": batch_size in drafter.decoder.graph_batch_sizes,
                       "sampling": name, "drafts_equal": torch.equal(eager_drafts, graph_drafts),
                       "generator_equal": torch.equal(eager_state, generator.get_state()),
                       "probs_equal": None, "probs_max_abs_diff": None}
                if eager_probs is not None:
                    row.update(probs_equal=torch.equal(eager_probs, graph_probs),
                               probs_max_abs_diff=float((eager_probs - graph_probs).abs().max()))
                rows.append(row)
    return rows


def sampling_row_passes(row: dict) -> bool:
    """The generator in lockstep at every batch size; drafts and q bit-identical at a captured one (greedy has no q)."""
    exact = row["drafts_equal"] and row["probs_equal"] is not False
    return row["generator_equal"] and (exact or not row["captured_size"])


def check_graphs(target: str, draft: str, panel: Path, seed: int) -> dict:
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.batch_metadata import draft_block_metadata

    llm = LLM(target, draft_model=draft, draft_type=block_draft_type(draft), max_num_seqs=8, enable_prefix_cache=False,
              kv_cache_memory_bytes=8 * GIB, seed=seed)
    spec = llm.speculative
    for entry in json.loads(panel.read_text())[:4]:
        llm.add_request(entry["prompt_token_ids"], SamplingParams(temperature=1.0, max_tokens=64))
    while llm.scheduler.waiting:
        llm.step()                                 # prefill: context K/V and the first tokens
    seqs, _ = llm.scheduler.schedule()             # a decode schedule reserves the blocks one round writes
    generator = torch.Generator(device=spec.device).manual_seed(seed)
    drafter = spec.drafter
    cache = drafter.kv_cache.flatten(2, 3)    # [K/V, layer, slot, head, dim]
    rows = []
    for batch_size in (4, 3):
        batch, query_len = seqs[:batch_size], spec.num_drafts + 1
        drafts = torch.randint(0, spec.vocab_size, (batch_size, spec.num_drafts), device=spec.device, generator=generator)
        with torch.inference_mode():
            eager_aux = torch.empty(batch_size * query_len, spec.drafter.aux_width, dtype=spec.drafter.aux_dtype,
                                    device=spec.device)
            eager = spec.target.model.compute_logits(spec._verify_eager(batch, drafts, eager_aux), all_logits=True)
            graph = spec.target.model.compute_logits(spec._verify_graph(batch, drafts), all_logits=True)
            graph_aux = spec.graph_vars["aux"][:batch_size * query_len]
            slots = torch.tensor(draft_block_metadata(batch, spec.block_size, spec.num_drafts, drafter.model.mask_token_id,
                                                      batch_size, spec.target.pad_block_id, spec.max_num_blocks).slot_mapping,
                                 device=spec.device)
            before = cache.clone()
            eager_block = drafter.model.compute_logits(drafter._forward_eager(batch))
            eager_kv = cache[:, :, slots].clone()
            cache[:, :, slots] = 0                                          # the graph must write them again
            graph_block = drafter.model.compute_logits(drafter._forward_graph(batch))
            graph_kv = cache[:, :, slots].clone()
            cache[:, :, slots] = before[:, :, slots]
            other_slots_unchanged = torch.equal(cache, before)
        rows.append({"batch": batch_size, "captured_size": batch_size in spec.graph_batch_sizes,
                     "logits_equal": torch.equal(eager, graph), "aux_equal": torch.equal(eager_aux, graph_aux),
                     "block_logits_equal": torch.equal(eager_block, graph_block),
                     "block_kv_equal": torch.equal(eager_kv, graph_kv), "other_slots_unchanged": other_slots_unchanged,
                     "logits_max_abs_diff": float((eager.float() - graph.float()).abs().max()),
                     "aux_max_abs_diff": float((eager_aux.float() - graph_aux.float()).abs().max()),
                     "block_logits_max_abs_diff": float((eager_block.float() - graph_block.float()).abs().max()),
                     "block_kv_max_abs_diff": float((eager_kv.float() - graph_kv.float()).abs().max())})
    sampling_rows = (check_sampling_graphs(drafter, seqs, spec.target.generator)
                     if drafter.model.markov_head is not None else [])
    llm.exit()
    exact = ("logits_equal", "aux_equal", "block_logits_equal", "block_kv_equal")
    passed = (all(row["other_slots_unchanged"] for row in rows)
              and all(row[key] for row in rows if row["captured_size"] for key in exact)
              and all(sampling_row_passes(row) for row in sampling_rows))
    return {"gate": "verify, draft block and sampling graphs against eager", "passed": passed,
            "threshold": "bit-identical verify logits, aux, block logits and block K/V at a captured batch size; "
                         "no other draft KV slot written at any batch size; DSpark: drafts and q bit-identical at a "
                         "captured batch size, the generator in the same state at any", "rows": rows,
            "sampling_rows": sampling_rows}


def check_alignment(paths: list[Path]) -> dict:
    runs = defaultdict(lambda: {"nano": [], "deepspec": []})
    for path in paths:
        for row in json.loads(path.read_text())["rows"]:
            runs[row["sampling"]]["nano"].append(row["nano"])
            runs[row["sampling"]]["deepspec"].append(row["deepspec"])
    rows = []
    for sampling, values in sorted(runs.items()):
        nano, deepspec = values["nano"], values["deepspec"]
        difference = statistics.fmean(nano) - statistics.fmean(deepspec)
        row = {"sampling": sampling, "nano_runs": nano, "deepspec_runs": deepspec, "difference": difference}
        if min(len(nano), len(deepspec)) < 2:
            rows.append({**row, "bound": None, "passed": False, "note": "needs at least 2 runs per side"})
            continue
        error = math.sqrt(statistics.variance(nano) / len(nano) + statistics.variance(deepspec) / len(deepspec))
        rows.append({**row, "bound": 2 * error, "passed": abs(difference) <= 2 * error})
    return {"gate": "tokens-per-round alignment over runs", "passed": all(row["passed"] for row in rows),
            "threshold": "|mean(nano runs) - mean(deepspec runs)| <= 2 x the standard error from the per-run means",
            "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["draft", "rounds", "graphs", "alignment"])
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--reference", type=Path, help="output directory of ref_dflash_deepspec.py (draft, rounds)")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--runs", type=Path, nargs="+", help="rounds results, one per seed (alignment)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.check == "draft":
        result = check_draft(args.draft, args.reference)
    elif args.check == "rounds":
        result = check_rounds(args.target, args.draft, args.reference, args.panel, args.seed)
    elif args.check == "graphs":
        result = check_graphs(args.target, args.draft, args.panel, args.seed)
    else:
        result = check_alignment(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
