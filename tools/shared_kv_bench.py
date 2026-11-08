"""Shared or private blocks under the same step (protocol v2.5 shared_kv): does a step get faster when every row reads
one physical copy of the shared prefix, rather than a copy of its own?

On the line's engine, for each C, the first C prompts of the panel are prefilled once with hashing on, after the warm-up
panel's first prompt, so every row's shared blocks are one physical copy; every row's first token is forced to the same
token. Then every row's shared blocks are copied, target and draft K/V, into blocks of its own. The two layouts are one
state but for the block tables, which point the rows at the one copy or at their own: the same values, tokens,
positions and graphs. Each layout, alternating over the repeats, runs the same step at the same position
warmup + steps times, committing nothing: k0's decode forward; a drafting line's draft block forward (not the whole
propose: no logits, Markov or sampling) and its verification of k fixed drafts. Timed with CUDA events; reports per C
and part the median of the timed steps per repeat, and d = 1 - shared / private of the medians over the repeats.

  python tools/shared_kv_bench.py --line LINE --protocol FILE --models-root DIR --commit C --out FILE
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bench_spec import assert_gpu, engine_kwargs, gpu_state, installed_packages, manifest  # noqa: E402
from gate_prefix import force_first  # noqa: E402
from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.utils.context import reset_context  # noqa: E402

FORCED_TOKEN = 13    # every row's every token from the first on: what it is does not matter, only that both layouts agree


def admit(llm, prompts: list[list[int]]) -> list:
    """Prefill the prompts, as many steps as the scheduler takes."""
    seqs = [llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)) for prompt in prompts]
    while llm.scheduler.waiting:
        llm.step()
    assert all(seq.num_completion_tokens == 1 for seq in seqs)
    return seqs


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def caches(llm) -> list[torch.Tensor]:
    """The target's KV and, with a draft, the draft's: [K/V, layer, block, ...] both, under the same block ids."""
    return [llm.model_runner.kv_cache] + ([llm.speculative.drafter.kv_cache] if llm.speculative is not None else [])


@torch.inference_mode()
def build_state(llm, prompts: list[list[int]], donor: list[int], shared_tokens: int):
    """The rows prefilled sharing their first shared_tokens / block blocks, and a private copy of those for every row."""
    manager = llm.scheduler.block_manager
    manager.hash_to_block_id.clear()
    manager.enable_prefix_cache = True
    for seq in admit(llm, [donor]):
        llm.scheduler._finish(seq)
    seqs = admit(llm, prompts)
    manager.enable_prefix_cache = False
    count = shared_tokens // manager.block_size
    shared = list(seqs[0].block_table[:count])
    assert all(seq.prefix_hit_tokens == shared_tokens and seq.block_table[:count] == shared for seq in seqs)
    force_first(seqs, [FORCED_TOKEN] * len(seqs))
    k = llm.config.num_speculative_tokens
    for seq in seqs:
        manager.reserve(seq, len(seq) + k)
    private = []
    for _ in seqs:
        ids = [manager._allocate_block() for _ in range(count)]
        for cache in caches(llm):
            cache[:, :, ids] = cache[:, :, shared]
        private.append(ids)
    return seqs, shared, private


def teardown(llm, seqs, shared: list[int], private: list[list[int]]) -> None:
    manager = llm.scheduler.block_manager
    for seq in seqs:
        seq.block_table[:len(shared)] = shared
        llm.scheduler._finish(seq)
    for ids in private:
        for block_id in ids:
            manager.blocks[block_id].ref_count -= 1
            manager._deallocate_block(block_id)
    manager.hash_to_block_id.clear()


def timed_steps(llm, seqs, steps: int) -> list[dict[str, tuple]]:
    """The same step `steps` times; per step the CUDA events that bound each part."""
    marks, decoder = [], llm.speculative
    if decoder is None:
        input_ids, positions = llm.model_runner.prepare_decode(seqs)
        for _ in range(steps):
            start = event()
            llm.model_runner.run_model(input_ids, positions, False)
            marks.append({"decode": (start, event())})
        reset_context()
        return marks
    drafts = torch.full((len(seqs), decoder.num_drafts), FORCED_TOKEN, dtype=torch.int64, device=decoder.device)
    for _ in range(steps):
        start = event()
        decoder.drafter._forward_graph(seqs)
        drafted = event()
        decoder.verify_logits(seqs, drafts, decoder._aux_buffer(len(seqs) * (decoder.num_drafts + 1)))
        marks.append({"draft_forward": (start, drafted), "verify": (drafted, event())})
    return marks


@torch.inference_mode()
def measure(llm, seqs, table: list[list[int]], count: int, spec: dict) -> dict[str, float]:
    """The rows pointed at `table` (each row's first `count` block ids), then warmup + steps timed steps: medians."""
    for seq, ids in zip(seqs, table):
        seq.block_table[:count] = ids
    marks = timed_steps(llm, seqs, spec["warmup"] + spec["steps"])
    torch.cuda.synchronize()
    timed = marks[spec["warmup"]:]
    return {part: statistics.median(a.elapsed_time(b) for a, b in (mark[part] for mark in timed)) for part in timed[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models-root", default="/content/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    state_start = gpu_state()
    assert_gpu(state_start, protocol["gpu"]["name"])
    spec = protocol["shared_kv"]
    line = protocol["lines"][args.line]
    panel_spec = protocol["panels"][spec["panel"]]
    panel = json.loads((ROOT / panel_spec["file"]).read_text())
    donor = json.loads((ROOT / protocol["panels"][panel_spec["warmup"]]["file"]).read_text())[0]["prompt_token_ids"]
    largest = max(spec["concurrency"])
    # C rows sharing the prefix plus a private copy of it for every row fit in the protocol's budget for C rows: a row
    # takes at most ceil((3,341 + 8) / 256) = 14 blocks, so 8 shared + C x 6 + C x 8 private copies <= C x 16
    target, _, kwargs = engine_kwargs(protocol, line, panel[:largest], largest, protocol["seeds"]["dev"][0],
                                      args.models_root)
    kwargs.update(enable_prefix_cache=True)
    kwargs.pop("verify_schedule", None)
    kwargs.pop("variable_length", None)
    llm = LLM(target, **kwargs)
    cells = []
    for concurrency in spec["concurrency"]:
        seqs, shared, private = build_state(llm, [entry["prompt_token_ids"] for entry in panel[:concurrency]], donor,
                                            panel_spec["shared_tokens"])
        runs = {"private": [], "shared": []}
        for repeat in range(spec["repeats"]):
            for layout in (("private", "shared") if repeat % 2 == 0 else ("shared", "private")):
                table = private if layout == "private" else [shared] * len(seqs)
                runs[layout].append(measure(llm, seqs, table, len(shared), spec))
        teardown(llm, seqs, shared, private)
        parts = list(runs["private"][0])
        median = {layout: {part: statistics.median(run[part] for run in runs[layout]) for part in parts}
                  for layout in runs}
        cells.append({"concurrency": concurrency, "runs": runs, "median_ms": median,
                      "d": {part: 1 - median["shared"][part] / median["private"][part] for part in parts}})
        print(json.dumps({"C": concurrency, "d": cells[-1]["d"], "median_ms": median}))
    llm.exit()
    _, packages_sha256 = installed_packages()
    files = [ROOT / panel_spec["file"], ROOT / protocol["panels"][panel_spec["warmup"]]["file"]]
    report = {"tool": "shared-kv-bench", "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
              "manifest": {**manifest(args.commit), "node": state_start["node"], "packages_sha256": packages_sha256,
                           "panel_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
                           "warmup_panel_sha256": hashlib.sha256(files[1].read_bytes()).hexdigest()},
              "gpu_state_start": state_start, "gpu_state_end": gpu_state(), "line": args.line, "panel": spec["panel"],
              "config": {key: kwargs.get(key) for key in ("max_model_len", "max_num_seqs", "kv_cache_memory_bytes",
                                                          "draft_type", "num_speculative_tokens")},
              "cells": cells}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
