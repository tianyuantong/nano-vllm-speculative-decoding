"""Shared or private blocks under the same rounds (protocol v2.5 shared_kv): does a step get faster when every row reads
one physical copy of the shared prefix, rather than a copy of its own?

On the line's engine, for each C, the first C prompts of the panel, in two layouts:
  private  hashing off: every row prefills its whole prompt, so it owns a copy of the shared prefix's K/V;
  shared   hashing on: the warm-up panel's first prompt prefilled and finished first, then every prompt hits its
           shared blocks, so every row reads the one copy.
Then every row's first token is forced to the same token, and warmup + iters steps run in which every row appends that
token (k0's decode step) or verifies the same k drafts and commits 3 tokens (a drafting line's round: the draft's block
forward, the target's verification, after_verify), so both layouts see the same tokens, context lengths, graphs and
verified lengths; only whether rows share physical blocks differs. Each layout runs `repeats` times, alternating.
Timed with CUDA events per part; reports the median of the timed steps per (C, layout, repeat), and d = 1 - shared /
private of each part's median over the repeats.

  python tools/shared_kv_bench.py --line LINE --protocol FILE --models-root DIR --commit C --out FILE
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bench_spec import engine_kwargs  # noqa: E402
from gate_prefix import commit, force_first  # noqa: E402
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


def run_steps(llm, seqs, steps: int) -> list[dict[str, tuple]]:
    """`steps` forced steps; per step the CUDA events that bound each part."""
    marks, decoder = [], llm.speculative
    for _ in range(steps):
        if decoder is None:
            for seq in seqs:
                llm.scheduler.block_manager.reserve(seq, len(seq))
            input_ids, positions = llm.model_runner.prepare_decode(seqs)
            start = event()
            llm.model_runner.run_model(input_ids, positions, False)
            end = event()
            reset_context()
            commit(seqs, [[FORCED_TOKEN]] * len(seqs))
            marks.append({"decode": (start, end)})
            continue
        k, drafter = decoder.num_drafts, decoder.drafter
        for seq in seqs:
            llm.scheduler.block_manager.reserve(seq, len(seq) + k)
        drafts = torch.full((len(seqs), k), FORCED_TOKEN, dtype=torch.int64, device=decoder.device)
        start = event()
        drafter._forward_graph(seqs)
        proposed = event()
        aux = decoder._aux_buffer(len(seqs) * (k + 1))
        decoder.verify_logits(seqs, drafts, aux)
        verified = event()
        drafter.after_verify(seqs, aux)
        stored = event()
        commit(seqs, [[FORCED_TOKEN] * 3] * len(seqs))
        marks.append({"propose": (start, proposed), "verify": (proposed, verified), "ctx_store": (verified, stored)})
    return marks


@torch.inference_mode()
def measure(llm, prompts: list[list[int]], donor: list[int], layout: str, spec: dict) -> dict:
    manager = llm.scheduler.block_manager
    manager.hash_to_block_id.clear()
    manager.enable_prefix_cache = layout == "shared"
    if layout == "shared":
        for seq in admit(llm, [donor]):
            llm.scheduler._finish(seq)
    seqs = admit(llm, prompts)
    hits = [seq.prefix_hit_tokens for seq in seqs]
    force_first(seqs, [FORCED_TOKEN] * len(seqs))
    marks = run_steps(llm, seqs, spec["warmup"] + spec["iters"])
    torch.cuda.synchronize()
    timed = marks[spec["warmup"]:]
    for seq in seqs:
        llm.scheduler._finish(seq)
    manager.hash_to_block_id.clear()
    manager.enable_prefix_cache = False
    return {"hits": sorted(set(hits)), "context_lens": [len(seq) for seq in seqs[:1]],
            "median_ms": {part: statistics.median(a.elapsed_time(b) for a, b in (mark[part] for mark in timed))
                          for part in timed[0]}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models-root", default="/content/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["shared_kv"]
    line = protocol["lines"][args.line]
    panel = json.loads((ROOT / protocol["panels"][spec["panel"]]["file"]).read_text())
    donor = json.loads((ROOT / protocol["panels"][protocol["panels"][spec["panel"]]["warmup"]]["file"]).read_text())[0]
    largest = max(spec["concurrency"])
    target, _, kwargs = engine_kwargs(protocol, line, panel[:largest] + [donor], largest, protocol["seeds"]["dev"][0],
                                      args.models_root)
    kwargs.update(enable_prefix_cache=True)
    kwargs.pop("verify_schedule", None)
    kwargs.pop("variable_length", None)
    llm = LLM(target, **kwargs)
    cells = []
    for concurrency in spec["concurrency"]:
        prompts = [entry["prompt_token_ids"] for entry in panel[:concurrency]]
        runs = {"private": [], "shared": []}
        for repeat in range(spec["repeats"]):
            for layout in (("private", "shared") if repeat % 2 == 0 else ("shared", "private")):
                runs[layout].append(measure(llm, prompts, donor["prompt_token_ids"], layout, spec))
        parts = list(runs["private"][0]["median_ms"])
        median = {layout: {part: statistics.median(run["median_ms"][part] for run in runs[layout]) for part in parts}
                  for layout in runs}
        cells.append({"concurrency": concurrency, "runs": runs, "median_ms": median,
                      "d": {part: 1 - median["shared"][part] / median["private"][part] for part in parts}})
        print(json.dumps({"C": concurrency, "d": cells[-1]["d"], "median_ms": median}))
    llm.exit()
    hits_ok = all(set(run["hits"]) == {0} for cell in cells for run in cell["runs"]["private"]) and \
        all(set(run["hits"]) == {protocol["panels"][spec["panel"]]["shared_tokens"]}
            for cell in cells for run in cell["runs"]["shared"])
    report = {"tool": "shared-kv-bench", "engine_commit": args.commit, "line": args.line, "panel": spec["panel"],
              "hits_as_designed": hits_ok, "cells": cells}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if hits_ok else 1


if __name__ == "__main__":
    sys.exit(main())
