"""Job A's cost table (protocol v2.2 verify_schedule.cost_curve): the time of a whole variable-length round of B rows
that each verify L positions, for every B and L of the grid, on fresh requests of context_len random tokens.

Every round takes the scheduled path, the confidence logits and their copy to the CPU included, with a schedule that
gives every row L. A round's time is its phase timings summed, propose through commit_copy; the engine step's time
is kept beside it. Load time, reserved memory and torch.compile's counters are kept too: this is the first run of
the variable-length graphs. tools/fit_confidence_sts.py turns the output into the frozen verify schedule.

  python tools/profile_verify_cost.py --protocol benchmarks/protocol-v2.2.json --models-root DIR --commit C --out FILE
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_spec import (assert_gpu, engine_kwargs, gpu_state, installed_packages, kv_block_bytes,  # noqa: E402
                        kv_budget_bytes, manifest)
from nanovllm import LLM, SamplingParams  # noqa: E402
from summarize_bench import percentile  # noqa: E402

SCHEMA = "verify-cost-v1"


class EveryRow:
    """A verify schedule that gives every row the same length."""

    def __init__(self, length: int) -> None:
        self.length = length

    def verify_lengths(self, confidence_logits: np.ndarray) -> list[int]:
        assert np.isfinite(confidence_logits).all(), "non-finite confidence logits"
        return [self.length] * confidence_logits.shape[0]


def round_totals(phases: list[dict]) -> list[float]:
    """Each round's phase durations summed; a round's phases begin with propose."""
    totals = []
    for phase in phases:
        if phase["phase"] == "propose":
            totals.append(0.0)
        totals[-1] += phase["duration_ms"]
    return totals


def time_cell(llm, batch_size: int, length: int, curve: dict, params, generator: torch.Generator) \
        -> tuple[list[float], list[float]]:
    """batch_size fresh requests of curve["context_len"] random tokens, prefilled, then curve["warmup"] and
    curve["iters"] rounds in which every row verifies `length` positions. Returns the timed rounds' phase totals and
    their engine step times."""
    for _ in range(batch_size):
        llm.add_request(torch.randint(llm.speculative.vocab_size, (curve["context_len"],), generator=generator).tolist(),
                        params)
    while llm.scheduler.waiting:
        llm.step()                                               # prefill
    llm.speculative.schedule = EveryRow(length)
    for _ in range(curve["warmup"]):
        llm.step()
    llm.start_measurement()
    for _ in range(curve["iters"]):
        llm.step()
    steps = llm.step_timings()
    assert len(steps) == curve["iters"] and all(step["kind"] != "prefill" and step["batch_size"] == batch_size
                                                for step in steps)
    totals = round_totals(llm.phase_timings())
    for seq in list(llm.scheduler.running):
        llm.scheduler._finish(seq)
    return totals, [step["duration_ms"] for step in steps]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.2.json")
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    curve = protocol["verify_schedule"]["cost_curve"]
    state_start = gpu_state()
    assert_gpu(state_start, protocol["gpu"]["name"])
    line, largest = protocol["lines"][curve["line"]], max(curve["batch"])
    max_tokens = (curve["warmup"] + curve["iters"]) * max(curve["query_len"]) + 1    # no request finishes in a cell
    panel = [{"prompt_token_ids": [0] * curve["context_len"]}]
    seed = protocol["seeds"]["dev"][0]
    target, _, kwargs = engine_kwargs(protocol, line, panel, largest, seed, args.models_root)
    del kwargs["verify_schedule"]                 # the rounds take EveryRow, not the schedule this table goes into
    kwargs["kv_cache_memory_bytes"] = kv_budget_bytes(panel, largest, max_tokens, line["k"], kv_block_bytes(target))
    load_start = time.perf_counter()
    llm = LLM(target, **kwargs)
    load_seconds = time.perf_counter() - load_start
    memory_reserved_bytes = torch.cuda.memory_reserved()
    sampling = protocol["sampling"]
    params = SamplingParams(temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
                            ignore_eos=True, max_tokens=max_tokens)
    generator = torch.Generator().manual_seed(seed)
    cells = {(batch_size, length): time_cell(llm, batch_size, length, curve, params, generator)
             for batch_size in curve["batch"] for length in curve["query_len"]}
    llm.exit()

    def table(statistic, index: int) -> list[list[float]]:
        return [[statistic(cells[(batch_size, length)][index]) for length in curve["query_len"]]
                for batch_size in curve["batch"]]

    result = {
        "schema": SCHEMA,
        "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        "manifest": {**manifest(args.commit), "node": state_start["node"], "packages_sha256": installed_packages()[1]},
        "gpu_state_start": state_start, "gpu_state_end": gpu_state(),
        "config": {"line": curve["line"], "batch_sizes": curve["batch"], "query_lens": curve["query_len"],
                   "context_len": curve["context_len"], "warmup": curve["warmup"], "iters": curve["iters"],
                   "max_num_seqs": largest, "seed": seed},
        "load_seconds": load_seconds,
        "memory_reserved_bytes": memory_reserved_bytes,
        "dynamo_counters": {name: dict(counter) for name, counter in torch._dynamo.utils.counters.items()},
        "round_ms": table(statistics.median, 0),
        "round_ms_p95": table(lambda values: percentile(values, 95), 0),
        "step_ms": table(statistics.median, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "load_seconds": round(load_seconds, 1),
                      "round_ms_at_largest_batch": result["round_ms"][-1]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
