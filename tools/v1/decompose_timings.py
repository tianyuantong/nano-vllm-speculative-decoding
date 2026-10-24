"""Decompose archived generate() timings into per-round and per-step costs.

Every archived result JSON carries, per timed call, the VERIFY-graph replay
counters bucketed by batch size (``replay_B4_Q4`` ...) plus the count of
eager fallback rounds (``unsupported_or_disabled``). Those counts are exact,
so a least-squares fit of ``generate_seconds`` against them is a
*decomposition* of the measured time, not a model of it.

For ordinary decoding (mode B) the engine drains a batch step by step, so the
number of steps at each batch occupancy follows from the per-request output
lengths alone.

Usage::

    python tools/decompose_timings.py <dir-with-result-json> [<dir> ...]

Point it at ``evidence/clean`` inside the release archives
(``expanded-k3-results``, ``greedy-verify-sampling-results``). Output is one
table per speculative series plus the matching ordinary-decoding series.
Requires numpy only.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STEM = re.compile(r"^(?P<key>.+?)[-_](?P<mode>S1?|B)(?:_(?P<tag>[A-Za-z0-9_]+))?$")


def replay_counts(counts: dict, previous: dict) -> tuple[list[int], int]:
    """Rounds at batch size 1..4 and eager rounds since ``previous``."""
    keys = set(counts) | set(previous)
    by_batch = [
        sum(counts.get(k, 0) - previous.get(k, 0)
            for k in keys if re.match(rf"replay_B{b}_Q\d+$", k))
        for b in (1, 2, 3, 4)
    ]
    eager = counts.get("unsupported_or_disabled", 0) - previous.get("unsupported_or_disabled", 0)
    return by_batch, eager


def load_pairs(directories: list[str]):
    """Yield (series label, speculative row, ordinary row) per timed call."""
    files: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            match = STEM.match(path.stem)
            if not match:
                continue
            tag = match["tag"] or ""
            base = re.sub(r"_(old|new)$", "", tag)
            files[(match["key"], base)][match["mode"] + ("_" + tag if tag else "")] = path
    for (key, base), members in files.items():
        ordinary = next((p for m, p in members.items() if m.startswith("B")), None)
        if ordinary is None:
            continue
        b_doc = json.load(open(ordinary))
        for mode, path in members.items():
            if mode.startswith("B"):
                continue
            s_doc = json.load(open(path))
            task = s_doc["task"]
            stage = task.get("stage") or task.get("variant") or "run"
            label = f"{stage} {mode} k={task.get('k')}"
            if task.get("sampling_mode"):
                label += f" {task['sampling_mode']}"
            previous = s_doc["warm_graph"]["verify_graph"]["counts"]
            for s_row, b_row in zip(s_doc["rows"], b_doc["rows"]):
                counts = s_row["performance"]["verify_graph"]["counts"]
                by_batch, eager = replay_counts(counts, previous)
                previous = counts
                s_lens = [len(o["token_ids"]) for o in s_row["outputs"]]
                b_lens = [len(o["token_ids"]) for o in b_row["outputs"]]
                steps = max(b_lens)
                occupancy = [sum(1 for n in b_lens if n > step) for step in range(steps)]
                yield label, {
                    "seconds": s_row["generate_seconds"],
                    "by_batch": by_batch,
                    "eager": eager,
                    "rounds": sum(by_batch) + eager,
                    "longest": max(s_lens),
                }, {
                    "seconds": b_row["generate_seconds"],
                    "by_batch": [occupancy.count(b) for b in (1, 2, 3, 4)],
                    "steps": steps,
                }


def fit(rows: list[dict], eager: bool) -> tuple[np.ndarray, float]:
    design = np.array([r["by_batch"] + ([r["eager"]] if eager else []) for r in rows], float)
    target = np.array([r["seconds"] for r in rows])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    error = float(np.mean(np.abs(design @ coefficients - target) / target))
    return coefficients * 1000.0, error * 100.0


MIN_CALLS = 8  # five unknowns per fit; fewer calls than this gives a degenerate fit


def report(label: str, spec: list[dict], ordinary: list[dict]) -> None:
    tokens_per_round = [r["longest"] / r["rounds"] for r in spec if r["rounds"]]
    print(f"\n== {label}  ({len(spec)} timed calls) ==")
    if len(spec) < MIN_CALLS:
        ms_round = 1000 * sum(r["seconds"] for r in spec) / sum(r["rounds"] for r in spec)
        ms_step = 1000 * sum(r["seconds"] for r in ordinary) / sum(r["steps"] for r in ordinary)
        print(f"  too few calls for a per-batch-size fit; aggregate only:")
        print(f"  S round {ms_round:.1f} ms   B step {ms_step:.1f} ms   ratio {ms_round / ms_step:.2f}")
        print(f"  tokens per round (longest request): mean {statistics.mean(tokens_per_round):.2f}")
        return
    s_cost, s_err = fit(spec, eager=True)
    b_cost, b_err = fit(ordinary, eager=False)
    slope_s, intercept_s = np.polyfit([1, 2, 3, 4], s_cost[:4], 1)
    slope_b, intercept_b = np.polyfit([1, 2, 3, 4], b_cost, 1)
    print("                    batch=1  batch=2  batch=3  batch=4   eager   fit err")
    print(f"  S round (ms)     {s_cost[0]:7.1f}  {s_cost[1]:7.1f}  {s_cost[2]:7.1f}  {s_cost[3]:7.1f}  {s_cost[4]:6.1f}   {s_err:4.1f}%")
    print(f"  B step  (ms)     {b_cost[0]:7.1f}  {b_cost[1]:7.1f}  {b_cost[2]:7.1f}  {b_cost[3]:7.1f}       -   {b_err:4.1f}%")
    print(f"  per extra request:  S {slope_s:.2f} ms/round   B {slope_b:.2f} ms/step")
    print(f"  intercept (batch->0): S {intercept_s:.1f} ms   B {intercept_b:.1f} ms")
    print(f"  tokens per round (longest request): mean {statistics.mean(tokens_per_round):.2f}"
          f"  min {min(tokens_per_round):.2f}  max {max(tokens_per_round):.2f}")
    print(f"  break-even tokens/round at batch=4: {s_cost[3] / b_cost[3]:.2f}")
    print(f"  eager rounds per call: mean {statistics.mean(r['eager'] for r in spec):.1f}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    series: dict[str, tuple[list[dict], list[dict]]] = defaultdict(lambda: ([], []))
    for label, s_row, b_row in load_pairs(argv[1:]):
        series[label][0].append(s_row)
        series[label][1].append(b_row)
    if not series:
        print("no speculative/ordinary result pairs found")
        return 1
    for label in sorted(series):
        report(label, *series[label])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
