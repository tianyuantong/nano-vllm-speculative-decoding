"""Markdown report over a matrix of bench_spec.py results: R per (line, B), latency, round anatomy."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from summarize_bench import percentile, summarize

NAME = re.compile(r"(?P<line>[a-z]+)-B(?P<batch>\d+)-k(?P<k>\d+)\.json")


def load(paths: list[Path]) -> dict:
    results = {}
    for path in paths:
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        result = json.loads(path.read_text())
        results[(match["line"], int(match["batch"]), int(match["k"]))] = (summarize(result), result)
    return results


def step_p50(summary: dict, kind: str, batch: int) -> float | None:
    entry = summary["step_ms"].get(f"{kind}@{batch}")
    return None if entry is None else entry["p50"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    results = load(args.results)
    manifests = {json.dumps(r["manifest"], sort_keys=True) for _, r in results.values()}
    print("## Manifest\n")
    for manifest in manifests:
        print(f"- `{manifest}`")
    print("\n## Throughput and R = tok/s(k) / tok/s(k=0)\n")
    print("| line | B | k | tok/s k=0 | tok/s k | R | tokens/row/round | round p50 / step p50 | ratio | TPOT p50 k=0 → k (ms) | TPOT p95 k=0 → k |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (line, batch, k) in sorted(results):
        if k == 0:
            continue
        base = results.get((line, batch, 0))
        if base is None:
            continue
        s0, _ = base
        s, r = results[(line, batch, k)]
        ratio_value = s["throughput_tok_s"] / s0["throughput_tok_s"]
        round_ms = step_p50(s, "round", batch)
        step_ms = step_p50(s0, "decode", batch)
        cost_ratio = "-" if round_ms is None or step_ms is None else f"{round_ms / step_ms:.2f}"
        round_text = "-" if round_ms is None or step_ms is None else f"{round_ms:.1f} / {step_ms:.1f}"
        print(f"| {line} | {batch} | {k} | {s0['throughput_tok_s']:.1f} | {s['throughput_tok_s']:.1f} | **{ratio_value:.3f}** | "
              f"{s['tokens_per_row_round']:.2f} | {round_text} | {cost_ratio} | "
              f"{s0['tpot_ms'][50]:.1f} → {s['tpot_ms'][50]:.1f} | {s0['tpot_ms'][95]:.1f} → {s['tpot_ms'][95]:.1f} |")
    print("\n## Round anatomy (p50 ms per phase, speculative runs)\n")
    print("| line | B | k | catch_up | propose | verify | accept | copy | sum | draft-prefix survival by position |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (line, batch, k) in sorted(results):
        if k == 0:
            continue
        s, _ = results[(line, batch, k)]
        phases = s.get("phase_ms_p50", {})
        if not phases:
            continue
        total = sum(phases.values())
        rates = ", ".join(f"{rate:.2f}" for rate in s["accept_rate_by_position"])
        print(f"| {line} | {batch} | {k} | {phases.get('catch_up', 0):.2f} | {phases.get('propose', 0):.2f} | "
              f"{phases.get('verify', 0):.2f} | {phases.get('accept', 0):.2f} | {phases.get('commit_copy', 0):.2f} | {total:.1f} | {rates} |")
    print("\n## Output validity (greedy line)\n")
    print("| B | k | capped share | repeated-tail share |")
    print("|---:|---:|---:|---:|")
    for (line, batch, k) in sorted(results):
        if line != "greedy":
            continue
        s, _ = results[(line, batch, k)]
        print(f"| {batch} | {k} | {s['capped_share']:.2f} | {s['repeated_tail_share']:.2f} |")


if __name__ == "__main__":
    main()
