"""Line B's paired gain over line A, Q_steady(B) / Q_steady(A), split per (panel, C, seed) into four factors whose
product is the gain: tokens per round (B / A), the median full-batch round (A / B), the steady window's share outside
prefill steps (B / A), and the rest. Beside them, the median propose and verify phases, B minus A. Medians over seeds;
runs count as report_v2.py counts them.

  python tools/gain_factors.py --protocol benchmarks/protocol-v2.1.json --lines dflash_k7 dspark_k7 \
      --results DIR [DIR ...]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import count_runs, load_runs  # noqa: E402
from stall_stats import stall  # noqa: E402
from summarize_bench import round_weighted_tokens  # noqa: E402

PHASES = ("propose", "verify")


def measures(result: dict, concurrency: int) -> dict:
    """One run's steady-state throughput, tokens per round, median full-batch round, the window's share outside
    prefill steps, and its median propose and verify phases."""
    window = stall(result, concurrency)
    by_phase = defaultdict(list)
    for phase in result["phases"]:
        by_phase[phase["phase"]].append(phase["duration_ms"])
    return {"q_steady": window["q_steady"], "tokens": round_weighted_tokens(result["speculative_stats"]),
            "round_ms": window["round_ms"], "outside_prefill": 1 - window["loss_high"],
            **{phase: statistics.median(by_phase[phase]) for phase in PHASES}}


def factors(runs: list[tuple[str, dict, dict]], protocol: dict, line_a: str,
            line_b: str) -> dict[tuple[str, int], dict[str, list[float]]]:
    """(panel, C) -> factor -> one value per seed at which both lines have a valid run. A later run with the same key
    replaces an earlier one, as in report_v2.count_runs."""
    invalid, _, _ = count_runs(runs, protocol)
    latest = {}
    for name, result, _ in runs:
        config = result["config"]
        if config["repeat"] == 1 and config["line"] in (line_a, line_b):
            latest[(config["panel"], config["concurrency"], config["seed"], config["line"])] = (name, result)
    table = defaultdict(lambda: defaultdict(list))
    for (panel, concurrency, seed, line), (name, result) in sorted(latest.items()):
        other = latest.get((panel, concurrency, seed, line_a))
        if line != line_b or other is None or invalid[name] or invalid[other[0]]:
            continue
        b, a = measures(result, concurrency), measures(other[1], concurrency)
        gain = b["q_steady"] / a["q_steady"]
        tokens = b["tokens"] / a["tokens"]
        round_ratio = a["round_ms"] / b["round_ms"]
        outside = b["outside_prefill"] / a["outside_prefill"]
        row = table[(panel, concurrency)]
        row["gain"].append(gain)
        row["tokens"].append(tokens)
        row["round"].append(round_ratio)
        row["outside"].append(outside)
        row["rest"].append(gain / (tokens * round_ratio * outside))
        for phase in PHASES:
            row[phase].append(b[phase] - a[phase])
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lines", nargs=2, required=True, metavar=("A", "B"), help="the gain of line B over line A")
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="directories of bench-v2 result files; a rerun directory last replaces the runs it repeats")
    args = parser.parse_args(argv)
    table = factors(load_runs(args.results), json.loads(args.protocol.read_text()), *args.lines)
    print("| panel | C | seeds | gain | tokens per round | round | outside prefill | rest | Δ propose ms | Δ verify ms |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for (panel, concurrency), row in sorted(table.items()):
        med = {key: statistics.median(values) for key, values in row.items()}
        print(f"| {panel} | {concurrency} | {len(row['gain'])} | {med['gain'] - 1:+.1%} "
              f"({min(row['gain']) - 1:+.1%}..{max(row['gain']) - 1:+.1%}) | {med['tokens'] - 1:+.1%} | "
              f"{med['round'] - 1:+.1%} | {med['outside'] - 1:+.1%} | {med['rest'] - 1:+.1%} | "
              f"{med['propose']:+.2f} | {med['verify']:+.2f} |")


if __name__ == "__main__":
    main()
