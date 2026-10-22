"""Pre-registered verdicts over bench-v2 results (bench-v2 design §2.4–§2.6): timing noise, R against k0,
line against line, crossover and invalid runs, as markdown.

  python tools/report_v2.py --protocol benchmarks/protocol-v2.json --results DIR [DIR ...]
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_bench import summarize_v2, tokens_per_round  # noqa: E402

BASELINE = "k0"


def timing_noise(pairs: list[tuple[float, float]]) -> float:
    """N_t: the largest |a/b − 1| over runs repeated with the same seed."""
    return max(abs(a / b - 1) for a, b in pairs)


def verdict_faster(ratios: list[float], noise: float) -> bool:
    """"Faster at this C": R above 1 + 2·N_t for every seed."""
    return all(ratio > 1 + 2 * noise for ratio in ratios)


def verdict_a_beats_b(gains: list[float], noise: float) -> int:
    """+1 or −1 when every paired gain Q_A/Q_B − 1 has that sign and |median| > 2·N_t; 0 otherwise."""
    for sign in (1, -1):
        if all(sign * gain > 0 for gain in gains) and abs(statistics.median(gains)) > 2 * noise:
            return sign
    return 0


def format_range(ratios: list[float]) -> str:
    """Median (min–max) to two decimals; only the range when it spans more than 0.05."""
    low, high = min(ratios), max(ratios)
    if high - low > 0.05:
        return f"{low:.2f}–{high:.2f}"
    return f"{statistics.median(ratios):.2f} ({low:.2f}–{high:.2f})"


def crossover(median_by_concurrency: dict[int, float]) -> int | None:
    """Smallest C whose median R is below 1."""
    return min((concurrency for concurrency, ratio in median_by_concurrency.items() if ratio < 1), default=None)


def invalid_reasons(result: dict, summary: dict, protocol: dict) -> list[str]:
    """The protocol's invalid_if conditions that one run shows on its own."""
    reasons = []
    if result["gpu_state_start"]["name"] != protocol["gpu"]["name"]:
        reasons.append("gpu name mismatch")
    if summary["excess_prefill_tokens"] > 0:
        reasons.append("preemption > 0")
    if not all(math.isfinite(summary[key]) for key in ("q_steady", "q_full")):
        reasons.append("non-finite value or assertion failure")
    durations = [step["duration_ms"] for step in result["steps"] if step["kind"] != "prefill"]
    if max(durations) > 3 * statistics.median(durations):
        reasons.append("decode step or round > 3 x median")
    return reasons


def tokens_equal(result_a: dict, result_b: dict) -> bool:
    return [request["token_ids"] for request in result_a["requests"]] == \
           [request["token_ids"] for request in result_b["requests"]]


def report(runs: list[tuple[str, dict, dict]], protocol: dict) -> str:
    """runs: (name, result, summary) of every bench-v2 result file."""
    invalid = {name: invalid_reasons(result, summary, protocol) for name, result, summary in runs}
    by_key = {}
    for name, result, summary in runs:
        config = result["config"]
        by_key[(config["panel"], config["concurrency"], config["line"], config["seed"], config["repeat"])] = (name, result, summary)

    pairs = []
    for (panel, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        twin = by_key.get((panel, concurrency, line, seed, 2))
        if repeat == 1 and twin is not None:
            pairs.append((summary["q_steady"], twin[2]["q_steady"]))
            if not tokens_equal(result, twin[1]):
                for run_name in (name, twin[0]):
                    invalid[run_name].append("timing-noise repeats differ in tokens")
    noise = timing_noise(pairs) if pairs else None

    q = defaultdict(dict)    # (panel, C, seed) -> line -> (Q_steady, median tokens per round) of valid repeat-1 runs
    for (panel, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        if repeat == 1 and not invalid[name]:
            per_round = [value for value in tokens_per_round(result["requests"]) if value is not None]
            q[(panel, concurrency, seed)][line] = (summary["q_steady"], statistics.median(per_round) if per_round else None)
    panels = sorted({key[0] for key in q})
    concurrencies = sorted({key[1] for key in q})
    lines = [line for line in protocol["lines"] if line != BASELINE]

    out = ["## Timing noise", "",
           f"N_t = {noise:.4f} from {len(pairs)} repeat pairs" if noise is not None else "N_t unavailable: no repeat pairs",
           "", "## R = Q_steady(line) / Q_steady(k0)", "",
           "| panel | C | line | seeds | R | faster | tokens/round |", "|---|---:|---|---:|---|---|---:|"]
    medians = defaultdict(dict)
    for panel in panels:
        for concurrency in concurrencies:
            for line in lines:
                runs_by_seed = {seed: by_line for (p, c, seed), by_line in q.items()
                                if p == panel and c == concurrency and line in by_line and BASELINE in by_line}
                if not runs_by_seed:
                    continue
                ratios = [by_line[line][0] / by_line[BASELINE][0] for by_line in runs_by_seed.values()]
                per_round = [by_line[line][1] for by_line in runs_by_seed.values() if by_line[line][1] is not None]
                medians[(panel, line)][concurrency] = statistics.median(ratios)
                faster = ("yes" if verdict_faster(ratios, noise) else "no") if noise is not None else "-"
                tokens = f"{statistics.median(per_round):.2f}" if per_round else "-"
                out.append(f"| {panel} | {concurrency} | {line} | {len(ratios)} | {format_range(ratios)} | {faster} | {tokens} |")

    out += ["", "## Crossover (smallest C with median R < 1)", ""]
    for (panel, line), by_concurrency in sorted(medians.items()):
        found = crossover(by_concurrency)
        out.append(f"- {panel} {line}: {found if found is not None else f'none up to C={max(by_concurrency)}'}")

    out += ["", "## Line against line: Q_A / Q_B − 1 paired by seed", "",
            "| panel | C | A | B | seeds | gain | verdict |", "|---|---:|---|---|---:|---|---:|"]
    for panel in panels:
        for concurrency in concurrencies:
            for line_a, line_b in combinations(lines, 2):
                gains = [by_line[line_a][0] / by_line[line_b][0] - 1 for (p, c, _), by_line in q.items()
                         if p == panel and c == concurrency and line_a in by_line and line_b in by_line]
                if gains:
                    verdict = verdict_a_beats_b(gains, noise) if noise is not None else "-"
                    span = f"{min(gains):+.3f}..{max(gains):+.3f}"
                    out.append(f"| {panel} | {concurrency} | {line_a} | {line_b} | {len(gains)} | {span} | {verdict} |")

    out += ["", "## Invalid runs", ""]
    out += [f"- {name}: {', '.join(reasons)}" for name, reasons in sorted(invalid.items()) if reasons] or ["- none"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="directories of bench-v2 result files")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    runs = []
    for directory in args.results:
        for path in sorted(directory.glob("*.json")):
            result = json.loads(path.read_text())
            if result.get("schema") == "bench-v2":
                runs.append((str(path), result, summarize_v2(result)))
    print(report(runs, protocol))


if __name__ == "__main__":
    main()
