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
from summarize_bench import round_weighted_tokens, summarize_v2, tokens_per_round, verified_per_row  # noqa: E402

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


def range_of(ratios: list[float]) -> tuple[float, float, float, bool]:
    """Median, min and max, and whether the protocol reports only the range (it spans more than 0.05)."""
    low, high = min(ratios), max(ratios)
    return statistics.median(ratios), low, high, high - low > 0.05


def format_range(ratios: list[float]) -> str:
    """Median (min–max) to two decimals; only the range when it spans more than 0.05."""
    median, low, high, range_only = range_of(ratios)
    if range_only:
        return f"{low:.2f}–{high:.2f}"
    return f"{median:.2f} ({low:.2f}–{high:.2f})"


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


def tree_of(protocol: dict, line: str) -> str:
    """The source tree a line runs from; before v2.2 there is one."""
    return protocol["lines"][line].get("tree", "main")


def k0_line_of(protocol: dict, line: str) -> str:
    """The k0 line a line's R divides by: its own tree's (v2.2), else k0."""
    return protocol["lines"][line].get("k0_line", BASELINE)


def paired_gains(q: dict, panel: str, concurrency: int, line_a: str, line_b: str) -> list[float]:
    """Q_A / Q_B − 1 for every seed with both lines."""
    return [by_line[line_a][0] / by_line[line_b][0] - 1 for (p, c, _), by_line in q.items()
            if p == panel and c == concurrency and line_a in by_line and line_b in by_line]


def tokens_equal(result_a: dict, result_b: dict) -> bool:
    return [request["token_ids"] for request in result_a["requests"]] == \
           [request["token_ids"] for request in result_b["requests"]]


def _median_or_dash(values: list[float | None]) -> str:
    values = [value for value in values if value is not None]
    return f"{statistics.median(values):.2f}" if values else "-"


def count_runs(runs: list[tuple[str, dict, dict]], protocol: dict) -> tuple[dict, list, dict]:
    """The runs that count: each run's invalid reasons by name, the timing-noise pairs, and per (panel, C, seed) each
    valid repeat-1 line's (Q_steady, tokens per round, its per-request median, positions verified per row). A later
    run with the same
    (panel, C, line, seed, repeat) replaces an earlier one, so a rerun directory listed last replaces the runs it repeats.
    Every run of a (panel, C, seed), each line and repeat, must come from one directory with one protocol and panel,
    and the lines of each source tree from one commit; otherwise none of them counts, so a rerun has to repeat the
    whole (C, seed)."""
    invalid = {name: invalid_reasons(result, summary, protocol) for name, result, summary in runs}
    by_key = {}
    for name, result, summary in runs:
        config = result["config"]
        by_key[(config["panel"], config["concurrency"], config["line"], config["seed"], config["repeat"])] = (name, result, summary)

    # A result file carries no job id, so its directory stands for the job. A job with two trees has two commits.
    jobs = defaultdict(set)       # (panel, C, seed) -> (directory, protocol, panel) of each of its runs
    commits = defaultdict(set)    # (panel, C, seed, tree) -> the commits of its runs
    for (panel, concurrency, line, seed, _), (name, result, _) in by_key.items():
        manifest = result["manifest"]
        jobs[(panel, concurrency, seed)].add((str(Path(name).parent), result["protocol_sha256"], manifest["panel_sha256"]))
        commits[(panel, concurrency, seed, tree_of(protocol, line))].add(manifest["commit"])
    split = ({key for key, found in jobs.items() if len(found) > 1}
             | {key[:3] for key, found in commits.items() if len(found) > 1})

    pairs = []
    for (panel, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        if (panel, concurrency, seed) in split:
            invalid[name].append("lines of one (C, seed) split across jobs")
            continue
        twin = by_key.get((panel, concurrency, line, seed, 2))
        if repeat == 1 and twin is not None:
            pairs.append((summary["q_steady"], twin[2]["q_steady"]))
            if not tokens_equal(result, twin[1]):
                for run_name in (name, twin[0]):
                    invalid[run_name].append("timing-noise repeats differ in tokens")

    q = defaultdict(dict)    # (panel, C, seed) -> line -> (Q_steady, tokens per round, per-request median, verified)
    for (panel, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        if repeat == 1 and not invalid[name]:
            per_request = [value for value in tokens_per_round(result["requests"]) if value is not None]
            stats = result["speculative_stats"]
            q[(panel, concurrency, seed)][line] = (summary["q_steady"], round_weighted_tokens(stats) if stats else None,
                                                   statistics.median(per_request) if per_request else None,
                                                   verified_per_row(stats) if stats else None)
    return invalid, pairs, q


def report(runs: list[tuple[str, dict, dict]], protocol: dict) -> str:
    """runs: (name, result, summary) of every bench-v2 result file."""
    invalid, pairs, q = count_runs(runs, protocol)
    noise = timing_noise(pairs) if pairs else None
    panels = sorted({key[0] for key in q})
    concurrencies = sorted({key[1] for key in q})
    k0_lines = [line for line, spec in protocol["lines"].items() if spec["k"] == 0]
    lines = [line for line in protocol["lines"] if line not in k0_lines]

    out = ["## Timing noise", "",
           f"N_t = {noise:.4f} from {len(pairs)} repeat pairs" if noise is not None else "N_t unavailable: no repeat pairs",
           "", "## R = Q_steady(line) / Q_steady(its k0 line)", "",
           "tokens/round: sum of (accepted + 1) over all rounds ÷ rounds, as vLLM and DeepSpec count it. per-request: "
           "the median over requests of each request's mean, the column earlier reports showed. verified/row: positions "
           "verified per row and round, variable-length lines only. All three are medians over seeds.",
           "", "| panel | C | line | k0 | seeds | R | faster | tokens/round | per-request | verified/row |",
           "|---|---:|---|---|---:|---|---|---:|---:|---:|"]
    medians = defaultdict(dict)
    for panel in panels:
        for concurrency in concurrencies:
            for line in lines:
                k0 = k0_line_of(protocol, line)
                runs_by_seed = {seed: by_line for (p, c, seed), by_line in q.items()
                                if p == panel and c == concurrency and line in by_line and k0 in by_line}
                if not runs_by_seed:
                    continue
                ratios = [by_line[line][0] / by_line[k0][0] for by_line in runs_by_seed.values()]
                medians[(panel, line)][concurrency] = statistics.median(ratios)
                faster = ("yes" if verdict_faster(ratios, noise) else "no") if noise is not None else "-"
                tokens = [_median_or_dash([by_line[line][index] for by_line in runs_by_seed.values()])
                          for index in (1, 2, 3)]
                out.append(f"| {panel} | {concurrency} | {line} | {k0} | {len(ratios)} | {format_range(ratios)} | {faster} | "
                           f"{tokens[0]} | {tokens[1]} | {tokens[2]} |")

    out += ["", "## Crossover (smallest C with median R < 1)", ""]
    for (panel, line), by_concurrency in sorted(medians.items()):
        found = crossover(by_concurrency)
        out.append(f"- {panel} {line}: {found if found is not None else f'none up to C={max(by_concurrency)}'}")

    if len(k0_lines) > 1:
        out += ["", "## The trees' k0 lines: Q_A / Q_B − 1 paired by seed", "",
                "Only where every |gain| <= 2·N_t do the trees' R go in one table (protocol confirm.cross_tree); "
                "elsewhere each line's R stands against its own k0 line only.", "",
                "| panel | C | A | B | seeds | gain | one table |", "|---|---:|---|---|---:|---|---|"]
        for panel in panels:
            for concurrency in concurrencies:
                for line_a, line_b in combinations(k0_lines, 2):
                    gains = paired_gains(q, panel, concurrency, line_a, line_b)
                    if gains:
                        together = (("yes" if all(abs(gain) <= 2 * noise for gain in gains) else "no")
                                    if noise is not None else "-")
                        span = f"{min(gains):+.3f}..{max(gains):+.3f}"
                        out.append(f"| {panel} | {concurrency} | {line_a} | {line_b} | {len(gains)} | {span} | {together} |")

    out += ["", "## Line against line: Q_A / Q_B − 1 paired by seed, within a tree", "",
            "| panel | C | A | B | seeds | gain | verdict |", "|---|---:|---|---|---:|---|---:|"]
    for panel in panels:
        for concurrency in concurrencies:
            for line_a, line_b in combinations(lines, 2):
                if tree_of(protocol, line_a) != tree_of(protocol, line_b):
                    continue                     # across trees only R compares, where the k0 lines agree
                gains = paired_gains(q, panel, concurrency, line_a, line_b)
                if gains:
                    verdict = verdict_a_beats_b(gains, noise) if noise is not None else "-"
                    span = f"{min(gains):+.3f}..{max(gains):+.3f}"
                    out.append(f"| {panel} | {concurrency} | {line_a} | {line_b} | {len(gains)} | {span} | {verdict} |")

    out += ["", "## Invalid runs", ""]
    out += [f"- {name}: {', '.join(reasons)}" for name, reasons in sorted(invalid.items()) if reasons] or ["- none"]
    return "\n".join(out)


def load_runs(directories: list[Path]) -> list[tuple[str, dict, dict]]:
    """(path, result, summary) of every bench-v2 result file, in the order of the directories."""
    runs = []
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            result = json.loads(path.read_text())
            if result.get("schema") == "bench-v2":
                runs.append((str(path), result, summarize_v2(result)))
    return runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="directories of bench-v2 result files")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    print(report(load_runs(args.results), protocol))


if __name__ == "__main__":
    main()
