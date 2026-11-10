"""Round-robin prefill against the default scheduler, a protocol's round_robin section, as markdown: for each pair of
lines and each C, round-robin / default of every measure per seed, then the median (min-max) over the seeds, and both
lines' own values. Effect sizes, no verdict.

A (C, seed) counts only when every line of the protocol's confirm.versions has a valid repeat 1 there. report_v2's
count_runs decides what is valid, and a rerun directory listed last replaces the group it repeats.

  python tools/round_robin_report.py --protocol benchmarks/protocol-v2.9.json --results DIR [DIR ...]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itl_stats import itl  # noqa: E402
from report_v2 import count_runs, load_runs  # noqa: E402

RATIOS = [("q_steady", "Q_steady"), ("q_full", "Q_full"), ("itl_p99", "ITL p99"), ("itl_p50", "ITL p50"),
          ("ttft_p50", "TTFT p50"), ("ttft_p99", "TTFT p99")]
VALUES = [("itl_p99", "ITL p99, ms", "{:.1f}"), ("itl_p50", "ITL p50, ms", "{:.1f}"), ("ttft_p50", "TTFT p50, ms", "{:.0f}"),
          ("prefill_gaps", "gaps with a prefill step", "{:.0%}")]


def measures(result: dict, summary: dict) -> dict[str, float]:
    gaps = itl(result, result["config"]["concurrency"])
    return {"q_steady": summary["q_steady"], "q_full": summary["q_full"], "itl_p50": gaps["p50"], "itl_p99": gaps["p99"],
            "prefill_gaps": gaps["stalled_share"], "ttft_p50": summary["ttft_ms"]["p50"],
            "ttft_p99": summary["ttft_ms"]["p99"]}


def complete_groups(runs: list[tuple[str, dict, dict]], protocol: dict) -> dict[tuple[int, int], dict[str, dict]]:
    """(C, seed) -> line -> measures, over the confirmation panel's groups in which every line stands."""
    invalid, _, _ = count_runs(runs, protocol)
    latest = {}
    for name, result, summary in runs:
        config = result["config"]
        if config["panel"] == protocol["confirm"]["panel"] and config["repeat"] == 1:
            latest[(config["concurrency"], config["seed"], config["line"])] = (name, result, summary)
    groups = defaultdict(dict)
    for (concurrency, seed, line), (name, result, summary) in latest.items():
        if not invalid[name]:
            groups[(concurrency, seed)][line] = measures(result, summary)
    lines = protocol["confirm"]["versions"]
    return {key: by_line for key, by_line in groups.items() if all(line in by_line for line in lines)}


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})"


def table(groups: dict[tuple[int, int], dict[str, dict]], pairs: dict[str, str], concurrencies: list[int]) -> list[str]:
    out = ["| pair | C | seeds | " + " | ".join(label for _, label in RATIOS) + " | " +
           " | ".join(label for _, label, _ in VALUES) + " |",
           "|---|---:|---:|" + "---|" * (len(RATIOS) + len(VALUES))]
    for default, round_robin in pairs.items():
        for concurrency in concurrencies:
            seeds = sorted(seed for (c, seed) in groups if c == concurrency)
            if not seeds:
                continue
            both = [(groups[(concurrency, seed)][default], groups[(concurrency, seed)][round_robin]) for seed in seeds]
            ratios = [spread([rr[key] / base[key] for base, rr in both]) for key, _ in RATIOS]
            values = [fmt.format(statistics.median(base[key] for base, _ in both)) + " → " +
                      fmt.format(statistics.median(rr[key] for _, rr in both)) for key, _, fmt in VALUES]
            out.append(f"| {round_robin} / {default} | {concurrency} | {len(seeds)} | " + " | ".join(ratios + values) + " |")
    return out


def report(runs: list[tuple[str, dict, dict]], protocol: dict) -> str:
    groups = complete_groups(runs, protocol)
    expected = len(protocol["confirm"]["concurrency"]) * len(protocol["seeds"]["confirm"])
    out = ["## Round-robin prefill against the default scheduler", "",
           f"{len(groups)} of {expected} (C, seed) groups stand. Ratios are round-robin / default per seed: median "
           "(min–max). Values are medians over the seeds, default → round-robin; ITL over the steady window.", ""]
    return "\n".join(out + table(groups, protocol["round_robin"]["pairs"], protocol["confirm"]["concurrency"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="directories of bench-v2 result files")
    args = parser.parse_args(argv)
    print(report(load_runs(args.results), json.loads(args.protocol.read_text())))


if __name__ == "__main__":
    main()
