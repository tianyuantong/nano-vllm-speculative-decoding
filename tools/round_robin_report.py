"""Round-robin prefill against the default scheduler, a protocol's round_robin section, as markdown: for each pair of
lines and each C, round-robin / default of every measure per seed, then the median (min-max) over the seeds, and both
lines' own values. Effect sizes, no verdict.

A (C, seed) of the protocol's confirm section counts only when it stands whole: every line of confirm.versions, with
timing_noise's repeat 2 where the protocol asks for it, is there and none of them is invalid. report_v2's count_runs
decides what is invalid, a result file holding a NaN or an Infinity is not read at all, and a rerun directory listed last
replaces the group it repeats.

  python tools/round_robin_report.py --protocol benchmarks/protocol-v2.9.json --results DIR [DIR ...]
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itl_stats import itl  # noqa: E402
from report_v2 import count_runs  # noqa: E402
from summarize_bench import summarize_v2  # noqa: E402

RATIOS = [("q_steady", "Q_steady"), ("q_full", "Q_full"), ("itl_p99", "ITL p99"), ("itl_p50", "ITL p50"),
          ("ttft_p50", "TTFT p50"), ("ttft_p99", "TTFT p99")]
VALUES = [("itl_p99", "ITL p99, ms", "{:.1f}"), ("itl_p50", "ITL p50, ms", "{:.1f}"), ("ttft_p50", "TTFT p50, ms", "{:.0f}"),
          ("ttft_p99", "TTFT p99, ms", "{:.0f}"), ("prefill_gaps", "gaps with a prefill step", "{:.0%}")]


def _reject(text: str) -> float:
    raise ValueError(f"non-finite value: {text}")


def _finite_float(text: str) -> float:
    value = float(text)
    return value if math.isfinite(value) else _reject(text)


def load_finite_runs(directories: list[Path]) -> list[tuple[str, dict, dict]]:
    """report_v2's load_runs without the result files that hold a NaN or an Infinity, which invalid_if voids."""
    runs = []
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            try:
                result = json.loads(path.read_text(), parse_constant=_reject, parse_float=_finite_float)
            except ValueError:
                continue
            if result.get("schema") == "bench-v2":
                runs.append((str(path), result, summarize_v2(result)))
    return runs


def measures(result: dict, summary: dict) -> dict[str, float]:
    gaps = itl(result, result["config"]["concurrency"])
    return {"q_steady": summary["q_steady"], "q_full": summary["q_full"], "itl_p50": gaps["p50"], "itl_p99": gaps["p99"],
            "prefill_gaps": gaps["stalled_share"], "ttft_p50": summary["ttft_ms"]["p50"],
            "ttft_p99": summary["ttft_ms"]["p99"]}


def standing_groups(runs: list[tuple[str, dict, dict]], protocol: dict) -> dict[tuple[int, int], dict[str, tuple]]:
    """(C, seed) -> line -> (result, summary) of its repeat 1, over the confirm section's groups that stand whole."""
    invalid, _, _ = count_runs(runs, protocol)
    confirm, noise = protocol["confirm"], protocol["timing_noise"]
    latest = {}    # a later directory's run replaces an earlier one's, as in count_runs
    for name, result, summary in runs:
        config = result["config"]
        if config["panel"] == confirm["panel"]:
            latest[(config["concurrency"], config["seed"], config["line"], config["repeat"])] = (name, result, summary)
    groups = {}
    for concurrency in confirm["concurrency"]:
        for seed in protocol["seeds"]["confirm"]:
            repeats = noise["repeats"] if seed == noise["seed"] and concurrency in noise["concurrency"] else 1
            keys = [(concurrency, seed, line, repeat) for line in confirm["versions"] for repeat in range(1, repeats + 1)]
            if all(key in latest and not invalid[latest[key][0]] for key in keys):
                groups[(concurrency, seed)] = {line: latest[(concurrency, seed, line, 1)][1:] for line in confirm["versions"]}
    return groups


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
    groups = {key: {line: measures(result, summary) for line, (result, summary) in by_line.items()}
              for key, by_line in standing_groups(runs, protocol).items()}
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
    print(report(load_finite_runs(args.results), json.loads(args.protocol.read_text())))


if __name__ == "__main__":
    main()
