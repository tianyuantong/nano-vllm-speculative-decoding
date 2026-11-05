"""M5 step 4: round-robin prefill against the same line without it (protocol v2.4 round_robin), on the protocol's
pre-registered verdict.

  python tools/round_robin_report.py --protocol benchmarks/protocol-v2.4.json --results DIR [--open DIR]

Throughput is Q_full over the same requests: the steady window would leave out the phase where requests wait for
prefill, which is where the policy looks worst. ITL is every consecutive pair of a request's deliveries from the
warm-up boundary on, the step of the last first delivery among the run's first C requests. --open compares the arms on
the same arrival plans: a tier the base line passes at a rate must still pass.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import count_runs, load_runs  # noqa: E402
from slo_report import load_runs as load_open_runs  # noqa: E402
from slo_report import meets, summarize_run  # noqa: E402
from summarize_bench import itl_ms, percentile, q_full, ttft_ms  # noqa: E402


def warmup_boundary(result: dict) -> int:
    concurrency = result["config"]["concurrency"]
    return max(request["deliveries"][0][0] for request in result["requests"][:concurrency])


def measures(result: dict) -> dict[str, float]:
    steps, requests = result["steps"], result["requests"]
    gaps = itl_ms(steps, requests, (warmup_boundary(result), len(steps) - 1))
    ttft = ttft_ms(steps, requests)
    return {"q_full": q_full(steps, requests), "itl_p99": percentile(gaps, 99), "ttft_p50": percentile(ttft, 50),
            "ttft_p99": percentile(ttft, 99)}


def seed_ratios(runs: dict, base: str, arm: str, concurrency: int) -> dict[int, dict[str, float]]:
    """seed -> measure -> arm / base."""
    ratios = {}
    for (line, c, seed), row in runs.items():
        if line == base and c == concurrency and (arm, c, seed) in runs:
            ratios[seed] = {key: runs[(arm, c, seed)][key] / row[key] for key in row}
    return ratios


def closed_loop_failures(ratios: dict[int, dict[str, float]], concurrency: int, verdict: dict) -> list[str]:
    """The verdict's conditions that one line's ratios at one C miss."""
    failures = []
    limit = verdict["itl_p99_ratio_max"][str(concurrency)]
    failures += [f"seed {seed}: ITL p99 ratio {row['itl_p99']:.3f} > {limit}" for seed, row in ratios.items() if row["itl_p99"] > limit]
    failures += [f"seed {seed}: TTFT p50 ratio {row['ttft_p50']:.3f} > {verdict['ttft_p50_ratio_max']}"
                 for seed, row in ratios.items() if row["ttft_p50"] > verdict["ttft_p50_ratio_max"]]
    failures += [f"seed {seed}: TTFT p99 ratio {row['ttft_p99']:.3f} > {verdict['ttft_p99_ratio_max']}"
                 for seed, row in ratios.items() if row["ttft_p99"] > verdict["ttft_p99_ratio_max"]]
    median = statistics.median(row["q_full"] for row in ratios.values())
    if median < verdict["q_full_ratio_median_min"]:
        failures.append(f"median Q_full ratio {median:.3f} < {verdict['q_full_ratio_median_min']}")
    return failures


def slo_failures(results: list[dict], pairs: dict[str, str], lines: list[str], spec: dict) -> list[str]:
    """Tiers a base line passes at a rate and its arm does not, on the same plan."""
    summaries = {(result["config"]["line"], result["config"]["rate"], result["config"]["seed"]):
                 summarize_run(result, spec["warmup_fraction"]) for result in results}
    failures = []
    for (line, rate, seed), summary in sorted(summaries.items()):
        if line not in lines or (pairs[line], rate, seed) not in summaries:
            continue
        for name, tier in spec["slo"]["tiers"].items():
            if meets(summary, tier) and not meets(summaries[(pairs[line], rate, seed)], tier):
                failures.append(f"{line} at {rate:g}/s: {name} passes without round robin and fails with it")
    return failures


def report(runs: list[tuple[str, dict, dict]], open_results: list[dict], protocol: dict) -> str:
    spec = protocol["round_robin"]
    invalid, _, _ = count_runs(runs, protocol)
    bad = [f"- {Path(name).name}: {', '.join(reasons)}" for name, reasons in sorted(invalid.items()) if reasons]
    if bad:
        return "\n".join(["## Invalid runs: rerun their whole (C, seed) before reading a verdict", "", *bad])
    measured = {(result["config"]["line"], result["config"]["concurrency"], result["config"]["seed"]): measures(result)
                for _, result, _ in runs}
    out = ["## rr / base, per seed", "", "| line | C | seed | Q_full | ITL p99 | TTFT p50 | TTFT p99 |", "|---|---:|---:|---:|---:|---:|---:|"]
    failures = defaultdict(list)
    for base, arm in spec["pairs"].items():
        for concurrency in spec["concurrency"]:
            ratios = seed_ratios(measured, base, arm, concurrency)
            for seed, row in sorted(ratios.items()):
                out.append(f"| {base} | {concurrency} | {seed} | {row['q_full']:.3f} | {row['itl_p99']:.3f} "
                           f"| {row['ttft_p50']:.3f} | {row['ttft_p99']:.3f} |")
            if base in spec["verdict_lines"]:
                failures[base] += [f"C={concurrency} {reason}" for reason in closed_loop_failures(ratios, concurrency, spec["verdict"])]
    missing = [] if open_results else ["no open-loop results: the SLO condition is not checked"]
    for line in slo_failures(open_results, spec["pairs"], spec["verdict_lines"], protocol["open_loop"]):
        failures[line.split()[0]].append(line)
    out += ["", "## Verdict", ""]
    for line in spec["verdict_lines"]:
        out.append(f"- {line}: " + ("; ".join(failures[line]) if failures[line] else "every condition holds"))
    passed = not missing and not any(failures.values())
    out += [f"- {reason}" for reason in missing]
    out += ["", f"round robin {'passes' if passed else 'does not pass'} step 4"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="directories of closed-loop results")
    parser.add_argument("--open", type=Path, nargs="*", default=[], help="directories of open-loop results")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    print(report(load_runs(args.results), load_open_runs(args.open), protocol))


if __name__ == "__main__":
    main()
