"""SLO capacity over open-loop results (protocol v2.4 open_loop): per line and arrival rate the latencies a request saw
from its planned arrival to the host's delivery, whether they meet each SLO tier, and the rate interval each line's
capacity lies in. An unfinished request counts as an infinite latency, so a run that does not drain fails its tiers.

  python tools/slo_report.py --protocol benchmarks/protocol-v2.4.json --results DIR [DIR ...]

The MLPerf rule: at the given percentile, TTFT and TPOT each under their limit. TPOT is per request: (end to end -
TTFT) / (tokens - 1), so a single request's stall is spread over its tokens; it is not the ITL.
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_bench import percentile  # noqa: E402

SCHEMA = "open-loop-v1"
SLOW_FACTOR = 3    # a step this many times the median of its kind and batch size is reported as slow


def request_latencies(request: dict) -> dict[str, float | None]:
    """One request's TTFT and end to end in ms from its planned arrival, infinite when unfinished; TPOT likewise, None
    for a one-token output that has no gap to divide."""
    arrival = request["arrival_seconds"]
    deliveries = request["deliveries"]
    ttft = (deliveries[0][2] - arrival) * 1000 if deliveries else math.inf
    if request["finish_step"] < 0:
        return {"ttft_ms": ttft, "e2e_ms": math.inf, "tpot_ms": math.inf}
    e2e = (deliveries[-1][2] - arrival) * 1000
    tokens = sum(n for _, n, _ in deliveries)
    return {"ttft_ms": ttft, "e2e_ms": e2e, "tpot_ms": (e2e - ttft) / (tokens - 1) if tokens > 1 else None}


def itl_ms(requests: list[dict]) -> list[float]:
    """Gaps between consecutive deliveries of each request."""
    return [(b[2] - a[2]) * 1000 for request in requests
            for a, b in zip(request["deliveries"], request["deliveries"][1:])]


def queue_lengths(requests: list[dict]) -> list[int]:
    """At each arrival, how many earlier requests had arrived and not had their first token."""
    first_tokens = sorted(request["deliveries"][0][2] if request["deliveries"] else math.inf for request in requests)
    served, lengths = 0, []
    for index, request in enumerate(requests):
        while served < len(first_tokens) and first_tokens[served] <= request["arrival_seconds"]:
            served += 1
        lengths.append(index - served)
    return lengths


def slow_steps(steps: list[dict]) -> int:
    """Decoding steps over SLOW_FACTOR x the median of their kind and batch size: reported, a tail stays in the tail."""
    durations = defaultdict(list)
    for step in steps:
        if step["kind"] != "prefill":
            durations[(step["kind"], step["batch_size"])].append(step["duration_ms"])
    return sum(duration > SLOW_FACTOR * statistics.median(values) for values in durations.values() for duration in values)


def summarize_run(result: dict, warmup_fraction: float) -> dict:
    """Metrics over the requests past the warm-up boundary, which depends only on the plan: every one of them counts."""
    requests = result["requests"]
    counted = requests[int(len(requests) * warmup_fraction):]
    latencies = [request_latencies(request) for request in counted]
    gaps = itl_ms(counted)
    tpot = [row["tpot_ms"] for row in latencies if row["tpot_ms"] is not None]
    finishes = [request["deliveries"][-1][2] for request in counted if request["finish_step"] >= 0]
    tokens = sum(n for request in counted for _, n, _ in request["deliveries"])
    span = max(finishes) - counted[0]["arrival_seconds"] if finishes else math.inf

    def at(values: list[float], q: float) -> float:
        return percentile(values, q)

    return {"ttft_p50": at([row["ttft_ms"] for row in latencies], 50), "ttft_p99": at([row["ttft_ms"] for row in latencies], 99),
            "e2e_p50": at([row["e2e_ms"] for row in latencies], 50), "e2e_p99": at([row["e2e_ms"] for row in latencies], 99),
            "tpot_p99": at(tpot, 99), "itl_p50": at(gaps, 50), "itl_p99": at(gaps, 99), "tokens_per_second": tokens / span,
            "unfinished": len(counted) - len(finishes), "queue_p50": at(queue_lengths(requests), 50),
            "queue_max": max(queue_lengths(requests)), "preemptions": result["preemptions"],
            "full_collections": result["full_collections"], "slow_steps": slow_steps(result["steps"])}


def meets(summary: dict, tier: dict) -> bool:
    return summary["ttft_p99"] <= tier["ttft_ms"] and summary["tpot_p99"] <= tier["tpot_ms"]


def capacity(passed_by_rate: dict[float, bool]) -> tuple[float | None, float | None]:
    """(last rate that passes before the first that fails, the first that fails); None where the grid has no such rate."""
    last_passed = first_failed = None
    for rate in sorted(passed_by_rate):
        if not passed_by_rate[rate]:
            first_failed = rate
            break
        last_passed = rate
    return last_passed, first_failed


def format_capacity(interval: tuple[float | None, float | None]) -> str:
    low, high = interval
    if high is None:
        return f"at least {low:g}"
    return f"below {high:g}" if low is None else f"{low:g} to {high:g}"


def load_runs(directories: list[Path]) -> list[dict]:
    results = [json.loads(path.read_text()) for directory in directories for path in sorted(directory.glob("*.json"))]
    return [result for result in results if result.get("schema") == SCHEMA]


def report(results: list[dict], protocol: dict) -> str:
    spec = protocol["open_loop"]
    runs = defaultdict(dict)    # (panel, line, rate) -> seed -> summary
    for result in results:
        config = result["config"]
        runs[(config["panel"], config["line"], config["rate"])][config["seed"]] = summarize_run(result, spec["warmup_fraction"])
    tiers = spec["slo"]["tiers"]
    out = ["## Open-loop latency", "", "Medians over the seeds, ms, from the planned arrival to the host's delivery.", "",
           "| panel | line | rate | seeds | tok/s | TTFT p50 / p99 | TPOT p99 | ITL p50 / p99 | e2e p50 / p99 | queue p50 / max "
           "| unfinished | preemptions | full GCs | slow steps | " + " | ".join(tiers) + " |",
           "|---|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|" + "---|" * len(tiers)]
    passed = defaultdict(dict)    # (panel, line, tier) -> rate -> every seed passes
    for (panel, line, rate), by_seed in sorted(runs.items()):
        rows = list(by_seed.values())

        def median(key: str) -> float:
            return statistics.median(row[key] for row in rows)

        cells = []
        for name, tier in tiers.items():
            passed[(panel, line, name)][rate] = all(meets(row, tier) for row in rows)
            cells.append("pass" if passed[(panel, line, name)][rate] else "fail")
        out.append(f"| {panel} | {line} | {rate:g} | {len(rows)} | {median('tokens_per_second'):.0f} "
                   f"| {median('ttft_p50'):.0f} / {median('ttft_p99'):.0f} | {median('tpot_p99'):.1f} "
                   f"| {median('itl_p50'):.1f} / {median('itl_p99'):.1f} | {median('e2e_p50'):.0f} / {median('e2e_p99'):.0f} "
                   f"| {median('queue_p50'):.0f} / {max(row['queue_max'] for row in rows):.0f} "
                   f"| {max(row['unfinished'] for row in rows)} | {max(row['preemptions'] for row in rows)} "
                   f"| {max(row['full_collections'] for row in rows)} | {max(row['slow_steps'] for row in rows)} | "
                   + " | ".join(cells) + " |")
    out += ["", "## Capacity in requests per second", "",
            "The last rate that passes on every seed before the first that fails.", "",
            "| panel | line | tier | capacity |", "|---|---|---|---|"]
    out += [f"| {panel} | {line} | {name} | {format_capacity(capacity(by_rate))} |"
            for (panel, line, name), by_rate in sorted(passed.items())]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="directories of open-loop result files")
    args = parser.parse_args(argv)
    print(report(load_runs(args.results), json.loads(args.protocol.read_text())))


if __name__ == "__main__":
    main()
