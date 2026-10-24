"""Inter-token latency of bench-v2 closed-loop results, split by whether a prefill step fell inside the interval
(decision record 2026-10-23, D7).

ITL here is the time between a request's consecutive deliveries, both inside the steady window, measured from the
end of one delivering step to the end of the next. nano-vllm's prefill steps carry no decodes, so an interval that
contains one is a stall. Per run: ITL p50 and p99, p99 over the intervals without a prefill step, the share of
intervals with one, and their p50. Medians over the seeds.

  python tools/itl_stats.py DIR      # a directory of <line>-C<c>-s<seed>-r<repeat>.json results
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_bench import _end_ms, percentile, steady_window  # noqa: E402

NAME = re.compile(r"(\w+)-C(\d+)-s(\d+)-r(\d+)\.json")


def itl(result: dict, concurrency: int) -> dict:
    steps = result["steps"]
    first, last = steady_window(steps, concurrency)
    end = {step["step"]: _end_ms(step) for step in steps}
    prefill_steps = [step["step"] for step in steps if step["kind"] == "prefill"]
    clean, stalled = [], []
    for request in result["requests"]:
        delivered = [step for step, _ in request["deliveries"]]
        for a, b in zip(delivered, delivered[1:]):
            if first <= a and b <= last:
                (stalled if any(a < p <= b for p in prefill_steps) else clean).append(end[b] - end[a])
    everything = clean + stalled
    return {"p50": percentile(everything, 50), "p99": percentile(everything, 99),
            "p99_clean": percentile(clean, 99) if clean else 0.0, "stalled_share": len(stalled) / len(everything),
            "stalled_p50": percentile(stalled, 50) if stalled else 0.0}


def main() -> None:
    runs = defaultdict(list)    # (line, C) -> itl() of each seed's first repeat
    for path in sorted(Path(sys.argv[1]).glob("*.json")):
        match = NAME.fullmatch(path.name)
        if match and match.group(4) == "1":
            line, concurrency = match.group(1), int(match.group(2))
            runs[(line, concurrency)].append(itl(json.loads(path.read_text()), concurrency))
    print("| line | C | ITL p50 ms | ITL p99 ms | p99 without prefill intervals | intervals with a prefill | their p50 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for (line, concurrency), rows in sorted(runs.items()):
        def median(key: str) -> float:
            return statistics.median(row[key] for row in rows)
        print(f"| {line} | {concurrency} | {median('p50'):.1f} | {median('p99'):.1f} | {median('p99_clean'):.1f} "
              f"| {100 * median('stalled_share'):.1f}% | {median('stalled_p50'):.1f} |")


if __name__ == "__main__":
    main()
