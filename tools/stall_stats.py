"""Prefill stalls in bench-v2 closed-loop results (decision record 2026-10-23, D7).

nano-vllm never decodes in a prefill step, so each request that arrives stops the whole batch for one prefill step.
Per run, inside the steady window (the one Q_steady uses): the number of prefill steps, their median length, batch
and tokens, the median full-batch round, and two bounds on the throughput lost to them: lower, each prefill step
costs one round; upper, all prefill time is lost. Then, per C, R as measured against R with the window's prefill time
removed from both lines. Medians over the seeds.

  python tools/stall_stats.py DIR      # a directory of <line>-C<c>-s<seed>-r<repeat>.json results
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_bench import _end_ms, steady_window  # noqa: E402

NAME = re.compile(r"(\w+)-C(\d+)-s(\d+)-r(\d+)\.json")
BASELINE = "k0"


def stall(result: dict, concurrency: int) -> dict:
    steps = result["steps"]
    first, last = steady_window(steps, concurrency)
    by_step = {step["step"]: step for step in steps}
    window_ms = _end_ms(by_step[last]) - by_step[first]["start_ms"]
    prefills = [step for step in steps if step["kind"] == "prefill" and first <= step["step"] <= last]
    prefill_ms = sum(step["duration_ms"] for step in prefills)
    round_ms = statistics.median(step["duration_ms"] for step in steps
                                 if step["kind"] != "prefill" and step["batch_size"] == concurrency)
    tokens = sum(n for request in result["requests"] for step, n in request["deliveries"] if first <= step <= last)

    def median_of(key: str) -> float:
        return statistics.median(step[key] for step in prefills) if prefills else 0.0

    return {"window_s": window_ms / 1000, "prefills": len(prefills), "prefill_ms": median_of("duration_ms"),
            "prefill_requests": median_of("batch_size"), "prefill_tokens": median_of("tokens_out"), "round_ms": round_ms,
            "loss_low": len(prefills) * round_ms / window_ms, "loss_high": prefill_ms / window_ms,
            "q_steady": tokens / window_ms * 1000, "q_decode_only": tokens / (window_ms - prefill_ms) * 1000}


def main() -> None:
    runs = defaultdict(list)    # (line, C) -> stall() of each seed's first repeat
    for path in sorted(Path(sys.argv[1]).glob("*.json")):
        match = NAME.fullmatch(path.name)
        if match and match.group(4) == "1":
            line, concurrency = match.group(1), int(match.group(2))
            runs[(line, concurrency)].append(stall(json.loads(path.read_text()), concurrency))

    def median(key: str, rows: list[dict]) -> float:
        return statistics.median(row[key] for row in rows)

    print("| line | C | window s | prefill steps | prefill ms | requests per prefill | tokens per prefill | round ms "
          "| loss low | loss high | Q_steady | Q without prefill time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (line, concurrency), rows in sorted(runs.items()):
        print(f"| {line} | {concurrency} | {median('window_s', rows):.1f} | {median('prefills', rows):.0f} "
              f"| {median('prefill_ms', rows):.1f} | {median('prefill_requests', rows):.0f} | {median('prefill_tokens', rows):.0f} "
              f"| {median('round_ms', rows):.2f} | {100 * median('loss_low', rows):.1f}% | {100 * median('loss_high', rows):.1f}% "
              f"| {median('q_steady', rows):.1f} | {median('q_decode_only', rows):.1f} |")
    print("\n| C | line | R measured | R without prefill time |\n|---:|---|---:|---:|")
    for concurrency in sorted({c for _, c in runs}):
        baseline = runs[(BASELINE, concurrency)]
        for line in sorted({name for name, c in runs if c == concurrency and name != BASELINE}):
            rows = runs[(line, concurrency)]
            print(f"| {concurrency} | {line} | {median('q_steady', rows) / median('q_steady', baseline):.2f} "
                  f"| {median('q_decode_only', rows) / median('q_decode_only', baseline):.2f} |")


if __name__ == "__main__":
    main()
