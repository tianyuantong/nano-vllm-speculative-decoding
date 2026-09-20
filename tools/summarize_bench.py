"""Metrics from a bench_spec.py result file, and a markdown table over several of them."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPEATED_NGRAM = 8
TAIL_WINDOW = 64


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def throughput(timings: list[dict], outputs: list[dict]) -> float:
    """Completion tokens per second over the GPU timeline of the whole generate() call."""
    last = timings[-1]
    elapsed_ms = last["start_ms"] + last["duration_ms"] - timings[0]["start_ms"]
    return sum(len(output["token_ids"]) for output in outputs) / (elapsed_ms / 1000)


def request_latencies(timings: list[dict], outputs: list[dict]) -> list[dict]:
    end_ms = [timing["start_ms"] + timing["duration_ms"] for timing in timings]
    latencies = []
    for output in outputs:
        ttft = end_ms[output["first_token_step"]]
        e2e = end_ms[output["finish_step"]]
        num_tokens = len(output["token_ids"])
        tpot = (e2e - ttft) / (num_tokens - 1) if num_tokens > 1 else 0.0
        latencies.append({"ttft_ms": ttft, "e2e_ms": e2e, "tpot_ms": tpot})
    return latencies


def has_repeated_tail(token_ids: list[int]) -> bool:
    tail = token_ids[-TAIL_WINDOW:]
    ngrams = [tuple(tail[i:i + REPEATED_NGRAM]) for i in range(len(tail) - REPEATED_NGRAM + 1)]
    return len(ngrams) != len(set(ngrams))


def summarize(result: dict) -> dict:
    timings, outputs = result["step_timings"], result["outputs"]
    latencies = request_latencies(timings, outputs)
    by_kind_batch = defaultdict(list)
    for timing in timings:
        by_kind_batch[(timing["kind"], timing["batch_size"])].append(timing["duration_ms"])
    stats = result.get("speculative_stats") or {}
    k = result["config"]["num_speculative_tokens"]
    rounds = stats.get("rounds", 0)
    accepted = [stats.get(f"accepted_{count}", 0) for count in range(k + 1)]
    summary = {
        "throughput_tok_s": throughput(timings, outputs),
        "ttft_ms": {q: percentile([l["ttft_ms"] for l in latencies], q) for q in (50, 95)},
        "tpot_ms": {q: percentile([l["tpot_ms"] for l in latencies], q) for q in (50, 95)},
        "e2e_ms": {q: percentile([l["e2e_ms"] for l in latencies], q) for q in (50, 95)},
        "step_ms": {f"{kind}@{batch}": {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}
                    for (kind, batch), values in sorted(by_kind_batch.items())},
        "capped_share": sum(len(o["token_ids"]) >= result["config"]["max_tokens"] for o in outputs) / len(outputs),
        "repeated_tail_share": sum(has_repeated_tail(o["token_ids"]) for o in outputs) / len(outputs),
    }
    rows = sum(accepted)
    if rows:
        summary["tokens_per_row_round"] = sum(count * (n + 1) for n, count in enumerate(accepted)) / rows
        summary["accept_rate_by_position"] = [sum(accepted[n + 1:]) / rows for n in range(k)]
        summary["rounds"] = rounds
    phases = defaultdict(list)
    for entry in result.get("phase_timings") or []:
        phases[entry["phase"]].append(entry["duration_ms"])
    if phases:
        summary["phase_ms_p50"] = {phase: percentile(values, 50) for phase, values in phases.items()}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.results:
        result = json.loads(path.read_text())
        summary = summarize(result)
        config = result["config"]
        rows.append((config["max_num_seqs"], config["temperature"], config["num_speculative_tokens"], path.name, summary))
    print("| B | T | k | file | tok/s | TPOT p50/p95 ms | TTFT p50 ms | tokens/row/round | capped | repeated tail |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for batch, temperature, k, name, s in sorted(rows):
        tokens_per_round = f"{s['tokens_per_row_round']:.2f}" if "tokens_per_row_round" in s else "-"
        print(f"| {batch} | {temperature} | {k} | {name} | {s['throughput_tok_s']:.1f} | "
              f"{s['tpot_ms'][50]:.2f}/{s['tpot_ms'][95]:.2f} | {s['ttft_ms'][50]:.0f} | {tokens_per_round} | "
              f"{s['capped_share']:.2f} | {s['repeated_tail_share']:.2f} |")


if __name__ == "__main__":
    main()
