"""Metrics from a bench_spec.py result file, and a markdown table over several of them."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
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


# ---------------------------------------------------------------- bench-v2 (closed loop, bench-v2 design §1.5)
# MBU constants of Qwen3-8B in bf16: 6.946 B parameters in the 36 layers plus 0.622 B in lm_head; K and V of
# 36 layers × 8 heads × 128 per token; RTX PRO 6000 Blackwell Server Edition peak bandwidth.
WEIGHT_BYTES = (6.946e9 + 0.622e9) * 2
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
PEAK_BYTES_PER_S = 1.79e12


def _end_ms(step: dict) -> float:
    return step["start_ms"] + step["duration_ms"]


def steady_window(steps: list[dict], concurrency: int) -> tuple[int, int]:
    """First and last step that is not a prefill and runs a full batch of C requests."""
    full = [step["step"] for step in steps if step["kind"] != "prefill" and step["batch_size"] == concurrency]
    return full[0], full[-1]


def q_steady(steps: list[dict], requests: list[dict], concurrency: int) -> float:
    """Tokens delivered inside the steady window over the window's length; prefills inside it count."""
    first, last = steady_window(steps, concurrency)
    tokens = sum(n for request in requests for step, n in request["deliveries"] if first <= step <= last)
    return tokens / ((_end_ms(steps[last]) - steps[first]["start_ms"]) / 1000)


def q_full(steps: list[dict], requests: list[dict]) -> float:
    tokens = sum(n for request in requests for _, n in request["deliveries"])
    return tokens / ((_end_ms(steps[-1]) - steps[0]["start_ms"]) / 1000)


def ttft_ms(steps: list[dict], requests: list[dict]) -> list[float]:
    """Per request, in request order: end of its first delivery's step minus start of its submit step.
    A list, not a dict by id: a panel cycled for large C repeats ids."""
    return [_end_ms(steps[request["deliveries"][0][0]]) - steps[request["submit_step"]]["start_ms"]
            for request in requests]


def itl_ms(steps: list[dict], requests: list[dict], window: tuple[int, int]) -> list[float]:
    """Gaps between consecutive deliveries of a request, when both ends lie in the window."""
    return [_end_ms(steps[b]) - _end_ms(steps[a])
            for request in requests for (a, _), (b, _) in pairwise(request["deliveries"])
            if window[0] <= a and b <= window[1]]


def round_ms(steps: list[dict], window: tuple[int, int]) -> list[float]:
    return [step["duration_ms"] for step in steps[window[0]:window[1] + 1] if step["kind"] != "prefill"]


def tokens_per_round(requests: list[dict]) -> list[float | None]:
    """Per request, in request order: tokens per delivery after the prefill's; None when the prefill finished it."""
    return [statistics.fmean(n for _, n in request["deliveries"][1:]) if len(request["deliveries"]) > 1 else None
            for request in requests]


def round_weighted_tokens(speculative_stats: dict) -> float | None:
    """Tokens per round over all of a run's rounds, as vLLM and DeepSpec count it: sum of (accepted + 1) ÷ rounds."""
    counts = {int(key.split("_")[1]): n for key, n in speculative_stats.items() if key.startswith("accepted_")}
    rounds = sum(counts.values())
    return sum((accepted + 1) * n for accepted, n in counts.items()) / rounds if rounds else None


def excess_prefill_tokens(steps: list[dict], requests: list[dict]) -> int:
    """Prefilled tokens beyond the prompts submitted; above 0 means a request was preempted and recomputed."""
    return (sum(step["tokens_out"] for step in steps if step["kind"] == "prefill")
            - sum(request["prompt_len"] for request in requests))


def full_batch_share(steps: list[dict], concurrency: int) -> float:
    batches = [step["batch_size"] for step in steps if step["kind"] != "prefill"]
    return sum(batch == concurrency for batch in batches) / len(batches)


def prefill_share(steps: list[dict]) -> float:
    return (sum(step["duration_ms"] for step in steps if step["kind"] == "prefill")
            / sum(step["duration_ms"] for step in steps))


def mbu(steps: list[dict], requests: list[dict], weight_bytes: float, kv_bytes_per_token: int,
        peak_bytes_per_s: float) -> list[float]:
    """Per decode step: (weights + context of the requests it decodes × KV per token) / step time / peak bandwidth.
    A request's context at a step is its prompt plus the tokens delivered before that step."""
    context = defaultdict(int)
    for request in requests:
        delivered = 0
        for step, n in request["deliveries"]:
            context[step] += request["prompt_len"] + delivered
            delivered += n
    return [(weight_bytes + context[step["step"]] * kv_bytes_per_token) / (step["duration_ms"] / 1000) / peak_bytes_per_s
            for step in steps if step["kind"] == "decode"]


def summarize_v2(result: dict) -> dict:
    steps, requests = result["steps"], result["requests"]
    concurrency = result["config"]["concurrency"]
    window = steady_window(steps, concurrency)
    ttft, itl, rounds = ttft_ms(steps, requests), itl_ms(steps, requests, window), round_ms(steps, window)
    by_category = defaultdict(list)
    for request, value in zip(requests, tokens_per_round(requests)):
        if value is not None:
            by_category[request["category"]].append(value)
    decode_steps = [step["step"] for step in steps if step["kind"] == "decode"]
    window_mbu = [value for step, value in zip(decode_steps, mbu(steps, requests, WEIGHT_BYTES, KV_BYTES_PER_TOKEN,
                                                                 PEAK_BYTES_PER_S))
                  if window[0] <= step <= window[1]]
    completion_tokens = sum(n for request in requests for _, n in request["deliveries"])
    return {
        "q_steady": q_steady(steps, requests, concurrency),
        "q_full": q_full(steps, requests),
        "q_full_wall": completion_tokens / result["wall_seconds"],
        "window": window,
        "ttft_ms": {"p50": percentile(ttft, 50), "p99": percentile(ttft, 99)},
        "itl_ms": {"p50": percentile(itl, 50), "p99": percentile(itl, 99)} if itl else None,
        "round_ms": {"p50": percentile(rounds, 50), "p99": percentile(rounds, 99)},
        "tokens_per_round": {category: statistics.median(values) for category, values in sorted(by_category.items())},
        "excess_prefill_tokens": excess_prefill_tokens(steps, requests),
        "full_batch_share": full_batch_share(steps, concurrency),
        "prefill_share": prefill_share(steps),
        "mbu_p50": statistics.median(window_mbu) if window_mbu else None,
    }


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
