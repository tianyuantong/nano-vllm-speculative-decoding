"""Back-to-back benchmark of Co-located `LLM` vs `DisaggLLM` (PD disaggregation) on a prompt panel."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_spec import (  # noqa: E402
    QUERY_FIELDS,
    engine_kwargs,
    installed_packages,
    kv_block_bytes,
    kv_budget_bytes,
    load_panel,
    manifest,
    run_closed_loop,
)
from itl_stats import itl as colocated_itl  # noqa: E402
from summarize_bench import _end_ms, percentile, steady_window, summarize_v2  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402


def all_gpu_states() -> list[dict]:
    """Query `QUERY_FIELDS` across all visible GPUs plus the hostname."""
    lines = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    node = subprocess.run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    return [{**dict(zip(QUERY_FIELDS, (v.strip() for v in line.split(",")))), "node": node} for line in lines if line.strip()]


def count_slow_steps(steps: list[dict], factor: float = 3.0) -> int:
    """Count steps whose duration exceeds `factor` times the median duration of Steps of the same kind."""
    slow = 0
    for kind in {s["kind"] for s in steps}:
        durations = [s["duration_ms"] for s in steps if s["kind"] == kind]
        if durations:
            med = percentile(durations, 50)
            slow += sum(1 for d in durations if d > factor * med)
    return slow


def disagg_cuda_itl(d_steps: list[dict], outputs: list[dict], concurrency: int) -> dict:
    """ITL on D's CUDA-event timeline (`end_ms[b] - end_ms[a]` between consecutive decode deliveries)."""
    if not d_steps:
        return {"p50": 0.0, "p99": 0.0, "round_p50": 0.0, "round_p99": 0.0}
    end = {step["step"]: _end_ms(step) for step in d_steps}
    try:
        first, last = steady_window(d_steps, concurrency)
    except (IndexError, ValueError):
        first, last = 0, d_steps[-1]["step"]
    gaps = []
    for out in outputs:
        steps = [s for s, _ in out["d_deliveries"]]
        for a, b in zip(steps, steps[1:]):
            if first <= a and b <= last and a in end and b in end:
                gaps.append(end[b] - end[a])
    if not gaps:
        for out in outputs:
            steps = [s for s, _ in out["d_deliveries"]]
            for a, b in zip(steps, steps[1:]):
                if a in end and b in end:
                    gaps.append(end[b] - end[a])
    round_ms = [s["duration_ms"] for s in d_steps if first <= s["step"] <= last] or [s["duration_ms"] for s in d_steps]
    return {
        "p50": round(percentile(gaps, 50), 2) if gaps else 0.0,
        "p99": round(percentile(gaps, 99), 2) if gaps else 0.0,
        "round_p50": round(percentile(round_ms, 50), 2),
        "round_p99": round(percentile(round_ms, 99), 2),
    }


def disagg_q_steady(d_steps: list[dict], outputs: list[dict], concurrency: int) -> tuple[float | None, float | None]:
    """Steady-state throughput on D's CUDA-event timeline from actual committed deliveries (`out['d_deliveries']`).

    - `q_steady_tok_s` counts actual D-committed tokens inside `[first, last]` plus 1 prefill first-token for each
      replacement request whose first decode round lands in `(first, last]`, matching `summarize_bench.q_steady`.
    - `q_steady_decode_only_tok_s` counts only tokens committed by D rounds inside `[first, last]`.
    Returns `(None, None)` if D never formed a full `concurrency` batch.
    """
    try:
        first, last = steady_window(d_steps, concurrency)
    except (IndexError, ValueError):
        return None, None
    by_step = {step["step"]: step for step in d_steps}
    steady_ms = _end_ms(by_step[last]) - by_step[first]["start_ms"]
    if steady_ms <= 0:
        return None, None
    decode_tokens = sum(delta for out in outputs for step, delta in out["d_deliveries"] if first <= step <= last)
    admitted_first_tokens = sum(
        1 for out in outputs if out["d_deliveries"] and first < out["d_deliveries"][0][0] <= last
    )
    return (
        round((decode_tokens + admitted_first_tokens) / (steady_ms / 1e3), 1),
        round(decode_tokens / (steady_ms / 1e3), 1),
    )


def run_closed_loop_with_wall(llm, panel: list[dict], params: SamplingParams, concurrency: int) -> tuple[list[dict], list[float], list[float]]:
    """Drive `llm` in a closed loop while recording both `RequestRecord`s and host `perf_counter()` TTFT / ITL."""
    records, in_flight = [], []
    ttfts_ms: list[float] = []
    itls_ms: list[float] = []
    next_entry = 0
    while in_flight or next_entry < len(panel):
        while len(in_flight) < concurrency and next_entry < len(panel):
            entry = panel[next_entry]
            rec = {
                "id": entry["id"],
                "category": entry["category"],
                "prompt_len": len(entry["prompt_token_ids"]),
                "submit_step": llm.num_steps,
                "deliveries": [],
                "finish_step": -1,
                "token_ids": [],
            }
            seq = llm.add_request(entry["prompt_token_ids"], params)
            in_flight.append([rec, seq, 0, perf_counter(), None])
            records.append(rec)
            next_entry += 1
        step = llm.num_steps
        llm.step()
        now = perf_counter()
        for item in in_flight:
            rec, seq, delivered, enqueue_t, last_t = item
            count = seq.num_completion_tokens
            if count > delivered:
                rec["deliveries"].append([step, count - delivered])
                if last_t is None:
                    ttfts_ms.append((now - enqueue_t) * 1e3)
                else:
                    itls_ms.append((now - last_t) * 1e3)
                item[2] = count
                item[4] = now
            if seq.is_finished:
                rec["finish_step"] = step
                rec["token_ids"] = seq.completion_token_ids
        in_flight = [item for item in in_flight if not item[1].is_finished]
    return records, ttfts_ms, itls_ms


def run_colocated(target: str, kwargs: dict, panel: list[dict], sampling: dict, concurrency: int) -> dict:
    from nanovllm import LLM
    warmup_params = SamplingParams(
        temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
        ignore_eos=True, max_tokens=sampling["warmup_max_tokens"],
    )
    params = SamplingParams(
        temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
        ignore_eos=sampling["ignore_eos"], max_tokens=sampling["max_tokens"],
    )
    llm = LLM(target, **kwargs)
    try:
        run_closed_loop(llm, panel[:concurrency], warmup_params, concurrency)
        llm.start_measurement()
        t0 = perf_counter()
        requests, wall_ttfts, wall_itls = run_closed_loop_with_wall(llm, panel, params, concurrency)
        torch.cuda.synchronize()
        wall_s = perf_counter() - t0
        steps = llm.step_timings()
    finally:
        llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    fake_result = {
        "config": {"concurrency": concurrency, "num_speculative_tokens": kwargs.get("num_speculative_tokens", 0)},
        "steps": steps,
        "requests": requests,
        "wall_seconds": wall_s,
        "speculative": {"stats": {}, "phases": []},
    }
    summary = summarize_v2(fake_result)
    gaps = colocated_itl(fake_result, concurrency)
    total_tokens = sum(len(r["token_ids"]) for r in requests)
    return {
        "mode": "colocated",
        "concurrency": concurrency,
        "steady_window_reached": True,
        "q_steady_tok_s": round(summary["q_steady"], 1),
        "q_wall_tok_s": round(total_tokens / wall_s, 1),
        "ttft_cuda_p50_ms": round(summary["ttft_ms"]["p50"], 2),
        "ttft_cuda_p99_ms": round(summary["ttft_ms"]["p99"], 2),
        "ttft_wall_p50_ms": round(percentile(wall_ttfts, 50), 2),
        "ttft_wall_p99_ms": round(percentile(wall_ttfts, 99), 2),
        "itl_cuda_p50_ms": round(gaps["p50"], 2),
        "itl_cuda_p99_ms": round(gaps["p99"], 2),
        "itl_cuda_p99_no_prefill_ms": round(gaps["p99_clean"], 2),
        "itl_wall_p50_ms": round(percentile(wall_itls, 50), 2) if wall_itls else 0.0,
        "itl_wall_p99_ms": round(percentile(wall_itls, 99), 2) if wall_itls else 0.0,
        "prefill_stalled_intervals_pct": round(gaps["stalled_share"] * 100, 1),
        "round_p50_ms": round(summary["round_ms"]["p50"], 2),
        "round_p99_ms": round(summary["round_ms"]["p99"], 2),
        "slow_steps": count_slow_steps(steps),
    }


def run_disagg(
    target: str,
    kwargs: dict,
    panel: list[dict],
    sampling: dict,
    concurrency: int,
    p_device: int,
    d_device: int,
    prefetch: int,
) -> dict:
    from nanovllm.disagg import DisaggLLM
    warmup_params = SamplingParams(
        temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
        ignore_eos=True, max_tokens=sampling["warmup_max_tokens"],
    )
    params = SamplingParams(
        temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
        ignore_eos=sampling["ignore_eos"], max_tokens=sampling["max_tokens"],
    )
    # Size KV cache for C + prefetch so P can prefill `prefetch` arrivals while D decodes `C` active requests.
    budget_k = max(kwargs.get("num_speculative_tokens", 0), 7)
    disagg_kwargs = dict(kwargs)
    disagg_kwargs["kv_cache_memory_bytes"] = kv_budget_bytes(
        panel, concurrency + max(prefetch, 2), sampling["max_tokens"], budget_k, kv_block_bytes(target)
    )
    prompts = [entry["prompt_token_ids"] for entry in panel]

    disagg = DisaggLLM(target, prefill_device=p_device, decode_device=d_device, **disagg_kwargs)
    try:
        disagg.generate(prompts[:concurrency], warmup_params, threaded=True, concurrency=concurrency)
        disagg.start_measurement()
        t0 = perf_counter()
        outputs = disagg.generate(prompts, params, threaded=True, concurrency=concurrency + prefetch)
        torch.cuda.synchronize(disagg.p_device)
        torch.cuda.synchronize(disagg.d_device)
        wall_s = perf_counter() - t0
        d_steps = disagg.decode_engine.step_timings()
        p_steps = disagg.prefill_engine.step_timings()
    finally:
        disagg.exit()
    del disagg
    gc.collect()
    torch.cuda.empty_cache()

    cuda_gaps = disagg_cuda_itl(d_steps, outputs, concurrency)
    wall_itls = [gap for out in outputs for gap in out["itl_ms"]]
    ttfts = [out["ttft_ms"] for out in outputs if out["ttft_ms"] is not None]
    total_tokens = sum(len(out["token_ids"]) for out in outputs)
    q_steady_val, q_steady_decode_val = disagg_q_steady(d_steps, outputs, concurrency)
    steady_reached = q_steady_val is not None
    if q_steady_val is None:
        q_steady_val = round(total_tokens / wall_s, 1)

    return {
        "mode": f"disagg_p{p_device}_d{d_device}_pf{prefetch}",
        "concurrency": concurrency,
        "steady_window_reached": steady_reached,
        "q_steady_tok_s": q_steady_val,
        "q_steady_decode_only_tok_s": q_steady_decode_val,
        "q_wall_tok_s": round(total_tokens / wall_s, 1),
        "ttft_wall_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_wall_p99_ms": round(percentile(ttfts, 99), 2),
        "itl_cuda_p50_ms": cuda_gaps["p50"],
        "itl_cuda_p99_ms": cuda_gaps["p99"],
        "itl_wall_p50_ms": round(percentile(wall_itls, 50), 2) if wall_itls else 0.0,
        "itl_wall_p99_ms": round(percentile(wall_itls, 99), 2) if wall_itls else 0.0,
        "prefill_stalled_intervals_pct": 0.0,
        "round_p50_ms": cuda_gaps["round_p50"],
        "round_p99_ms": cuda_gaps["round_p99"],
        "p_steps": len(p_steps),
        "d_steps": len(d_steps),
        "p_active_ms": round(sum(s["duration_ms"] for s in p_steps), 1),
        "d_active_ms": round(sum(s["duration_ms"] for s in d_steps), 1),
        "slow_steps_p": count_slow_steps(p_steps),
        "slow_steps_d": count_slow_steps(d_steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.3.json")
    parser.add_argument("--panel", type=str, default="D48")
    parser.add_argument("--lines", nargs="+", default=["k0", "dspark_k7"])
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--prefill-device", type=int, default=0)
    parser.add_argument("--decode-device", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--prefetches", nargs="+", type=int, default=None)
    parser.add_argument("--models-root", type=str, default="/content/models")
    parser.add_argument("--commit", type=str, default=None)
    parser.add_argument("--worker-mode", choices=["colocated", "disagg"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    panel_spec = protocol["panels"][args.panel]
    panel_path = ROOT / panel_spec["file"]
    sampling = {**protocol["sampling"], "warmup_max_tokens": protocol["warmup"]["max_tokens"]}

    if args.worker_mode is not None:
        line_name = args.lines[0]
        c = args.concurrencies[0]
        line = protocol["lines"][line_name]
        panel = load_panel(panel_path, c, panel_spec["take"])
        target, _, kwargs = engine_kwargs(protocol, line, panel, c, args.seed, args.models_root)
        if args.worker_mode == "colocated":
            row = run_colocated(target, kwargs, panel, sampling, c)
        else:
            row = run_disagg(
                target, kwargs, panel, sampling, c, args.prefill_device, args.decode_device, args.prefetch
            )
        row["line"] = line_name
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2) + "\n")
        return 0

    commit = args.commit or subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    state_start = all_gpu_states()
    prefetches = args.prefetches if args.prefetches is not None else [args.prefetch]
    rows = []
    for line_name in args.lines:
        for c in args.concurrencies:
            pair = {}
            runs = [("colocated", 0)] + [("disagg", pf) for pf in prefetches]
            for mode, pf in runs:
                tag = mode if mode == "colocated" else f"disagg_pf{pf}"
                tmp_out = args.output.with_suffix(f".{line_name}.c{c}.{tag}.json")
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--protocol", str(args.protocol),
                    "--panel", args.panel,
                    "--lines", line_name,
                    "--concurrencies", str(c),
                    "--seed", str(args.seed),
                    "--prefill-device", str(args.prefill_device),
                    "--decode-device", str(args.decode_device),
                    "--prefetch", str(pf),
                    "--models-root", args.models_root,
                    "--worker-mode", mode,
                    "--output", str(tmp_out),
                ]
                subprocess.run(cmd, check=True)
                row = json.loads(tmp_out.read_text())
                tmp_out.unlink(missing_ok=True)
                rows.append(row)
                pair[tag] = row
            print(json.dumps({"line": line_name, "c": c, **pair}, indent=2), flush=True)

    state_end = all_gpu_states()
    packages, packages_sha256 = installed_packages()
    out_data = {
        "schema": "bench-pd-v1",
        "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        "manifest": {
            **manifest(commit),
            "gpus": [g["name"] for g in state_start],
            "node": state_start[0]["node"],
            "packages_sha256": packages_sha256,
            "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        },
        "gpu_state_start": state_start,
        "gpu_state_end": state_end,
        "gpu": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "prefill_device": args.prefill_device,
        "decode_device": args.decode_device,
        "seed": args.seed,
        "panel": args.panel,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_data, indent=2) + "\n")
    args.output.with_name("packages.txt").write_text(packages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
