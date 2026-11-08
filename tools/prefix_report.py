"""Protocol v2.5's main verdicts over one shared-prefix panel's results, as markdown, and every run's metrics as JSON.

Per run (main.definitions): Q_steady, Q_full, p, t_pf, t_r, h_tok, TTFT over every request and over those submitted
inside the window, tokens per round, the median output length and the hits check. Per (C, line) with the cache on
against off: G, and per drafting line rho, rho_hat_0 and rho_hat_1 from the same seed's four runs. The verdicts:
gain and rho on the panel's N_t, from its timing-noise pairs; formula and the rest reported only. Invalid runs and
the split-job rule are report_v2's; a rerun directory listed last replaces the runs it repeats.

  python tools/prefix_report.py --protocol benchmarks/protocol-v2.5.json --panel S2048 --results DIR [DIR ...] \\
      [--json OUT.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import count_runs, final_runs, format_range, load_runs, timing_noise, verdict_a_beats_b  # noqa: E402
from summarize_bench import (decode_rows, percentile, prefill_tokens, round_weighted_tokens, steady_window,  # noqa: E402
                             ttft_ms)


def _end_ms(step: dict) -> float:
    return step["start_ms"] + step["duration_ms"]


def hits_check(result: dict, shared_tokens: int, on: bool) -> list[str]:
    """main.verdicts.hits for one run, whose line has the cache on or off: what fails, empty when it holds."""
    requests = result["requests"]
    problems = []
    if on and any(request["cached_tokens"] < shared_tokens for request in requests):
        problems.append(f"a request hit fewer than {shared_tokens} tokens")
    if not on and any(request["cached_tokens"] for request in requests):
        problems.append("a request hit with the cache off")
    if sum(prefill_tokens(step) for step in result["steps"]) != sum(request["prompt_len"] - request["cached_tokens"]
                                                                    for request in requests):
        problems.append("prefilled tokens differ from prompt_len - cached_tokens")
    return problems


def run_metrics(result: dict, summary: dict) -> dict:
    steps, requests, concurrency = result["steps"], result["requests"], result["config"]["concurrency"]
    first, last = steady_window(steps, concurrency)
    window = steps[first:last + 1]
    window_ms = _end_ms(steps[last]) - steps[first]["start_ms"]
    prefill_ms = sum(step["duration_ms"] for step in window if prefill_tokens(step))
    prefill_steps = {step["step"] for step in window if prefill_tokens(step)}
    prefilled = [request for request in requests if request["deliveries"][0][0] in prefill_steps]
    full_rounds = [step["duration_ms"] for step in window if decode_rows(step) == concurrency]
    ttft = ttft_ms(steps, requests)
    steady_ttft = [value for value, request in zip(ttft, requests) if first <= request["submit_step"] <= last]
    stats = result["speculative_stats"]
    return {"q_steady": summary["q_steady"], "q_full": summary["q_full"], "p": prefill_ms / window_ms,
            "t_pf_ms": prefill_ms / len(prefilled), "t_r_ms": statistics.fmean(full_rounds),
            "h_tok": sum(r["cached_tokens"] for r in prefilled) / sum(r["prompt_len"] for r in prefilled),
            "prefilled_in_window": len(prefilled),
            "ttft_ms": {"p50": percentile(ttft, 50), "p99": percentile(ttft, 99)},
            "steady_ttft_ms": ({"p50": percentile(steady_ttft, 50), "p99": percentile(steady_ttft, 99)}
                               if steady_ttft else None),
            "tokens_per_round": round_weighted_tokens(stats) if stats else None,
            "output_len_p50": statistics.median(len(request["token_ids"]) for request in requests)}


def compare(off: dict, on: dict) -> dict:
    """One line's on/off pair of one seed: G, h and d."""
    return {"G": on["q_steady"] / off["q_steady"] - 1, "h": 1 - on["t_pf_ms"] / off["t_pf_ms"],
            "d": 1 - on["t_r_ms"] / off["t_r_ms"]}


def rho_terms(k0_off: dict, k0_on: dict, line_off: dict, line_on: dict) -> dict:
    """rho, and its two predictions from the four runs of one seed (main.definitions)."""
    k0, line = compare(k0_off, k0_on), compare(line_off, line_on)
    rho = (line_on["q_steady"] / k0_on["q_steady"]) / (line_off["q_steady"] / k0_off["q_steady"])
    p0, p = k0_off["p"], line_off["p"]
    hat_0 = (1 - p0 * k0["h"]) / (1 - p * line["h"])
    hat_1 = (1 - p0 * k0["h"] - (1 - p0) * k0["d"]) / (1 - p * line["h"] - (1 - p) * line["d"])
    return {"rho": rho, "rho_hat_0": hat_0, "rho_hat_1": hat_1}


def gain_verdict(gains: list[float], noise: float) -> str:
    if all(1 + gain > 1 + 2 * noise for gain in gains):
        return "faster"
    if all(1 + gain < 1 - 2 * noise for gain in gains):
        return "slower"
    return "undecided"


def tokens_equal_share(off: dict, on: dict) -> float:
    pairs = list(zip(off["requests"], on["requests"]))
    assert all(a["id"] == b["id"] for a, b in pairs)
    return sum(a["token_ids"] == b["token_ids"] for a, b in pairs) / len(pairs)


def analyse(runs: list[tuple[str, dict, dict]], protocol: dict, panel: str) -> dict:
    main_spec = protocol["main"]
    shared = protocol["panels"][panel]["shared_tokens"]
    runs = [run for run in runs if run[1]["config"]["panel"] == panel]
    invalid, pairs, _ = count_runs(runs, protocol)
    for name, result, _ in runs:
        on = protocol["lines"][result["config"]["line"]].get("prefix_cache", False)
        invalid[name] += [f"hits: {problem}" for problem in hits_check(result, shared, on)]
    noise = timing_noise(pairs) if pairs else None
    chosen = {key: value for key, value in final_runs(runs).items() if key[4] == 1}
    metrics, results = {}, {}
    for (_, concurrency, line, seed, _), (name, result, summary) in chosen.items():
        if not invalid[name]:
            metrics[(concurrency, line, seed)] = run_metrics(result, summary)
            results[(concurrency, line, seed)] = result
    bases = [line for line in main_spec["lines"] if not protocol["lines"][line].get("prefix_cache")]
    seeds = protocol["seeds"][main_spec["seeds"]]
    table = {}
    for concurrency in main_spec["concurrency"]:
        for line in bases:
            entries = []
            for seed in seeds:
                keys = [(concurrency, name, seed) for name in (line, f"{line}_pc", "k0", "k0_pc")]
                if not all(key in metrics for key in keys):
                    continue
                off, on, k0_off, k0_on = (metrics[key] for key in keys)
                entry = {"seed": seed, **compare(off, on), "off": off, "on": on,
                         "tokens_equal_share": tokens_equal_share(results[keys[0]], results[keys[1]])}
                if line != "k0":
                    entry.update(rho_terms(k0_off, k0_on, off, on))
                entries.append(entry)
            table[(concurrency, line)] = entries
    return {"panel": panel, "shared_tokens": shared, "noise": noise, "pairs": len(pairs),
            "invalid": {name: reasons for name, reasons in invalid.items() if reasons}, "table": table}


def _fmt(values: list[float], scale: float = 1, digits: int = 2) -> str:
    values = [value * scale for value in values]
    if not values:
        return "-"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def markdown(analysis: dict) -> str:
    noise = analysis["noise"]
    out = [f"# {analysis['panel']}: shared prefix {analysis['shared_tokens']} tokens", "",
           f"N_t = {noise:.4f} from {analysis['pairs']} repeat pairs" if noise is not None else "N_t unavailable", ""]
    if analysis["invalid"]:
        out += ["## Invalid runs", ""] + [f"- {Path(name).name}: {', '.join(reasons)}"
                                          for name, reasons in sorted(analysis["invalid"].items())] + [""]
    out += ["## Cache on against off", "",
            "| C | line | seeds | G % | verdict | h_tok | p (off) | h | d | ρ | ρ verdict | ρ − ρ̂₀ % | ρ − ρ̂₁ % | same tokens |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for (concurrency, line), entries in analysis["table"].items():
        gains = [entry["G"] for entry in entries]
        verdict = gain_verdict(gains, noise) if entries and noise is not None else "-"
        rho, rho_verdict, gap_0, gap_1 = "-", "-", "-", "-"
        if entries and "rho" in entries[0]:
            rhos = [entry["rho"] for entry in entries]
            rho = format_range(rhos)
            if noise is not None:
                rho_verdict = {1: "larger", -1: "smaller", 0: "undecided"}[verdict_a_beats_b([r - 1 for r in rhos], noise)]
            gap_0 = _fmt([entry["rho"] - entry["rho_hat_0"] for entry in entries], 100)
            gap_1 = _fmt([entry["rho"] - entry["rho_hat_1"] for entry in entries], 100)
        out.append(f"| {concurrency} | {line} | {len(entries)} | {_fmt(gains, 100)} | {verdict} | "
                   f"{_fmt([entry['on']['h_tok'] for entry in entries])} | {_fmt([entry['off']['p'] for entry in entries])} | "
                   f"{_fmt([entry['h'] for entry in entries])} | {_fmt([entry['d'] for entry in entries], 1, 3)} | {rho} | "
                   f"{rho_verdict} | {gap_0} | {gap_1} | {_fmt([entry['tokens_equal_share'] for entry in entries])} |")
    out += ["", "## Throughput, TTFT and tokens per round, off → on", "",
            "| C | line | Q_steady tok/s | Q_full tok/s | TTFT p50 ms | TTFT p99 ms | steady TTFT p50 ms | steady TTFT p99 ms | "
            "tokens/round | output p50 |",
            "|---:|---|---|---|---|---|---|---|---|---|"]

    def pair(entries, pick, digits=0):
        off = [pick(entry["off"]) for entry in entries]
        on = [pick(entry["on"]) for entry in entries]
        if not entries or off[0] is None:
            return "-"
        return f"{statistics.median(off):.{digits}f} → {statistics.median(on):.{digits}f}"

    for (concurrency, line), entries in analysis["table"].items():
        steady = lambda run, q: run["steady_ttft_ms"][q] if run["steady_ttft_ms"] else None   # noqa: E731
        out.append(f"| {concurrency} | {line} | {pair(entries, lambda run: run['q_steady'])} | "
                   f"{pair(entries, lambda run: run['q_full'])} | {pair(entries, lambda run: run['ttft_ms']['p50'])} | "
                   f"{pair(entries, lambda run: run['ttft_ms']['p99'])} | {pair(entries, lambda run: steady(run, 'p50'))} | "
                   f"{pair(entries, lambda run: steady(run, 'p99'))} | "
                   f"{pair(entries, lambda run: run['tokens_per_round'], 2)} | {pair(entries, lambda run: run['output_len_p50'])} |")
    out += ["", "Medians over the seeds. G = Q_steady(on) / Q_steady(off) − 1; ρ = R(on) / R(off); h, d, p, h_tok as "
            "protocol v2.5 main.definitions. Verdicts: main.verdicts.gain and rho on this panel's N_t; the rest is reported only."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    analysis = analyse(load_runs(args.results), protocol, args.panel)
    print(markdown(analysis))
    if args.json:
        rows = [{"C": concurrency, "line": line, **entry} for (concurrency, line), entries in analysis["table"].items()
                for entry in entries]
        args.json.write_text(json.dumps({**{key: analysis[key] for key in ("panel", "shared_tokens", "noise", "pairs",
                                                                            "invalid")}, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
