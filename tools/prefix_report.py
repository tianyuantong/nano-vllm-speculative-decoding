"""Protocol v2.5's main verdicts over one shared-prefix panel's results, as markdown, and every pair's numbers as JSON.

Per run (main.definitions): the steady window's F, D, I, O, N_pf and N_r; Q_steady, Q_full, p, t_pf, t_r, h_tok, the
mean context of its steps of C rows and the mean of each round phase over them. Per (C, line, seed), the cache on
against off: G and its split into f, r and i, h, d, kappa, d of each phase, TTFT over every request and over the
requests both windows submitted, paired; per drafting line also rho, rho_hat_tok, rho_hat_0 and rho_hat_1. The
verdicts: null (S0's tokens), hits (a run's), gain on N_t and rho on N_rho, both from the job's repeats, only where all
seeds are there. Invalid runs and the split-job rule are report_v2's; a rerun directory listed last replaces the runs it
repeats.

  python tools/prefix_report.py --protocol benchmarks/protocol-v2.5.json --panel S2048 --results DIR [DIR ...] \\
      [--json OUT.json]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import (count_runs, final_runs, format_range, load_runs, timing_noise, tokens_equal,  # noqa: E402
                       verdict_a_beats_b)
from summarize_bench import decode_rows, percentile, prefill_tokens, round_weighted_tokens, steady_window, ttft_ms  # noqa: E402


def _end_ms(step: dict) -> float:
    return step["start_ms"] + step["duration_ms"]


def hits_check(result: dict, shared_tokens: int, on: bool) -> list[str]:
    """main.verdicts.hits for one run, whose line has the cache on or off: what fails, empty when it holds."""
    requests = result["requests"]
    problems = []
    expected = shared_tokens if on else 0
    if any(request["cached_tokens"] != expected for request in requests):
        problems.append(f"a request hit other than {expected} tokens")
    if sum(prefill_tokens(step) for step in result["steps"]) != sum(request["prompt_len"] - request["cached_tokens"]
                                                                    for request in requests):
        problems.append("prefilled tokens differ from prompt_len - cached_tokens")
    return problems


def round_phases(result: dict) -> dict[int, dict[str, float]]:
    """step -> its round's phase durations: the phases come round by round, each round's first is propose."""
    groups = []
    for phase in result.get("phases") or []:
        if phase["phase"] == "propose":
            groups.append(defaultdict(float))
        groups[-1][phase["phase"]] += phase["duration_ms"]
    rounds = [step["step"] for step in result["steps"] if step["kind"] == "round"]
    assert not groups or len(groups) == len(rounds), "phases and rounds do not line up"
    return dict(zip(rounds, groups))


def run_metrics(result: dict, summary: dict) -> dict:
    steps, requests, concurrency = result["steps"], result["requests"], result["config"]["concurrency"]
    first, last = steady_window(steps, concurrency)
    window = steps[first:last + 1]
    window_ms = _end_ms(steps[last]) - steps[first]["start_ms"]
    prefill_ms = sum(step["duration_ms"] for step in window if prefill_tokens(step))
    full = [step for step in window if decode_rows(step) == concurrency]
    full_ms = sum(step["duration_ms"] for step in full)
    prefill_steps = {step["step"] for step in window if prefill_tokens(step)}
    prefilled = [request for request in requests if request["deliveries"][0][0] in prefill_steps]
    assert len(prefilled) == sum(step["batch_size"] for step in window if prefill_tokens(step)), "a chunked prefill"
    delivered = sum(n for request in requests for step, n in request["deliveries"] if first <= step <= last)
    context = defaultdict(int)
    for request in requests:
        done = 0
        for step, n in request["deliveries"]:
            context[step] += request["prompt_len"] + done
            done += n
    phases = round_phases(result)
    full_phases = [phases[step["step"]] for step in full if step["step"] in phases]
    ttft = ttft_ms(steps, requests)
    stats = result["speculative_stats"]
    return {"window": [first, last], "window_ms": window_ms, "F": prefill_ms, "D": full_ms,
            "I": window_ms - prefill_ms - full_ms, "O": delivered, "N_pf": len(prefilled), "N_r": len(full),
            "q_steady": summary["q_steady"], "q_full": summary["q_full"], "p": prefill_ms / window_ms,
            "s": full_ms / window_ms,
            "t_pf_ms": prefill_ms / len(prefilled), "t_r_ms": full_ms / len(full),
            "h_tok": sum(r["cached_tokens"] for r in prefilled) / sum(r["prompt_len"] for r in prefilled),
            "prompt_len_prefilled": sum(r["prompt_len"] for r in prefilled),
            "context_mean": statistics.fmean(context[step["step"]] / concurrency for step in full),
            "phase_ms": ({name: statistics.fmean(group.get(name, 0.0) for group in full_phases)
                          for name in full_phases[0]} if full_phases else {}),
            "ttft_by_request": {request["id"]: value for request, value in zip(requests, ttft)},
            "steady_ids": [request["id"] for request in requests if first <= request["submit_step"] <= last],
            "ttft_ms": {"p50": percentile(ttft, 50), "p99": percentile(ttft, 99)},
            "tokens_per_round": round_weighted_tokens(stats) if stats else None,
            "output_len_p50": statistics.median(len(request["token_ids"]) for request in requests)}


def compare(off: dict, on: dict) -> dict:
    """One line's cache-off and cache-on runs of one seed."""
    per_token = {side: {key: run[key] / run["O"] for key in ("F", "D", "I")} for side, run in (("off", off), ("on", on))}
    total_on = sum(per_token["on"].values())
    paired = sorted(set(off["steady_ids"]) & set(on["steady_ids"]))
    ttft = {side: [run["ttft_by_request"][request] for request in paired] for side, run in (("off", off), ("on", on))}
    return {"G": on["q_steady"] / off["q_steady"] - 1,
            "G_split": {key: (per_token["off"][key] - per_token["on"][key]) / total_on for key in ("F", "D", "I")},
            "h": 1 - on["t_pf_ms"] / off["t_pf_ms"], "d": 1 - on["t_r_ms"] / off["t_r_ms"],
            "kappa_pf": (on["N_pf"] / on["O"]) / (off["N_pf"] / off["O"]),
            "kappa_r": (on["N_r"] / on["O"]) / (off["N_r"] / off["O"]),
            "d_phase": {name: 1 - on["phase_ms"][name] / off["phase_ms"][name]
                        for name in off["phase_ms"] if off["phase_ms"][name] > 0 and name in on["phase_ms"]},
            "steady_ttft_ms": {"n": len(paired), **({side: {"p50": percentile(values, 50), "p99": percentile(values, 99)}
                                                      for side, values in ttft.items()} if paired else {})}}


def h_tok_off(run: dict, shared_tokens: int) -> float:
    """The hit rate the panel's design gives the requests a cache-off run prefills inside its window."""
    return shared_tokens * run["N_pf"] / run["prompt_len_prefilled"]


def rho_terms(k0_off: dict, k0_on: dict, line_off: dict, line_on: dict, shared_tokens: int) -> dict:
    """rho and its three model values from the four runs of one seed (main.definitions): rho_hat_tok from the cache-off
    runs alone, a prediction conditional on them; rho_hat_0 and rho_hat_1 accountings of the measured h and d."""
    k0, line = compare(k0_off, k0_on), compare(line_off, line_on)
    rho = (line_on["q_steady"] / k0_on["q_steady"]) / (line_off["q_steady"] / k0_off["q_steady"])
    p0, p, s0, s = k0_off["p"], line_off["p"], k0_off["s"], line_off["s"]
    return {"rho": rho,
            "rho_hat_tok": (1 - p0 * h_tok_off(k0_off, shared_tokens)) / (1 - p * h_tok_off(line_off, shared_tokens)),
            "rho_hat_0": (1 - p0 * k0["h"]) / (1 - p * line["h"]),
            "rho_hat_1": (1 - p0 * k0["h"] - s0 * k0["d"]) / (1 - p * line["h"] - s * line["d"])}


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


def rho_of(q: dict, line: str) -> float:
    return (q[f"{line}_pc"] / q["k0_pc"]) / (q[line] / q["k0"])


def analyse(runs: list[tuple[str, dict, dict]], protocol: dict, panel: str) -> dict:
    main_spec = protocol["main"]
    shared = protocol["panels"][panel]["shared_tokens"]
    runs = [run for run in runs if run[1]["config"]["panel"] == panel]
    invalid, _, _ = count_runs(runs, protocol)
    for name, result, _ in runs:
        on = protocol["lines"][result["config"]["line"]].get("prefix_cache", False)
        invalid[name] += [f"hits: {problem}" for problem in hits_check(result, shared, on)]
    by_key = final_runs(runs)
    repeats = defaultdict(dict)    # (C, repeat) -> line -> Q_steady of a valid run, at the timing-noise seed
    for (_, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        if seed == protocol["timing_noise"]["seed"] and not invalid[name]:
            repeats[(concurrency, repeat)][line] = summary["q_steady"]
    bases = [off for off, _ in main_spec["pairs"]]
    drafting = [line for line in bases if protocol["lines"][line]["k"] > 0]
    pairs = [(repeats[(c, 1)][line], repeats[(c, 2)][line]) for c in main_spec["concurrency"] for line in main_spec["lines"]
             if all(line in repeats[(c, r)] for r in (1, 2))]
    rho_pairs = [(rho_of(repeats[(c, 1)], line), rho_of(repeats[(c, 2)], line))
                 for c in main_spec["concurrency"] for line in drafting
                 if all(name in repeats[(c, r)] for r in (1, 2) for name in (line, f"{line}_pc", "k0", "k0_pc"))]
    noise_complete = (len(pairs) == len(main_spec["concurrency"]) * len(main_spec["lines"])
                      and len(rho_pairs) == len(main_spec["concurrency"]) * len(drafting))
    noise = timing_noise(pairs) if pairs else None
    noise_rho = timing_noise(rho_pairs) if rho_pairs else None
    metrics, results = {}, {}
    for (_, concurrency, line, seed, repeat), (name, result, summary) in by_key.items():
        if repeat == 1 and not invalid[name]:
            metrics[(concurrency, line, seed)] = run_metrics(result, summary)
            results[(concurrency, line, seed)] = result
    seeds = protocol["seeds"][main_spec["seeds"]]
    null = []
    if shared == 0:    # main.verdicts.null reads every run, a timing-invalid one too
        for (_, concurrency, line, seed, repeat), (_, result, _) in by_key.items():
            twin = by_key.get((panel, concurrency, f"{line}_pc", seed, repeat))
            if line in bases and twin is not None and not tokens_equal(result, twin[1]):
                null.append(f"C={concurrency} {line} seed {seed} repeat {repeat}: "
                            f"{tokens_equal_share(result, twin[1]):.4f} of the requests the same")
    table = {}
    for concurrency in main_spec["concurrency"]:
        for line in bases:
            entries = []
            for seed in seeds:
                keys = [(concurrency, name, seed) for name in (line, f"{line}_pc", "k0", "k0_pc")]
                if not all(key in metrics for key in keys):
                    continue
                off, on, k0_off, k0_on = (metrics[key] for key in keys)
                same = tokens_equal_share(results[keys[0]], results[keys[1]])
                entry = {"seed": seed, **compare(off, on), "tokens_equal_share": same,
                         "off": {key: value for key, value in off.items() if key not in ("ttft_by_request", "steady_ids")},
                         "on": {key: value for key, value in on.items() if key not in ("ttft_by_request", "steady_ids")}}
                if line in drafting:
                    entry.update(rho_terms(k0_off, k0_on, off, on, shared))
                entries.append(entry)
            table[(concurrency, line)] = entries
    null_failures = null if shared == 0 else None
    stopped = bool(null_failures) or any(reason.startswith("hits:") for reasons in invalid.values() for reason in reasons)
    return {"panel": panel, "shared_tokens": shared, "noise": noise, "pairs": len(pairs), "noise_rho": noise_rho,
            "rho_pairs": len(rho_pairs), "noise_complete": noise_complete, "stopped": stopped, "seeds": len(seeds),
            "null_failures": null_failures,
            "invalid": {name: reasons for name, reasons in invalid.items() if reasons}, "table": table}


def _fmt(values: list[float], scale: float = 1, digits: int = 2) -> str:
    values = [value * scale for value in values]
    if not values:
        return "-"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def markdown(analysis: dict) -> str:
    complete = analysis["seeds"]
    judged = analysis["noise_complete"] and not analysis["stopped"]
    noise, noise_rho = (analysis["noise"], analysis["noise_rho"]) if judged else (None, None)
    out = [f"# {analysis['panel']}: shared prefix {analysis['shared_tokens']} tokens", "",
           "- verdicts: " + ("given" if judged else "NOT GIVEN: " + ("main.verdicts.null or hits failed"
                                                                       if analysis["stopped"] else "the noise repeats are incomplete")),
           f"- N_t = {analysis['noise']:.4f} from {analysis['pairs']} repeat pairs" if analysis["noise"] is not None
           else "- N_t unavailable",
           f"- N_rho = {analysis['noise_rho']:.4f} from {analysis['rho_pairs']} repeat pairs"
           if analysis["noise_rho"] is not None else "- N_rho unavailable"]
    if analysis["null_failures"] is not None:
        out.append("- null (S0, tokens the same on and off): " +
                   ("holds" if not analysis["null_failures"] else "FAILS: " + "; ".join(analysis["null_failures"])))
    out.append("")
    if analysis["invalid"]:
        out += ["## Invalid runs", ""] + [f"- {Path(name).name}: {', '.join(reasons)}"
                                          for name, reasons in sorted(analysis["invalid"].items())] + [""]
    out += ["## Cache on against off", "",
            "| C | line | seeds | G % | verdict | of it F / D / I % | h | d | κ_pf | κ_r | ρ | ρ verdict | ρ − ρ̂_tok % | "
            "ρ − ρ̂₀ % | ρ − ρ̂₁ % | same tokens |",
            "|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for (concurrency, line), entries in analysis["table"].items():
        whole = len(entries) == complete
        gains = [entry["G"] for entry in entries]
        verdict = gain_verdict(gains, noise) if whole and noise is not None else "-"
        rho, rho_verdict, gaps = "-", "-", ["-", "-", "-"]
        if entries and "rho" in entries[0]:
            rhos = [entry["rho"] for entry in entries]
            rho = format_range(rhos)
            if whole and noise_rho is not None:
                rho_verdict = {1: "larger", -1: "smaller", 0: "undecided"}[verdict_a_beats_b([r - 1 for r in rhos], noise_rho)]
            gaps = [_fmt([entry["rho"] - entry[key] for entry in entries], 100)
                    for key in ("rho_hat_tok", "rho_hat_0", "rho_hat_1")]
        split = " / ".join(f"{statistics.median(entry['G_split'][key] for entry in entries) * 100:.1f}"
                           for key in ("F", "D", "I")) if entries else "-"
        out.append(f"| {concurrency} | {line} | {len(entries)} | {_fmt(gains, 100)} | {verdict} | {split} | "
                   f"{_fmt([entry['h'] for entry in entries])} | {_fmt([entry['d'] for entry in entries], 1, 3)} | "
                   f"{_fmt([entry['kappa_pf'] for entry in entries], 1, 3)} | "
                   f"{_fmt([entry['kappa_r'] for entry in entries], 1, 3)} | {rho} | {rho_verdict} | {' | '.join(gaps)} | "
                   f"{_fmt([entry['tokens_equal_share'] for entry in entries])} |")
    out += ["", "## Throughput, TTFT, context and tokens per round, off → on (medians over the seeds)", "",
            "| C | line | Q_steady tok/s | Q_full tok/s | TTFT p50 ms | TTFT p99 ms | steady TTFT p50 ms (n) | "
            "steady TTFT p99 ms | context | tokens/round | output p50 | d by phase |",
            "|---:|---|---|---|---|---|---|---|---|---|---|---|"]

    def pair(entries, pick, digits=0):
        values = [(pick(entry["off"]), pick(entry["on"])) for entry in entries]
        if not values or values[0][0] is None:
            return "-"
        return (f"{statistics.median(a for a, _ in values):.{digits}f} → "
                f"{statistics.median(b for _, b in values):.{digits}f}")

    for (concurrency, line), entries in analysis["table"].items():
        steady = [entry["steady_ttft_ms"] for entry in entries if entry["steady_ttft_ms"]["n"]]
        steady_p50 = (f"{statistics.median(s['off']['p50'] for s in steady):.0f} → "
                      f"{statistics.median(s['on']['p50'] for s in steady):.0f} "
                      f"({min(s['n'] for s in steady)}–{max(s['n'] for s in steady)})") if steady else "-"
        steady_p99 = (f"{statistics.median(s['off']['p99'] for s in steady):.0f} → "
                      f"{statistics.median(s['on']['p99'] for s in steady):.0f}") if steady else "-"
        phases = sorted({name for entry in entries for name in entry["d_phase"]})
        by_phase = ", ".join(f"{name} {statistics.median(entry['d_phase'][name] for entry in entries if name in entry['d_phase']):.3f}"
                             for name in phases) or "-"
        out.append(f"| {concurrency} | {line} | {pair(entries, lambda run: run['q_steady'])} | "
                   f"{pair(entries, lambda run: run['q_full'])} | {pair(entries, lambda run: run['ttft_ms']['p50'])} | "
                   f"{pair(entries, lambda run: run['ttft_ms']['p99'])} | {steady_p50} | {steady_p99} | "
                   f"{pair(entries, lambda run: run['context_mean'])} | {pair(entries, lambda run: run['tokens_per_round'], 2)} | "
                   f"{pair(entries, lambda run: run['output_len_p50'])} | {by_phase} |")
    out += ["", "G = Q_steady(on) / Q_steady(off) − 1, split exactly into what F, D and I per delivered token each moved; "
            "ρ = R(on) / R(off); the rest as protocol v2.5 main.definitions. Verdicts only where all seeds are there: "
            "gain on N_t, ρ on N_rho; the rest is reported only."]
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
                                                                            "noise_rho", "rho_pairs", "null_failures",
                                                                            "invalid")}, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
