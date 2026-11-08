"""M6 ③ (protocol v2.7 samehost_prefix): nano and vLLM back to back on one shared-prefix panel, on one machine, the
prefix cache on, and off too where the job runs both.

  report     one job: wall-clock Q_full of each line at each C over the dev seeds; V1, B = nano / vLLM, the ratio of the
             medians, judged where the two sides' seed ranges do not overlap and B is off 1 by more than 2 N1; on a job
             with both caches V2, each engine's throughput gain from its cache, 1 + G = Q(on) / Q(off) by seed, judged the
             same way on Gamma, the ratio of the two gains' medians, with N2. N1 and N2 come from the repeated groups: B
             and Gamma of each repeat, the largest relative difference. The checks each group had to pass, run again
             (invalid first, then nano's tokens against M6 ①'s run of the same name, then both sides' hits); V4, with
             the slow-step screening's effect: the verdicts again on the original runs of every rerun group. A rerun
             directory listed after the original replaces its runs.
  combine    V3 over the jobs' report JSON: B with the cache on at each P, and at the two ends B(S2048, on) / B(S0, on)
             split into the longer prefix with the cache off and the change of the cache's effect on B.
  reference  the sha256 of each M6 ① run's tokens a job repeats, with its wall-clock Q_full and packages_sha256.

  python tools/samehost_prefix_report.py report --protocol benchmarks/protocol-v2.7.json --panel S2048 \\
      --nano OUT/main/nano [OUT/rerun/nano] --vllm OUT/main/vllm [OUT/rerun/vllm] --reference REF.json [--json OUT.json]
  python tools/samehost_prefix_report.py combine --protocol benchmarks/protocol-v2.7.json REPORT.json [REPORT.json ...]
  python tools/samehost_prefix_report.py reference --protocol benchmarks/protocol-v2.7.json --out REF.json \\
      --results S2048=DIR[,DIR] S0=DIR[,DIR] ...
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_report import hits_check  # noqa: E402
from report_v2 import count_runs, final_runs, load_runs  # noqa: E402
from summarize_bench import round_weighted_tokens  # noqa: E402

CACHES = ("on", "off")


# ---------------------------------------------------------------- what the protocol's samehost_prefix section names

def section(protocol: dict) -> dict:
    return protocol["samehost_prefix"]


def job_of(protocol: dict, panel: str) -> dict:
    jobs = [job for job in section(protocol)["jobs"] if job["panel"] == panel]
    if len(jobs) != 1:
        raise ValueError(f"{panel} is not one of samehost_prefix.jobs")
    return jobs[0]


def nano_line(base: str, cache: str) -> str:
    """The nano line of a base line with the cache on or off: v2.6's lines, k0 and k0_pc."""
    return f"{base}_pc" if cache == "on" else base


def vllm_name(base: str, cache: str) -> str:
    """A vLLM run's line name: bench_vllm.py's line, with _pc when its prefix cache is on."""
    return f"{base}_pc" if cache == "on" else base


def requests_of(concurrency: int) -> int:
    """The measured requests of a run: the first max(48, 4C) of the panel (take "prefix")."""
    return max(48, 4 * concurrency)


def seeds_of(protocol: dict) -> list[int]:
    return protocol["seeds"][section(protocol)["seeds"]]


def repeats_of(protocol: dict, seed: int) -> int:
    return 2 if seed == section(protocol)["timing_noise"]["seed"] else 1


def expected_runs(protocol: dict, panel: str, concurrency: int, seed: int) -> tuple[dict, dict]:
    """(cache, base line, repeat) -> the file name of every nano and every vLLM run of one group."""
    caches, pairs = job_of(protocol, panel)["cache"], section(protocol)["pairs"]
    repeats = range(1, repeats_of(protocol, seed) + 1)
    nano = {(cache, base, repeat): f"{nano_line(base, cache)}-C{concurrency}-s{seed}-r{repeat}.json"
            for cache in caches for base, _ in pairs for repeat in repeats}
    vllm = {(cache, base, repeat): f"{vllm_name(base, cache)}-C{concurrency}-s{seed}-r{repeat}.json"
            for cache in caches for _, base in pairs for repeat in repeats}
    return nano, vllm


# ---------------------------------------------------------------- one run's numbers and checks

def token_digest(token_lists: list[list[int]]) -> str:
    """sha256 of a run's tokens, request by request in panel order."""
    return hashlib.sha256(json.dumps(token_lists, separators=(",", ":")).encode()).hexdigest()


def nano_tokens(result: dict) -> list[list[int]]:
    return [request["token_ids"] for request in result["requests"]]


def vllm_tokens(result: dict) -> list[list[int]]:
    return [record["token_ids"] for record in result["runs"][0]["request_records"]]


def nano_q(result: dict) -> float:
    """Wall-clock Q_full: every token the measured requests produced over the measured call's wall clock."""
    return sum(len(tokens) for tokens in nano_tokens(result)) / result["wall_seconds"]


def nano_measured(result: dict) -> tuple[int, float]:
    return sum(len(tokens) for tokens in nano_tokens(result)), result["wall_seconds"]


def vllm_measured(result: dict) -> tuple[int, float]:
    run = result["runs"][0]
    return run["output_tokens"], run["seconds"]


def expected_vllm_hit(drafting: bool, shared_tokens: int, cache: str, block_size: int) -> int:
    """Each measured request's cached tokens in vLLM 372477d2: the shared part with the cache on, one block less for a
    drafter that reads the target's hidden states (use_eagle_block_drop, on by default for the dspark method)."""
    if cache == "off":
        return 0
    return max(0, shared_tokens - block_size) if drafting else shared_tokens


def vllm_invalid(result: dict, protocol: dict, panel: str, cache: str, base: str, concurrency: int, seed: int,
                 repeat: int) -> list[str]:
    """invalid_if.both for one vLLM run: what reruns its group."""
    reasons = []
    run = result["runs"][0]
    if (result.get("panel"), result.get("prefix_cache"), result["line"], result["concurrency"], result["engine"]["seed"],
            result.get("group_repeat")) != (panel, cache == "on", base, concurrency, seed, repeat):
        reasons.append("not the run its name says")
    if result["gpu"] != protocol["gpu"]["name"]:
        reasons.append("gpu name mismatch")
    if run["preemptions"] is None:
        reasons.append("no preemption count")
    elif run["preemptions"] > 0:
        reasons.append("preemption > 0")
    if run["requests"] != requests_of(concurrency) or len(run["request_records"]) != requests_of(concurrency):
        reasons.append(f"{run['requests']} requests, not {requests_of(concurrency)}")
    if not (isinstance(run["tokens_per_second"], (int, float)) and math.isfinite(run["tokens_per_second"])
            and run["tokens_per_second"] > 0):
        reasons.append("non-finite value or assertion failure")
    return reasons


def nano_invalid(result: dict, panel: str, cache: str, concurrency: int, seed: int, repeat: int) -> list[str]:
    """invalid_if.both for one nano run beyond report_v2's count_runs: the run its name says, its cache, its request
    count and a positive throughput."""
    config, reasons = result["config"], []
    if (config["panel"], config["concurrency"], config["seed"], config["repeat"], bool(config.get("enable_prefix_cache"))) \
            != (panel, concurrency, seed, repeat, cache == "on"):
        reasons.append("not the run its name says")
    if len(result["requests"]) != requests_of(concurrency):
        reasons.append(f"{len(result['requests'])} requests, not {requests_of(concurrency)}")
    if not (result["wall_seconds"] > 0 and math.isfinite(nano_q(result)) and nano_q(result) > 0):
        reasons.append("non-finite value or assertion failure")
    return reasons


def vllm_hits(result: dict, protocol: dict, panel: str, cache: str) -> list[str]:
    """hits.vllm for one valid vLLM run: what stops the job."""
    block_size = result["engine_cache"]["block_size"]
    if not isinstance(block_size, int) or block_size <= 0:
        return [f"hits: no block size recorded ({block_size})"]
    drafting = result["engine"].get("speculative_config") is not None
    expected = expected_vllm_hit(drafting, protocol["panels"][panel]["shared_tokens"], cache, block_size)
    hits = sorted({record["cached_tokens"] for record in result["runs"][0]["request_records"]}, key=str)
    return [] if hits == [expected] else [f"hits: {hits}, expected {expected} for every request"]


def reference_key(panel: str, line: str, concurrency: int, seed: int) -> str:
    return f"{panel}/{line}-C{concurrency}-s{seed}"


def nano_identity_and_hits(result: dict, protocol: dict, panel: str, cache: str, reference: dict) -> list[str]:
    """identity and hits.nano for one valid nano run: what stops the job."""
    config = result["config"]
    stops = [f"hits: {problem}" for problem in hits_check(result, protocol["panels"][panel]["shared_tokens"], cache == "on")]
    key = reference_key(panel, config["line"], config["concurrency"], config["seed"])
    if key not in reference["runs"]:
        stops.append(f"identity: no reference run {key}")
    elif token_digest(nano_tokens(result)) != reference["runs"][key]["digest"]:
        stops.append(f"identity: the reproduction failed, tokens differ from M6 ①'s {key}")
    return stops


# ---------------------------------------------------------------- loading a job's runs

def load_vllm(directories: list[Path]) -> list[tuple[str, dict]]:
    """(path, result) of every vLLM result of samehost_prefix, in the order of the directories."""
    found = []
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            result = json.loads(path.read_text())
            if "engine_cache" in result and "group_repeat" in result:
                found.append((str(path), result))
    return found


def round_of(name: str) -> str:
    """OUT/main or OUT/rerun: the round of the job a result file belongs to."""
    return str(Path(name).parent.parent)


SLOW_STEP = "decode step or round > 3 x median"    # report_v2.invalid_reasons' wording of v2.6's step rule


def analyse(protocol: dict, panel: str, runs: list[tuple[str, dict, dict]], vllm_runs: list[tuple[str, dict]],
            reference: dict, originals: bool = False) -> dict:
    """runs: report_v2.load_runs of nano's directories; vllm_runs: load_vllm of vLLM's. A later run with the same
    (side, cache, line, C, seed, repeat) replaces an earlier one; with originals the earliest stays and the slow-step
    rule is set aside, which is what the verdicts would be without nano's screening (V4)."""
    job, spec, seeds = job_of(protocol, panel), section(protocol), seeds_of(protocol)
    invalid, _, _ = count_runs(runs, protocol)
    by_line = {nano_line(base, cache): (cache, base) for cache in job["cache"] for base, _ in spec["pairs"]}
    pair_of = {vllm: nano for nano, vllm in spec["pairs"]}

    chosen, replaced = {}, defaultdict(list)    # (side, cache, nano base, C, seed, repeat) -> (name, result)
    for name, result, _ in runs:
        config = result["config"]
        if config["panel"] != panel or config["line"] not in by_line:
            continue
        cache, base = by_line[config["line"]]
        key = ("nano", cache, base, config["concurrency"], config["seed"], config["repeat"])
        if key in chosen:
            if originals:
                continue
            replaced[key].append(chosen[key])
        chosen[key] = (name, result)
    for name, result in vllm_runs:
        if result.get("panel") != panel or result["line"] not in pair_of:
            continue
        cache = "on" if result["prefix_cache"] else "off"
        key = ("vllm", cache, pair_of[result["line"]], result["concurrency"], result["engine"]["seed"], result["group_repeat"])
        if key in chosen:
            if originals:
                continue
            replaced[key].append(chosen[key])
        chosen[key] = (name, result)

    rounds = defaultdict(set)    # one (C, seed) must come from one round of the job on both sides
    for (_, _, _, concurrency, seed, _), (name, _) in chosen.items():
        rounds[(concurrency, seed)].add(round_of(name))
    split = {key for key, found in rounds.items() if len(found) > 1}

    # check_order: a group counts only whole and valid, both sides and both repeats; only then identity and hits
    problems, stops, valid = [], [], {}
    for concurrency in spec["concurrency"]:
        for seed in seeds:
            nano_names, vllm_names = expected_runs(protocol, panel, concurrency, seed)
            keys = [("nano", *key[:2], concurrency, seed, key[2]) for key in nano_names] + \
                   [("vllm", key[0], pair_of[key[1]], concurrency, seed, key[2]) for key in vllm_names]
            group_bad = False
            for key in keys:
                side, cache, base, _, _, repeat = key
                if key not in chosen:
                    problems.append(f"{side} {cache} {base} C={concurrency} seed {seed} repeat {repeat}: missing")
                    group_bad = True
                    continue
                name, result = chosen[key]
                if side == "nano":
                    reasons = [reason for reason in invalid[name] if not (originals and reason == SLOW_STEP)]
                    reasons += nano_invalid(result, panel, cache, concurrency, seed, repeat)
                else:
                    reasons = vllm_invalid(result, protocol, panel, cache, dict(spec["pairs"])[base], concurrency, seed,
                                           repeat)
                if (concurrency, seed) in split:
                    reasons.append("lines of one (C, seed) split across rounds of the job")
                problems += [f"{side} {Path(name).name}: {reason}" for reason in reasons]
                group_bad |= bool(reasons)
            if group_bad:
                continue
            group = {}
            for key in keys:
                name, result = chosen[key]
                found = (nano_identity_and_hits(result, protocol, panel, key[1], reference) if key[0] == "nano"
                         else vllm_hits(result, protocol, panel, key[1]))
                stops += [f"{key[0]} {Path(name).name}: {stop}" for stop in found]
                group[key] = None if found else result
            if all(result is not None for result in group.values()):
                valid.update(group)

    def q_of(side: str, result: dict) -> float:
        return nano_q(result) if side == "nano" else result["runs"][0]["tokens_per_second"]

    # the repeated groups: each side's Q twice, and B and Gamma of each repeat for N1 and N2
    noise_seed = spec["timing_noise"]["seed"]
    repeat_pairs, b_pairs, gamma_pairs = [], [], []
    for cache in job["cache"]:
        for base, _ in spec["pairs"]:
            for concurrency in spec["concurrency"]:
                q = {(side, repeat): q_of(side, valid[(side, cache, base, concurrency, noise_seed, repeat)])
                     for side in ("nano", "vllm") for repeat in (1, 2)
                     if (side, cache, base, concurrency, noise_seed, repeat) in valid}
                for side in ("nano", "vllm"):
                    found = [valid.get((side, cache, base, concurrency, noise_seed, repeat)) for repeat in (1, 2)]
                    tokens = nano_tokens if side == "nano" else vllm_tokens
                    repeat_pairs.append(dict(side=side, cache=cache, line=base, concurrency=concurrency,
                                             q1=q.get((side, 1)), q2=q.get((side, 2)),
                                             same_tokens=tokens(found[0]) == tokens(found[1]) if all(found) else None))
                b_pairs.append(dict(cache=cache, line=base, concurrency=concurrency,
                                    b=[q[("nano", k)] / q[("vllm", k)] if ("nano", k) in q and ("vllm", k) in q else None
                                       for k in (1, 2)]))
    if set(job["cache"]) == set(CACHES):
        for base, _ in spec["pairs"]:
            for concurrency in spec["concurrency"]:
                gammas = []
                for k in (1, 2):
                    keys = [(side, cache, base, concurrency, noise_seed, k) for side in ("nano", "vllm") for cache in CACHES]
                    if all(key in valid for key in keys):
                        q = {key[:2]: q_of(key[0], valid[key]) for key in keys}
                        gammas.append((q[("nano", "on")] / q[("nano", "off")]) / (q[("vllm", "on")] / q[("vllm", "off")]))
                    else:
                        gammas.append(None)
                gamma_pairs.append(dict(line=base, concurrency=concurrency, gamma=gammas))

    def largest(pairs, field):
        values = [pair[field] for pair in pairs]
        return max(abs(a / b - 1) for a, b in values) if values and all(a and b for a, b in values) else None

    n1 = largest(b_pairs, "b")
    n2 = largest(gamma_pairs, "gamma") if gamma_pairs else None

    def q_by_seed(side, cache, base, concurrency):
        return {seed: q_of(side, valid[(side, cache, base, concurrency, seed, 1)]) for seed in seeds
                if (side, cache, base, concurrency, seed, 1) in valid}

    gaps = []
    for cache in job["cache"]:
        for base, vllm_base in spec["pairs"]:
            for concurrency in spec["concurrency"]:
                ours, theirs = q_by_seed("nano", cache, base, concurrency), q_by_seed("vllm", cache, base, concurrency)
                ratio = statistics.median(ours.values()) / statistics.median(theirs.values()) if ours and theirs else None
                extra = {side: per_seed_extra(side, {seed: valid[(side, cache, base, concurrency, seed, 1)] for seed in
                                                     (ours if side == "nano" else theirs)}, panel, reference)
                         for side in ("nano", "vllm")}
                gaps.append(dict(cache=cache, nano_line=nano_line(base, cache), vllm_line=vllm_name(vllm_base, cache),
                                 concurrency=concurrency, nano=ours, vllm=theirs, ratio=ratio,
                                 seed_ratios={seed: ours[seed] / theirs[seed] for seed in ours if seed in theirs},
                                 verdict=verdict(ours, theirs, seeds, n1, "nano faster", "nano slower"),
                                 nano_extra=extra["nano"], vllm_extra=extra["vllm"]))

    gains = []
    if set(job["cache"]) == set(CACHES):
        for base, vllm_base in spec["pairs"]:
            for concurrency in spec["concurrency"]:
                one_plus_g, tokens_ratio, time_ratio = {}, {}, {}
                for side in ("nano", "vllm"):
                    measured = nano_measured if side == "nano" else vllm_measured
                    one_plus_g[side], tokens_ratio[side], time_ratio[side] = {}, {}, {}
                    for seed in seeds:
                        on, off = (valid.get((side, cache, base, concurrency, seed, 1)) for cache in ("on", "off"))
                        if on is None or off is None:
                            continue
                        one_plus_g[side][seed] = q_of(side, on) / q_of(side, off)
                        (tokens_on, seconds_on), (tokens_off, seconds_off) = measured(on), measured(off)
                        tokens_ratio[side][seed], time_ratio[side][seed] = tokens_on / tokens_off, seconds_on / seconds_off
                ratio_on, ratio_off = (next(gap["ratio"] for gap in gaps if gap["cache"] == cache and
                                            gap["nano_line"] == nano_line(base, cache) and gap["concurrency"] == concurrency)
                                       for cache in ("on", "off"))
                gains.append(dict(nano_line=base, vllm_line=vllm_base, concurrency=concurrency, one_plus_g=one_plus_g,
                                  tokens_ratio=tokens_ratio, time_ratio=time_ratio,
                                  gap_on_over_off=ratio_on / ratio_off if ratio_on and ratio_off else None,
                                  verdict=verdict(one_plus_g["nano"], one_plus_g["vllm"], seeds, n2,
                                                  "nano's gain larger", "vLLM's gain larger")))

    screening = []    # each nano run voided for a slow step, against the run that replaced it
    for key, earlier in replaced.items():
        if key[0] != "nano" or key not in valid:
            continue
        for name, result in earlier:
            if SLOW_STEP in invalid[name]:
                screening.append(dict(run=Path(name).name, q_voided=nano_q(result), q_rerun=nano_q(valid[key])))

    packages = sorted({result["manifest"]["packages_sha256"] for (side, *_), result in valid.items() if side == "nano"})
    nodes = sorted({result["manifest"]["node"] for (side, *_), result in valid.items() if side == "nano"} |
                   {result["manifest"]["node"] for (side, *_), result in valid.items() if side == "vllm"})
    return dict(panel=panel, caches=job["cache"], seeds=seeds, n1=n1, n2=n2, repeat_pairs=repeat_pairs, b_pairs=b_pairs,
                gamma_pairs=gamma_pairs, nodes=nodes, gaps=gaps,
                gains=gains, screening=screening, stops=stops, invalid=problems, nano_packages_sha256=packages,
                reference_packages_sha256=sorted({run.get("packages_sha256") for run in reference["runs"].values()} - {None}))


def per_seed_extra(side: str, results: dict, panel: str, reference: dict) -> dict:
    """V4 by seed: tokens per round (round-weighted), the output length's median; nano's environment reproduction
    deviation; vLLM's cached tokens, block size, KV tokens and memory in use."""
    extra = {}
    for seed, result in results.items():
        if side == "nano":
            stats, config = result["speculative_stats"], result["config"]
            ref = reference["runs"].get(reference_key(panel, config["line"], config["concurrency"], seed), {})
            extra[seed] = dict(tokens_per_round=round_weighted_tokens(stats) if stats else None,
                               output_median=statistics.median(len(t) for t in nano_tokens(result)),
                               reproduction=nano_q(result) / ref["q_wall"] if ref.get("q_wall") else None)
        else:
            run = result["runs"][0]
            extra[seed] = dict(tokens_per_round=run["tokens_per_round"],
                               output_median=statistics.median(len(t) for t in vllm_tokens(result)),
                               cached_tokens=sorted({record["cached_tokens"] for record in run["request_records"]}, key=str),
                               block_size=result["engine_cache"]["block_size"],
                               kv_cache_size_tokens=result["engine_cache"]["kv_cache_size_tokens"],
                               gpu_memory_used_mib=run["gpu_memory_used_mib"])
    return extra


def verdict(ours: dict, theirs: dict, seeds: list[int], noise: float | None, above: str, below: str) -> str:
    """V1 / V2: with every seed on both sides and the job's N, `above` when all of ours exceed all of theirs and the
    medians' ratio exceeds 1 + 2N, `below` when the reverse; else undecided; short of seeds or N, incomplete."""
    if set(ours) != set(seeds) or set(theirs) != set(seeds) or noise is None:
        return "incomplete"
    ratio = statistics.median(ours.values()) / statistics.median(theirs.values())
    if min(ours.values()) > max(theirs.values()) and ratio > 1 + 2 * noise:
        return above
    if max(ours.values()) < min(theirs.values()) and ratio < 1 - 2 * noise:
        return below
    return "undecided"


# ---------------------------------------------------------------- markdown

def spread(values, scale: float = 1, digits: int = 1) -> str:
    values = [value * scale for value in values if value is not None]
    if not values:
        return "-"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def markdown(analysis: dict) -> str:
    def percent(value):
        return f"{value * 100:.2f}%" if value is not None else "incomplete"

    out = [f"# {analysis['panel']}: nano and vLLM, the prefix cache {' and '.join(analysis['caches'])}", "",
           f"N1 = {percent(analysis['n1'])} over {len(analysis['b_pairs'])} repeated (cache, line, C); "
           f"N2 = {percent(analysis['n2'])} over {len(analysis['gamma_pairs'])} (line, C). Wall-clock Q_full, tok/s, "
           "median (min–max) over the seeds; nano / vLLM is B, the ratio of the medians.", "",
           "| cache | nano line | vLLM line | C | nano | vLLM | nano / vLLM | V1 | per-seed ratio | tokens/round nano | vLLM |",
           "|---|---|---|---:|---|---|---:|---|---|---|---|"]
    for gap in analysis["gaps"]:
        tpr = [[row["tokens_per_round"] for row in gap[side].values()] for side in ("nano_extra", "vllm_extra")]
        ratio = f"{gap['ratio']:.3f}" if gap["ratio"] is not None else "-"
        out.append(f"| {gap['cache']} | {gap['nano_line']} | {gap['vllm_line']} | {gap['concurrency']} "
                   f"| {spread(gap['nano'].values())} | {spread(gap['vllm'].values())} | {ratio} | {gap['verdict']} "
                   f"| {spread(gap['seed_ratios'].values(), digits=3)} | {spread(tpr[0], digits=2)} | {spread(tpr[1], digits=2)} |")
    if analysis["gains"]:
        out += ["", "1 + G = Q(on) / Q(off) by seed within one engine, as G %; output tokens and measured time on / off.", "",
                "| line | C | G nano | G vLLM | V2 | gap on / off | tokens nano | vLLM | time nano | vLLM |",
                "|---|---:|---|---|---|---:|---|---|---|---|"]
        for gain in analysis["gains"]:
            g = {side: [value - 1 for value in gain["one_plus_g"][side].values()] for side in ("nano", "vllm")}
            quotient = f"{gain['gap_on_over_off']:.3f}" if gain["gap_on_over_off"] else "-"
            out.append(f"| {gain['nano_line']} | {gain['concurrency']} | {spread(g['nano'], 100)} | {spread(g['vllm'], 100)} "
                       f"| {gain['verdict']} | {quotient} | {spread(gain['tokens_ratio']['nano'].values(), digits=3)} "
                       f"| {spread(gain['tokens_ratio']['vllm'].values(), digits=3)} "
                       f"| {spread(gain['time_ratio']['nano'].values(), digits=3)} | {spread(gain['time_ratio']['vllm'].values(), digits=3)} |")
    out += ["", "Repeat pairs (seed 17011): |Q1 / Q2 - 1| %, and whether the two gave the same tokens.", "",
            "| side | cache | line | C | difference | same tokens |", "|---|---|---|---:|---:|---|"]
    for pair in analysis["repeat_pairs"]:
        difference = f"{abs(pair['q1'] / pair['q2'] - 1) * 100:.2f}" if pair["q1"] and pair["q2"] else "-"
        out.append(f"| {pair['side']} | {pair['cache']} | {pair['line']} | {pair['concurrency']} | {difference} | {pair['same_tokens']} |")
    deviation = [row["reproduction"] for gap in analysis["gaps"] for row in gap["nano_extra"].values()]
    vllm_rows = [row for gap in analysis["gaps"] for row in gap["vllm_extra"].values()]
    out += ["", f"Environment reproduction deviation, nano's wall-clock Q over M6 ①'s run of the same name (same tokens): "
                f"{spread(deviation, digits=4)}. nano packages_sha256 {analysis['nano_packages_sha256']}, M6 ①'s "
                f"{analysis['reference_packages_sha256']}.",
            f"vLLM: cached tokens {sorted({str(row['cached_tokens']) for row in vllm_rows})}, block size "
            f"{sorted({row['block_size'] for row in vllm_rows})}, KV tokens {sorted({row['kv_cache_size_tokens'] for row in vllm_rows}, key=str)}, "
            f"memory in use {spread([row['gpu_memory_used_mib'] for row in vllm_rows], digits=0)} MiB."]
    if analysis["screening"]:
        out += ["", "nano runs voided for a slow step, against their reruns (wall-clock Q):", "",
                *(f"- {row['run']}: {row['q_voided']:.1f} voided, {row['q_rerun']:.1f} rerun" for row in analysis["screening"])]
    if analysis.get("without_screening"):
        changed = analysis["without_screening"]
        out += ["", f"With every rerun group's original runs in place of its rerun and the slow-step rule set aside: N1 "
                    f"{percent(changed['n1'])}, N2 {percent(changed['n2'])}; verdicts that change: "
                    f"{', '.join(changed['changed']) or 'none'}."]
    if analysis["stops"]:
        out += ["", "STOP (identity or hits):", "", *(f"- {line}" for line in analysis["stops"])]
    if analysis["invalid"]:
        out += ["", "Invalid, their whole (C, seed) rerun:", "", *(f"- {line}" for line in analysis["invalid"])]
    return "\n".join(out)


def without_screening(final: dict, originals: dict) -> dict:
    """V4's effect of the slow-step screening: N1, N2 and the cells whose verdict differs on the original runs."""
    def cells(analysis):
        return {**{f"V1 {gap['cache']} {gap['nano_line']} C={gap['concurrency']}": gap["verdict"] for gap in analysis["gaps"]},
                **{f"V2 {gain['nano_line']} C={gain['concurrency']}": gain["verdict"] for gain in analysis["gains"]}}
    before, after = cells(final), cells(originals)
    return dict(n1=originals["n1"], n2=originals["n2"],
                changed=[f"{cell}: {before[cell]} -> {after[cell]}" for cell in before if before[cell] != after.get(cell)])


# ---------------------------------------------------------------- V3 over the jobs

def combine(protocol: dict, reports: list[dict]) -> str:
    """V3: B with the cache on at each P; at the two ends B(S2048, on) / B(S0, on) = [B(S2048, off) / B(S0, off)] x
    {[B(S2048, on) / B(S2048, off)] / [B(S0, on) / B(S0, off)]}, the longer prefix with the cache off and the change of
    the cache's effect on B. Report only."""
    by_panel = {report["panel"]: report for report in reports}
    nodes = {panel: report["nodes"] for panel, report in by_panel.items()}
    b = {(panel, gap["cache"], gap["nano_line"].removesuffix("_pc"), gap["concurrency"]): gap["ratio"]
         for panel, report in by_panel.items() for gap in report["gaps"]}
    shared = {panel: protocol["panels"][panel]["shared_tokens"] for panel in by_panel}
    panels = sorted(by_panel, key=shared.get)
    out = ["| line | C | " + " | ".join(f"B on, P = {shared[panel]}" for panel in panels) +
           " | S2048 on / S0 on | longer prefix, off | cache's effect, S2048 over S0 |",
           "|---|---:|" + "---:|" * (len(panels) + 3)]
    for base, _ in section(protocol)["pairs"]:
        for concurrency in section(protocol)["concurrency"]:
            def at(panel, cache):
                return b.get((panel, cache, base, concurrency))
            cells = [at(panel, "on") for panel in panels]
            ends = ratio_of(at("S2048", "on"), at("S0", "on"))
            longer = ratio_of(at("S2048", "off"), at("S0", "off"))
            effect = ratio_of(ratio_of(at("S2048", "on"), at("S2048", "off")), ratio_of(at("S0", "on"), at("S0", "off")))
            out.append(f"| {base} | {concurrency} | " + " | ".join(f"{cell:.3f}" if cell else "-" for cell in cells) +
                       f" | {fmt(ends)} | {fmt(longer)} | {fmt(effect)} |")
    machines = {tuple(found) for found in nodes.values()}
    out += ["", "Nodes: " + "; ".join(f"{panel} {', '.join(found)}" for panel, found in sorted(nodes.items())) +
            (". The panels ran on different machines: V3 compares across VMs." if len(machines) > 1 else ". One machine.")]
    return "\n".join(out)


def ratio_of(a: float | None, b: float | None) -> float | None:
    return a / b if a and b else None


def fmt(value: float | None) -> str:
    return f"{value:.3f}" if value else "-"


# ---------------------------------------------------------------- the reference digests

def reference(protocol: dict, results: dict[str, list[Path]]) -> dict:
    """Every nano run of M6 ① the samehost jobs repeat: its tokens' digest, wall-clock Q_full and packages_sha256, from
    its final run (a rerun replaces the original; the two must have the same tokens)."""
    spec = section(protocol)
    wanted = {(job["panel"], nano_line(base, cache)) for job in spec["jobs"] for cache in job["cache"]
              for base, _ in spec["pairs"]}
    entries = {}
    for panel, directories in results.items():
        by_key = defaultdict(list)
        for name, result, _ in load_runs(directories):
            config = result["config"]
            if config["panel"] == panel and config["repeat"] == 1 and (panel, config["line"]) in wanted \
                    and config["concurrency"] in spec["concurrency"]:
                by_key[reference_key(panel, config["line"], config["concurrency"], config["seed"])].append((name, result))
        for key, found in by_key.items():
            digests = {token_digest(nano_tokens(result)) for _, result in found}
            if len(digests) != 1:
                raise ValueError(f"{key}: the original and its rerun differ in tokens")
            name, result = found[-1]
            entries[key] = dict(digest=digests.pop(), q_wall=nano_q(result),
                                packages_sha256=result["manifest"]["packages_sha256"], source=name)
    expected = {reference_key(panel, line, concurrency, seed) for panel, line in wanted
                for concurrency in spec["concurrency"] for seed in seeds_of(protocol)}
    if set(entries) != expected:
        raise ValueError(f"missing reference runs: {sorted(expected - set(entries))[:5]}")
    return dict(schema="m6-samehost-reference", runs=dict(sorted(entries.items())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["report", "combine", "reference"])
    parser.add_argument("reports", nargs="*", type=Path, help="report JSON files (combine)")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel")
    parser.add_argument("--nano", type=Path, nargs="+")
    parser.add_argument("--vllm", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--results", nargs="+", help="PANEL=DIR[,DIR] (reference)")
    parser.add_argument("--out", type=Path, help="the reference file (reference)")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    if args.command == "reference":
        results = {item.split("=", 1)[0]: [Path(path) for path in item.split("=", 1)[1].split(",")] for item in args.results}
        args.out.write_text(json.dumps(reference(protocol, results), indent=1) + "\n")
        return
    if args.command == "combine":
        print(combine(protocol, [json.loads(path.read_text()) for path in args.reports]))
        return
    runs, vllm_runs, reference_runs = load_runs(args.nano), load_vllm(args.vllm), json.loads(args.reference.read_text())
    analysis = analyse(protocol, args.panel, runs, vllm_runs, reference_runs)
    if analysis["screening"] or any("rerun" in round_of(name) for name, _ in vllm_runs):
        analysis["without_screening"] = without_screening(analysis, analyse(protocol, args.panel, runs, vllm_runs,
                                                                            reference_runs, originals=True))
    print(markdown(analysis))
    if args.json:
        args.json.write_text(json.dumps(analysis, indent=1) + "\n")


if __name__ == "__main__":
    main()
