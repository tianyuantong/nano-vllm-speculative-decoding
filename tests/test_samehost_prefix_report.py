import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from samehost_prefix_report import (analyse, combine, expected_runs, expected_vllm_hit, markdown,  # noqa: E402
                                    token_digest, verdict, vllm_hits, vllm_invalid, without_screening)

GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
SEEDS = [1, 2, 3, 4, 5]
PROTOCOL = {
    "gpu": {"name": GPU},
    "seeds": {"dev": SEEDS},
    "lines": {"k0": {"tree": "main", "k": 0}, "k0_pc": {"tree": "main", "k": 0, "prefix_cache": True},
              "dspark_k7": {"tree": "main", "k": 7}, "dspark_k7_pc": {"tree": "main", "k": 7, "prefix_cache": True}},
    "panels": {"S2048": {"shared_tokens": 2048}, "S0": {"shared_tokens": 0}},
    "samehost_prefix": {"jobs": [{"panel": "S2048", "cache": ["on", "off"]}, {"panel": "S0", "cache": ["on", "off"]}],
                        "pairs": [["k0", "k0_v2"], ["dspark_k7", "dspark_k7"]], "concurrency": [8], "seeds": "dev",
                        "timing_noise": {"seed": 1}},
}
LINES = {"k0": "k0_v2", "dspark_k7": "dspark_k7"}


def nano_run(line, seed, wall, repeat=1, directory="OUT/main/nano", spike=1.0, panel="S2048"):
    on = line.endswith("_pc")
    shared = PROTOCOL["panels"][panel]["shared_tokens"] if on else 0
    requests = [{"token_ids": [seed, index], "deliveries": [[0, 1], [1, 1]], "prompt_len": 3000, "cached_tokens": shared}
                for index in range(48)]
    kind = "round" if "dspark" in line else "decode"
    steps = [{"kind": "prefill", "duration_ms": 50.0, "prefill_tokens": 48 * (3000 - shared)}] + \
            [{"kind": kind, "duration_ms": 10.0, "prefill_tokens": 0}] * 5 + \
            [{"kind": kind, "duration_ms": 10.0 * spike, "prefill_tokens": 0}]
    result = {"gpu_state_start": {"name": GPU}, "steps": steps, "wall_seconds": wall, "speculative_stats": None,
              "requests": requests, "protocol_sha256": "p", "manifest": {"commit": "a", "panel_sha256": panel, "packages_sha256": "pk"},
              "config": {"panel": panel, "concurrency": 8, "line": line, "seed": seed, "repeat": repeat}}
    return f"{directory}/{line}-C8-s{seed}-r{repeat}.json", result, {"preemptions": 0, "q_steady": 1.0, "q_full": 1.0}


def vllm_run(line, seed, q, on, repeat=1, directory="OUT/main/vllm", hit=None, preemptions=0, panel="S2048"):
    drafting = line != "k0_v2"
    shared = PROTOCOL["panels"][panel]["shared_tokens"]
    expected = expected_vllm_hit(drafting, shared, "on" if on else "off", 16)
    records = [{"prompt_len": 3000, "cached_tokens": expected if hit is None else hit, "token_ids": [1] * 10,
                "finish_reason": "length"}] * 48
    result = {"line": line, "concurrency": 8, "prefix_cache": on, "panel": panel, "gpu": GPU, "group_repeat": repeat,
              "engine": {"seed": seed, **({"speculative_config": {"method": "dspark"}} if drafting else {})},
              "engine_cache": {"block_size": 16, "kv_cache_size_tokens": 306_888},
              "runs": [{"requests": 48, "tokens_per_second": q, "preemptions": preemptions, "output_tokens": 480,
                        "seconds": 480 / q, "tokens_per_round": 2.8 if drafting else None, "gpu_memory_used_mib": 70_000,
                        "request_records": records}]}
    return f"{directory}/{line}{'_pc' if on else ''}-C8-s{seed}-r{repeat}.json", result


def reference_of(runs):
    return {"runs": {f"{result['config']['panel']}/{result['config']['line']}-C8-s{result['config']['seed']}":
                     {"digest": token_digest([r["token_ids"] for r in result["requests"]]), "q_wall": 1.0,
                      "packages_sha256": "pk"}
                     for _, result, _ in runs if result["config"]["repeat"] == 1}}


def job(nano_on=10.0, nano_off=12.0, vllm_on=100.0, vllm_off=80.0, panel="S2048"):
    """Every run of one job; nano's wall clock in seconds (96 tokens a run), vLLM's throughput; seed 1 twice."""
    nano, vllm = [], []
    for seed in SEEDS:
        for repeat in ((1, 2) if seed == 1 else (1,)):
            for base in LINES:
                nano += [nano_run(f"{base}_pc", seed, nano_on + seed * 0.01, repeat, panel=panel),
                         nano_run(base, seed, nano_off + seed * 0.01, repeat, panel=panel)]
                vllm += [vllm_run(LINES[base], seed, vllm_on + seed * 0.1, True, repeat, panel=panel),
                         vllm_run(LINES[base], seed, vllm_off + seed * 0.1, False, repeat, panel=panel)]
    return nano, vllm


def test_vllm_hits_one_block_less_on_a_drafting_line():
    assert expected_vllm_hit(False, 2048, "on", 16) == 2048
    assert expected_vllm_hit(True, 2048, "on", 16) == 2032
    assert expected_vllm_hit(True, 0, "on", 16) == 0 and expected_vllm_hit(True, 2048, "off", 16) == 0


def test_a_group_at_the_timing_noise_seed_holds_both_repeats():
    nano, vllm = expected_runs(PROTOCOL, "S2048", 8, 1)
    assert len(nano) == len(vllm) == 8 and nano[("on", "k0", 2)] == "k0_pc-C8-s1-r2.json"
    assert vllm[("off", "k0_v2", 1)] == "k0_v2-C8-s1-r1.json"
    assert len(expected_runs(PROTOCOL, "S2048", 8, 2)[0]) == 4


def test_a_verdict_needs_every_seed_no_overlap_and_more_than_twice_the_noise():
    ours, theirs = {seed: 1.0 + seed * 0.001 for seed in SEEDS}, {seed: 1.01 + seed * 0.001 for seed in SEEDS}
    assert verdict(ours, theirs, SEEDS, 0.001, "faster", "slower") == "slower"
    assert verdict(ours, theirs, SEEDS, 0.01, "faster", "slower") == "undecided"    # 0.99 is within 1 - 2N
    assert verdict(theirs, ours, SEEDS, 0.001, "faster", "slower") == "faster"
    assert verdict({**ours, 5: 1.5}, theirs, SEEDS, 0.001, "faster", "slower") == "undecided"
    assert verdict(ours, theirs, SEEDS, None, "faster", "slower") == "incomplete"


def test_a_missing_preemption_count_invalidates_and_wrong_hits_stop():
    _, result = vllm_run("k0_v2", 2, 100.0, True, preemptions=None)
    assert vllm_invalid(result, PROTOCOL, "S2048", "on", "k0_v2", 8, 2, 1) == ["no preemption count"]
    assert vllm_invalid(result, PROTOCOL, "S2048", "on", "k0_v2", 8, 2, 2) == ["not the run its name says", "no preemption count"]
    _, result = vllm_run("dspark_k7", 2, 100.0, True, hit=2048)
    assert vllm_hits(result, PROTOCOL, "S2048", "on") == ["hits: [2048], expected 2032 for every request"]


def test_the_job_reports_the_gap_each_engines_gain_and_the_noise():
    nano, vllm = job()
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    assert not analysis["stops"] and not analysis["invalid"] and analysis["n1"] == 0 and analysis["n2"] == 0
    on = next(gap for gap in analysis["gaps"] if gap["cache"] == "on" and gap["vllm_line"] == "dspark_k7_pc")
    assert on["verdict"] == "nano slower" and abs(on["ratio"] - 96 / 10.03 / 100.3) < 1e-9
    gain = next(gain for gain in analysis["gains"] if gain["nano_line"] == "k0")
    assert gain["verdict"] == "vLLM's gain larger"            # nano 12/10 - 1 = 20 %, vLLM 100/80 - 1 = 25 %
    assert gain["tokens_ratio"]["nano"][3] == 1 and abs(gain["time_ratio"]["nano"][3] - 10.03 / 12.03) < 1e-12
    assert "| on | k0_pc | k0_v2_pc | 8 |" in markdown(analysis)


def test_without_its_repeats_a_job_gives_no_verdict():
    nano, vllm = job()
    vllm = [(name, result) for name, result in vllm if not name.endswith("-r2.json")]
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    assert analysis["n1"] is None and {gap["verdict"] for gap in analysis["gaps"]} == {"incomplete"}


def test_nano_tokens_other_than_m6_1s_stop_the_job():
    nano, vllm = job()
    reference = reference_of(nano)
    reference["runs"]["S2048/k0_pc-C8-s3"]["digest"] = "0" * 64
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference)
    assert analysis["stops"] == ["nano k0_pc-C8-s3-r1.json: identity: the reproduction failed, tokens differ from "
                                 "M6 ①'s S2048/k0_pc-C8-s3"]
    gap = next(gap for gap in analysis["gaps"] if gap["nano_line"] == "k0_pc")
    assert set(gap["nano"]) == {1, 2, 4, 5} and gap["verdict"] == "incomplete"


def test_a_rerun_round_replaces_its_group_and_the_screening_is_reported():
    nano, vllm = job()
    nano = [run if not run[0].endswith("k0_pc-C8-s2-r1.json") else nano_run("k0_pc", 2, 10.02, spike=4.0) for run in nano]
    nano += [nano_run(line, 2, 10.5, directory="OUT/rerun/nano") for line in ("k0_pc", "k0", "dspark_k7_pc", "dspark_k7")]
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    assert any("split across rounds" in line for line in analysis["invalid"])    # vLLM's seed 2 still from main
    vllm += [vllm_run(LINES[base], 2, q, on, directory="OUT/rerun/vllm") for base in LINES
             for q, on in ((100.0, True), (80.0, False))]
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    assert not analysis["invalid"] and not analysis["stops"]
    assert analysis["screening"] == [{"run": "k0_pc-C8-s2-r1.json", "q_voided": 96 / 10.02, "q_rerun": 96 / 10.5}]


def test_combine_splits_the_ends_into_the_longer_prefix_and_the_cache():
    reports = []
    for panel, (nano_on, nano_off) in (("S0", (10.0, 10.0)), ("S2048", (10.0, 12.0))):
        nano, vllm = job(nano_on, nano_off, panel=panel)
        reports.append(analyse(PROTOCOL, panel, nano, vllm, reference_of(nano)))
    table = combine(PROTOCOL, reports)
    assert table.splitlines()[0].startswith("| line | C | B on, P = 0 | B on, P = 2048 | S2048 on / S0 on |")
    # B(S2048, on) / B(S0, on) = [B(S2048, off) / B(S0, off)] x the change of the cache's effect
    row = next(line for line in table.splitlines() if line.startswith("| k0 | 8 |")).split("|")[3:-1]
    on_0, on_2048, ends, longer, effect = (float(cell) for cell in row)
    assert abs(ends - longer * effect) < 2e-3 and abs(ends - on_2048 / on_0) < 2e-3


def test_n2_holds_the_swing_of_a_ratio_of_four_throughputs():
    """GPT's hand example: every Q of the repeated group off by 1 % between the repeats, Gamma off by 4 %."""
    nano, vllm = job()
    swing = {("nano", "on"): (101, 100), ("nano", "off"): (100, 101), ("vllm", "on"): (100, 101), ("vllm", "off"): (101, 100)}
    nano = [run if "-s1-" not in run[0] else
            nano_run(run[1]["config"]["line"], 1, 96 / swing[("nano", "on" if run[1]["config"]["line"].endswith("_pc") else "off")][run[1]["config"]["repeat"] - 1],
                     run[1]["config"]["repeat"]) for run in nano]
    vllm = [run if "-s1-" not in run[0] else
            vllm_run(run[1]["line"], 1, swing[("vllm", "on" if run[1]["prefix_cache"] else "off")][run[1]["group_repeat"] - 1],
                     run[1]["prefix_cache"], run[1]["group_repeat"]) for run in vllm]
    analysis = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    assert abs(analysis["n1"] - (1.01 * 1.01 - 1)) < 1e-9                               # B: 1.01 against 1 / 1.01
    assert abs(analysis["n2"] - ((101 / 100) ** 2 / (100 / 101) ** 2 - 1)) < 1e-9       # Gamma: 1.0201 against 0.9803


def test_the_screening_effect_puts_the_original_runs_back():
    nano, vllm = job()
    nano = [run if not run[0].endswith("k0_pc-C8-s2-r1.json") else nano_run("k0_pc", 2, 10.02, spike=4.0) for run in nano]
    nano += [nano_run(line, 2, 10.5, directory="OUT/rerun/nano") for line in ("k0_pc", "k0", "dspark_k7_pc", "dspark_k7")]
    vllm += [vllm_run(LINES[base], 2, q, on, directory="OUT/rerun/vllm") for base in LINES
             for q, on in ((100.0, True), (80.0, False))]
    final = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano))
    original = analyse(PROTOCOL, "S2048", nano, vllm, reference_of(nano), originals=True)
    k0 = next(gap for gap in original["gaps"] if gap["nano_line"] == "k0_pc")
    assert not original["invalid"] and abs(k0["nano"][2] - 96 / 10.02) < 1e-12
    assert without_screening(final, original)["changed"] == []
