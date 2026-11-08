"""prefix_report on simulated closed loops whose prefill and round times are known, so h, d and the predictions are."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prefix_report import analyse, gain_verdict, hits_check, markdown  # noqa: E402
from summarize_bench import summarize_v2  # noqa: E402

GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
LINES = {"k0": {"k": 0, "k0_line": "k0"}, "k0_pc": {"k": 0, "k0_line": "k0_pc", "prefix_cache": True},
         "dspark_k7": {"k": 7, "k0_line": "k0"}, "dspark_k7_pc": {"k": 7, "k0_line": "k0_pc", "prefix_cache": True}}
PROTOCOL = {"gpu": {"name": GPU}, "lines": LINES, "seeds": {"dev": [1, 2, 3]}, "timing_noise": {"seed": 1},
            "panels": {"S1024": {"shared_tokens": 1024}},
            "main": {"lines": list(LINES), "concurrency": [8], "seeds": "dev",
                     "pairs": [["k0", "k0_pc"], ["dspark_k7", "dspark_k7_pc"]]}}


def closed_loop(line, seed, concurrency=8, requests=48, output=60, tokens_per_round=1, round_ms=10.0, prefill_ms=40.0,
                prompt_len=2000, cached=0, repeat=1):
    """A prefill-first closed loop: every waiting request prefills in one step of prefill_ms each, otherwise every
    running one gains tokens_per_round in a round of round_ms."""
    steps, records, waiting, running, now, next_id = [], [], [], [], 0.0, 0

    def submit():
        nonlocal next_id
        record = {"id": f"q{next_id}", "category": "NIAH", "prompt_len": prompt_len, "submit_step": len(steps),
                  "deliveries": [], "finish_step": -1, "token_ids": [], "cached_tokens": cached}
        next_id += 1
        records.append(record)
        waiting.append(record)

    for _ in range(concurrency):
        submit()
    while waiting or running:
        index = len(steps)
        if waiting:
            batch, kind, duration = list(waiting), "prefill", prefill_ms * len(waiting)
            waiting.clear()
            for record in batch:
                record["deliveries"].append([index, 1])
                record["token_ids"].append(1)
            running += batch
            step = {"kind": kind, "batch_size": len(batch), "decode_rows": 0,
                    "prefill_tokens": len(batch) * (prompt_len - cached)}
        else:
            batch, duration = list(running), round_ms
            for record in batch:
                n = min(tokens_per_round, output - len(record["token_ids"]))
                record["deliveries"].append([index, n])
                record["token_ids"] += [1] * n
            step = {"kind": "decode" if tokens_per_round == 1 else "round", "batch_size": len(batch),
                    "decode_rows": len(batch), "prefill_tokens": 0}
        steps.append({"step": index, "start_ms": now, "duration_ms": duration, "tokens_out": 0, **step})
        now += duration
        for record in [r for r in running if len(r["token_ids"]) >= output]:
            record["finish_step"] = index
            running.remove(record)
            if next_id < requests:
                submit()
    stats = None if tokens_per_round == 1 else {"rounds": 1, f"accepted_{tokens_per_round - 1}": 1}
    return {"schema": "bench-v2", "protocol_sha256": "p", "manifest": {"commit": "abc", "panel_sha256": "s"},
            "gpu_state_start": {"name": GPU}, "config": {"panel": "S1024", "concurrency": concurrency, "line": line,
                                                       "seed": seed, "repeat": repeat, "num_speculative_tokens": 0},
            "steps": steps, "requests": records, "speculative_stats": stats, "preemptions": 0, "wall_seconds": now / 1000}


def runs_of(results):
    return [(f"job/{r['config']['line']}-s{r['config']['seed']}-r{r['config']['repeat']}.json", r, summarize_v2(r))
            for r in results]


def matrix(h=0.5, d=0.2, spec_d=0.1):
    results = []
    for seed in (1, 2, 3):
        results += [closed_loop("k0", seed), closed_loop("k0_pc", seed, prefill_ms=40 * (1 - h), round_ms=10 * (1 - d),
                                                          cached=1024),
                    closed_loop("dspark_k7", seed, tokens_per_round=3, round_ms=20.0),
                    closed_loop("dspark_k7_pc", seed, tokens_per_round=3, round_ms=20 * (1 - spec_d),
                                prefill_ms=40 * (1 - h), cached=1024)]
    results += [closed_loop("k0", 1, repeat=2), closed_loop("k0_pc", 1, repeat=2, prefill_ms=40 * (1 - h),
                                                            round_ms=10 * (1 - d), cached=1024),
                closed_loop("dspark_k7", 1, repeat=2, tokens_per_round=3, round_ms=20.0),
                closed_loop("dspark_k7_pc", 1, repeat=2, tokens_per_round=3, round_ms=20 * (1 - spec_d),
                            prefill_ms=40 * (1 - h), cached=1024)]
    return runs_of(results)


def test_h_d_and_the_predictions_come_out_of_the_step_times():
    analysis = analyse(matrix(), PROTOCOL, "S1024")
    assert not analysis["invalid"] and analysis["noise"] == 0 and analysis["pairs"] == 4
    assert analysis["noise_rho"] == 0 and analysis["rho_pairs"] == 1
    k0 = analysis["table"][(8, "k0")]
    spec = analysis["table"][(8, "dspark_k7")]
    assert len(k0) == len(spec) == 3
    for entry in k0:
        assert entry["h"] == pytest.approx(0.5) and entry["d"] == pytest.approx(0.2)
        assert entry["on"]["h_tok"] == pytest.approx(1024 / 2000) and entry["off"]["h_tok"] == 0
        assert entry["tokens_equal_share"] == 1
    for entry in spec:
        assert entry["h"] == pytest.approx(0.5) and entry["d"] == pytest.approx(0.1)
        # the simulation is the model the prediction with d assumes, up to the window's edges
        assert entry["rho"] == pytest.approx(entry["rho_hat_1"], rel=0.02)
        assert abs(entry["rho"] - entry["rho_hat_0"]) > abs(entry["rho"] - entry["rho_hat_1"])
        assert entry["G"] == pytest.approx(sum(entry["G_split"].values()))           # the split is exact
        assert entry["kappa_pf"] == pytest.approx(1, abs=0.05) and entry["kappa_r"] == pytest.approx(1, abs=0.05)
    assert "dspark_k7" in markdown(analysis)


def test_the_hits_check():
    off, on = closed_loop("k0", 1), closed_loop("k0_pc", 1, cached=1024)
    assert hits_check(off, 1024, False) == [] and hits_check(on, 1024, True) == []
    assert hits_check(closed_loop("k0_pc", 1, cached=768), 1024, True) == ["a request hit other than 1024 tokens"]
    on["requests"][0]["cached_tokens"] = 1280               # the steps still prefilled prompt_len - 1024
    assert hits_check(on, 1024, True) == ["a request hit other than 1024 tokens",
                                          "prefilled tokens differ from prompt_len - cached_tokens"]
    assert hits_check(closed_loop("k0", 1, cached=256), 0, False)[0] == "a request hit other than 0 tokens"


def test_a_failed_hits_check_voids_the_run():
    runs = matrix()
    name, result, summary = runs[1]                           # k0_pc of seed 1
    result["requests"][3]["cached_tokens"] = 0
    analysis = analyse(runs, PROTOCOL, "S1024")
    assert any("hits:" in reason for reason in analysis["invalid"][name])
    assert [entry["seed"] for entry in analysis["table"][(8, "k0")]] == [2, 3]
    assert "| 8 | k0 | 2 |" in markdown(analysis) and "| - |" in markdown(analysis)    # 2 of 3 seeds: not judged


def test_the_null_on_a_panel_without_a_shared_part():
    protocol = {**PROTOCOL, "panels": {"S1024": {"shared_tokens": 0}}}
    results = [closed_loop(line, seed, tokens_per_round=3 if "dspark" in line else 1) for seed in (1, 2, 3)
               for line in LINES]
    analysis = analyse(runs_of(results), protocol, "S1024")
    assert analysis["null_failures"] == []
    results[1]["requests"][0]["token_ids"] = results[1]["requests"][0]["token_ids"][:-1] + [2]
    assert analyse(runs_of(results), protocol, "S1024")["null_failures"]


def test_gain_verdict():
    assert gain_verdict([0.10, 0.12, 0.09], 0.004) == "faster"
    assert gain_verdict([-0.02, -0.03, -0.01], 0.004) == "slower"
    assert gain_verdict([0.02, -0.001, 0.03], 0.004) == "undecided"


def test_without_every_repeat_no_verdict_is_given():
    runs = [run for run in matrix() if not run[0].endswith("dspark_k7_pc-s1-r2.json")]
    analysis = analyse(runs, PROTOCOL, "S1024")
    assert not analysis["noise_complete"] and analysis["pairs"] == 3 and analysis["rho_pairs"] == 0
    text = markdown(analysis)
    assert "NOT GIVEN" in text and "faster" not in text and "undecided" not in text
