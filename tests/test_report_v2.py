import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from report_v2 import (count_runs, crossover, format_range, invalid_reasons, report, timing_noise,  # noqa: E402
                       tokens_equal, verdict_a_beats_b, verdict_faster)

PROTOCOL = {"gpu": {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}}


def test_timing_noise():
    assert timing_noise([(100.0, 100.3), (200.0, 199.2)]) == pytest.approx(0.0040161, abs=1e-6)


def test_verdict_faster():
    assert verdict_faster([1.10, 1.12, 1.09, 1.15, 1.11], 0.004) is True
    assert verdict_faster([1.01, 0.99, 1.02, 1.00, 1.03], 0.004) is False


def test_verdict_a_beats_b():
    assert verdict_a_beats_b([0.03, 0.05, 0.04, 0.02, 0.06], 0.004) == 1
    assert verdict_a_beats_b([0.03, -0.01, 0.04, 0.02, 0.06], 0.004) == 0
    assert verdict_a_beats_b([0.005, 0.006, 0.004, 0.007, 0.006], 0.004) == 0


def test_format_range():
    assert format_range([1.10, 1.12, 1.09, 1.15, 1.11]) == "1.09–1.15"
    assert format_range([1.48, 1.49, 1.50, 1.47, 1.49]) == "1.49 (1.47–1.50)"


def test_crossover():
    assert crossover({1: 1.49, 8: 1.12, 16: 1.01, 32: 0.97, 64: 0.90}) == 32
    assert crossover({1: 1.5, 8: 1.1}) is None


def run(name=PROTOCOL["gpu"]["name"], spike=1.0):
    steps = [{"kind": "decode", "duration_ms": 10.0} for _ in range(5)] + [{"kind": "decode", "duration_ms": 10.0 * spike}]
    return {"gpu_state_start": {"name": name}, "steps": steps}


SUMMARY = {"excess_prefill_tokens": 0, "q_steady": 100.0, "q_full": 90.0}


def test_invalid_reasons():
    assert invalid_reasons(run(), SUMMARY, PROTOCOL) == []
    assert invalid_reasons(run(name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"), SUMMARY, PROTOCOL) == \
        ["gpu name mismatch"]
    assert invalid_reasons(run(), {**SUMMARY, "excess_prefill_tokens": 5}, PROTOCOL) == ["preemption > 0"]
    assert invalid_reasons(run(spike=4.0), SUMMARY, PROTOCOL) == ["decode step or round > 3 x median"]


def test_tokens_equal():
    result = {"requests": [{"token_ids": [1, 2, 3]}, {"token_ids": [4]}]}
    assert tokens_equal(result, {"requests": [{"token_ids": [1, 2, 3]}, {"token_ids": [4]}]}) is True
    assert tokens_equal(result, {"requests": [{"token_ids": [1, 2, 9]}, {"token_ids": [4]}]}) is False


# ---------------------------------------------------------------- two source trees in one job (protocol v2.2, §8)
TWO_TREES = {**PROTOCOL, "lines": {
    "k0": {"tree": "main", "k": 0, "k0_line": "k0"},
    "dspark_var": {"tree": "main", "k": 7, "k0_line": "k0"},
    "k0_base": {"tree": "baseline", "k": 0, "k0_line": "k0_base"},
    "model_k3": {"tree": "baseline", "k": 3, "k0_line": "k0_base"},
}}
COMMIT_OF_TREE = {"main": "aaaaaaa", "baseline": "bbbbbbb"}


def bench_run(line, q_steady, repeat=1, commit=None):
    """One result of a K128b, C=64, seed-17011 job in directory job/."""
    commit = commit or COMMIT_OF_TREE[TWO_TREES["lines"][line]["tree"]]
    result = {**run(), "config": {"panel": "K128b", "concurrency": 64, "line": line, "seed": 17011, "repeat": repeat},
              "manifest": {"commit": commit, "panel_sha256": "panel"}, "protocol_sha256": "protocol",
              "requests": [{"token_ids": [1, 2], "deliveries": [[0, 1], [1, 1]]}], "speculative_stats": None}
    return f"job/{line}-C64-s17011-r{repeat}.json", result, {**SUMMARY, "q_steady": q_steady}


def test_each_line_divides_by_its_own_trees_k0():
    # The design's example: main k0 100, baseline k0 90, model_k3 99. Over its own k0 model_k3 is 1.10, not 0.99;
    # the two k0 differ by 11%, far beyond 2 N_t = 0.01, so the trees' R must not share a table.
    runs = [bench_run("k0", 100.0), bench_run("k0", 100.5, repeat=2), bench_run("dspark_var", 120.0),
            bench_run("k0_base", 90.0), bench_run("model_k3", 99.0)]
    text = report(runs, TWO_TREES)
    assert "| K128b | 64 | model_k3 | k0_base | 1 | 1.10 (1.10–1.10) |" in text
    assert "| K128b | 64 | dspark_var | k0 | 1 | 1.20 (1.20–1.20) |" in text
    assert "| K128b | 64 | k0 | k0_base | 1 | +0.111..+0.111 | no |" in text
    assert "| dspark_var | model_k3 |" not in text                    # no line against line across trees


def test_a_job_holds_one_commit_per_tree():
    runs = [bench_run("k0", 100.0), bench_run("dspark_var", 120.0), bench_run("k0_base", 90.0),
            bench_run("model_k3", 99.0)]
    invalid, _, q = count_runs(runs, TWO_TREES)
    assert not any(invalid.values()) and set(q[("K128b", 64, 17011)]) == set(TWO_TREES["lines"])
    runs[1] = bench_run("dspark_var", 120.0, commit="ccccccc")         # a main line from another commit
    invalid, _, _ = count_runs(runs, TWO_TREES)
    assert all(reasons == ["lines of one (C, seed) split across jobs"] for reasons in invalid.values())
