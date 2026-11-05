import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from round_robin_report import closed_loop_failures, measures, slo_failures, warmup_boundary  # noqa: E402

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "protocol-v2.4.json").read_text())
VERDICT = PROTOCOL["round_robin"]["verdict"]


def result():
    """Two requests in a closed loop at C=2: the second's prefill ends at step 1, the warm-up boundary."""
    steps = [{"step": index, "start_ms": 10.0 * index, "duration_ms": 10.0} for index in range(5)]
    requests = [{"submit_step": 0, "deliveries": [[0, 1], [2, 1], [4, 1]]},
                {"submit_step": 0, "deliveries": [[1, 1], [2, 1], [3, 1]]}]
    return {"config": {"concurrency": 2}, "steps": steps, "requests": requests}


def test_the_boundary_is_the_initial_groups_last_first_delivery():
    assert warmup_boundary(result()) == 1


def test_gaps_before_the_boundary_do_not_count():
    # request 0's first gap starts at step 0, before the boundary; the others are 20, 10 and 10 ms.
    assert measures(result())["itl_p99"] == pytest.approx(20.0)


def ratios(**row):
    return {seed: {"q_full": 1.0, "itl_p99": 0.6, "ttft_p50": 1.5, "ttft_p99": 1.1, **row} for seed in (1, 2, 3)}


def test_a_clean_line_meets_the_verdict():
    assert closed_loop_failures(ratios(), 32, VERDICT) == []


def test_c8_has_its_own_itl_limit():
    assert closed_loop_failures(ratios(itl_p99=1.01), 8, VERDICT) == []
    assert closed_loop_failures(ratios(itl_p99=1.03), 8, VERDICT) != []


def test_q_full_is_judged_on_the_median_and_only_the_median():
    rows = ratios()
    rows[1]["q_full"] = 0.90
    assert closed_loop_failures(rows, 32, VERDICT) == []           # one seed down, the median holds
    assert any("median Q_full" in reason for reason in closed_loop_failures(ratios(q_full=0.97), 32, VERDICT))


def test_itl_and_ttft_hold_for_every_seed():
    rows = ratios()
    rows[2]["ttft_p99"] = 1.3
    assert len(closed_loop_failures(rows, 32, VERDICT)) == 1
    rows[2]["itl_p99"] = 0.75
    assert len(closed_loop_failures(rows, 32, VERDICT)) == 2


def open_run(line, ttft_seconds):
    deliveries = [[0, 1, ttft_seconds], [1, 1, ttft_seconds + 0.01]]
    return {"config": {"line": line, "rate": 5.0, "seed": 1}, "preemptions": 0, "full_collections": 0, "steps": [],
            "requests": [{"arrival_seconds": 0.0, "deliveries": deliveries, "finish_step": 1}]}


def test_a_tier_lost_on_the_same_plan_fails_the_arm():
    spec = PROTOCOL["open_loop"]
    pairs, lines = PROTOCOL["round_robin"]["pairs"], ["dspark_k7"]
    assert slo_failures([open_run("dspark_k7", 0.4), open_run("dspark_k7_rr", 0.45)], pairs, lines, spec) == []
    failures = slo_failures([open_run("dspark_k7", 0.4), open_run("dspark_k7_rr", 0.6)], pairs, lines, spec)
    assert failures == ["dspark_k7 at 5/s: Interactive passes without round robin and fails with it"]
