import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from slo_report import (capacity, format_capacity, itl_ms, meets, queue_lengths, report, request_latencies,  # noqa: E402
                        slow_steps, summarize_run)

TIERS = {"Conversational": {"ttft_ms": 2000, "tpot_ms": 100}, "Interactive": {"ttft_ms": 500, "tpot_ms": 30}}
PROTOCOL = {"open_loop": {"warmup_fraction": 0.0, "slo": {"percentile": 99, "tiers": TIERS}}}


def request(arrival, deliveries, finished=True):
    return {"arrival_seconds": arrival, "deliveries": deliveries, "finish_step": 9 if finished else -1}


def test_latencies_run_from_the_planned_arrival():
    row = request_latencies(request(1.0, [[0, 1, 1.5], [1, 2, 1.7], [2, 1, 1.9]]))
    assert row["ttft_ms"] == pytest.approx(500)
    assert row["e2e_ms"] == pytest.approx(900)
    assert row["tpot_ms"] == pytest.approx(400 / 3)               # 4 tokens: per request, not a mean of the gaps (133 vs 200)


def test_an_unfinished_request_is_an_infinite_latency():
    row = request_latencies(request(0.0, [[0, 1, 0.1]], finished=False))
    assert row["ttft_ms"] == pytest.approx(100) and math.isinf(row["e2e_ms"]) and math.isinf(row["tpot_ms"])
    assert math.isinf(request_latencies(request(0.0, [], finished=False))["ttft_ms"])


def test_a_one_token_output_has_no_tpot():
    assert request_latencies(request(0.0, [[0, 1, 0.2]]))["tpot_ms"] is None


def test_itl_is_the_gap_between_a_requests_deliveries():
    assert itl_ms([request(0.0, [[0, 1, 0.1], [1, 1, 0.3]]), request(0.0, [[0, 1, 0.1]])]) == pytest.approx([200])


def test_queue_lengths_count_earlier_requests_without_a_first_token():
    requests = [request(0.0, [[0, 1, 1.0]]), request(0.5, [[1, 1, 3.0]]), request(0.6, [[1, 1, 3.0]]), request(2.0, [[2, 1, 4.0]])]
    assert queue_lengths(requests) == [0, 1, 2, 2]


def test_slow_steps_are_judged_within_a_kind_and_batch_size():
    steps = [{"kind": "round", "batch_size": 8, "duration_ms": 20.0}] * 5 + [{"kind": "round", "batch_size": 64, "duration_ms": 50.0}] * 5
    assert slow_steps(steps) == 0                                  # 50 > 3 x 16 would flag every large batch
    steps += [{"kind": "round", "batch_size": 8, "duration_ms": 61.0}]
    assert slow_steps(steps) == 1


def run(arrivals, finished=True, line="k0", rate=3.0, seed=1, ttft=0.1, gap=0.01, tokens=11, preemptions=0):
    requests = []
    for index, arrival in enumerate(arrivals):
        deliveries = [[index, 1, arrival + ttft]] + [[index, 1, arrival + ttft + gap * n] for n in range(1, tokens)]
        requests.append(request(arrival, deliveries, finished))
    return {"schema": "open-loop-v1", "config": {"panel": "D48", "line": line, "rate": rate, "seed": seed}, "requests": requests,
            "steps": [], "preemptions": preemptions, "full_collections": 0}


def test_summary_skips_only_the_warm_up_requests_by_position():
    result = run([0.0, 1.0, 2.0, 3.0])
    result["requests"][0]["deliveries"] = [[0, 1, 50.0]]           # the warm-up request is slow and does not count
    assert summarize_run(result, 0.25)["ttft_p99"] == pytest.approx(100)
    assert summarize_run(result, 0.0)["ttft_p99"] == pytest.approx(50000)


def test_meets_needs_both_limits():
    assert meets({"ttft_p99": 1999, "tpot_p99": 99}, TIERS["Conversational"])
    assert not meets({"ttft_p99": 2001, "tpot_p99": 99}, TIERS["Conversational"])
    assert not meets({"ttft_p99": 400, "tpot_p99": 31}, TIERS["Interactive"])


def test_capacity_is_the_interval_before_the_first_failure():
    assert capacity({3: True, 4: True, 5: False, 6: True, 7: False}) == (4, 5)    # a later pass does not move it
    assert format_capacity(capacity({3: True, 4: True})) == "at least 4"
    assert format_capacity(capacity({3: False})) == "below 3"
    assert format_capacity(capacity({3: True, 4: False})) == "3 to 4"


def test_every_seed_must_pass_a_rate():
    good, late = run([0.0, 1.0], seed=1), run([0.0, 1.0], seed=2, ttft=3.0)
    text = report([good, late], PROTOCOL)
    assert "| D48 | k0 | Conversational | below 3 |" in text


def test_a_run_that_does_not_drain_fails_the_rate():
    text = report([run([0.0, 1.0], finished=False)], PROTOCOL)
    assert "| D48 | k0 | Interactive | below 3 |" in text
    assert "inf" in text
