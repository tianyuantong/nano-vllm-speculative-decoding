import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pytest  # noqa: E402

from summarize_bench import (excess_prefill_tokens, full_batch_share, itl_ms, mbu, percentile,  # noqa: E402
                             prefill_share, q_full, q_steady, request_latencies, round_ms, steady_window,
                             throughput, tokens_per_round, ttft_ms)

TIMINGS = [
    {"step": 0, "kind": "prefill", "batch_size": 2, "tokens_out": 20, "start_ms": 0.0, "duration_ms": 10.0},
    {"step": 1, "kind": "round", "batch_size": 2, "tokens_out": 6, "start_ms": 10.0, "duration_ms": 20.0},
    {"step": 2, "kind": "round", "batch_size": 1, "tokens_out": 2, "start_ms": 30.0, "duration_ms": 15.0},
]
OUTPUTS = [
    {"token_ids": [1, 2, 3, 4], "first_token_step": 0, "finish_step": 1},
    {"token_ids": [1, 2, 3, 4, 5], "first_token_step": 0, "finish_step": 2},
]


def test_throughput_is_completion_tokens_over_the_gpu_timeline():
    assert throughput(TIMINGS, OUTPUTS) == 9 / (45.0 / 1000)


def test_request_latencies_from_step_indices():
    latencies = request_latencies(TIMINGS, OUTPUTS)
    assert latencies[0] == {"ttft_ms": 10.0, "e2e_ms": 30.0, "tpot_ms": 20.0 / 3}
    assert latencies[1] == {"ttft_ms": 10.0, "e2e_ms": 45.0, "tpot_ms": 35.0 / 4}


def test_percentile_is_nearest_rank():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([3.0, 1.0, 2.0], 95) == 3.0


# ---------------------------------------------------------------- v2: C=2; prompts rA 100, rB 120, rC 90
def v2_steps(prefill_2_tokens=90):
    rows = [("prefill", 2, 220, 0, 10), ("decode", 2, 2, 10, 5), ("prefill", 1, prefill_2_tokens, 15, 8),
            ("decode", 2, 2, 23, 5), ("decode", 1, 1, 28, 4)]
    return [{"step": index, "kind": kind, "batch_size": batch, "tokens_out": tokens, "start_ms": float(start),
             "duration_ms": float(duration)} for index, (kind, batch, tokens, start, duration) in enumerate(rows)]


V2_REQUESTS = [
    {"id": "rA", "category": "NIAH", "prompt_len": 100, "submit_step": 0, "deliveries": [[0, 1], [1, 1], [3, 1]], "finish_step": 3},
    {"id": "rB", "category": "NIAH", "prompt_len": 120, "submit_step": 0, "deliveries": [[0, 1], [1, 1]], "finish_step": 1},
    {"id": "rC", "category": "NIAH", "prompt_len": 90, "submit_step": 2, "deliveries": [[2, 1], [3, 1], [4, 1]], "finish_step": 4},
]


def test_v2_window_and_throughputs():
    steps = v2_steps()
    assert steady_window(steps, 2) == (1, 3)
    assert q_steady(steps, V2_REQUESTS, 2) == pytest.approx(5 / 0.018)
    assert q_full(steps, V2_REQUESTS) == pytest.approx(250.0)


def test_v2_latencies():
    steps = v2_steps()
    assert ttft_ms(steps, V2_REQUESTS) == [10.0, 10.0, 8.0]
    assert itl_ms(steps, V2_REQUESTS, (1, 3)) == [13.0, 5.0]
    assert round_ms(steps, (1, 3)) == [5.0, 5.0]


def test_v2_tokens_per_round_skips_the_prefill_delivery():
    request_d = {"id": "rD", "category": "NIAH", "prompt_len": 10, "submit_step": 0, "deliveries": [[0, 1], [1, 3], [2, 2]]}
    assert tokens_per_round(V2_REQUESTS + [request_d]) == [1.0, 1.0, 1.0, 2.5]


def test_v2_preemption_and_shares():
    assert excess_prefill_tokens(v2_steps(), V2_REQUESTS) == 0
    assert excess_prefill_tokens(v2_steps(prefill_2_tokens=95), V2_REQUESTS) == 5
    assert full_batch_share(v2_steps(), 2) == pytest.approx(2 / 3)
    assert prefill_share(v2_steps()) == pytest.approx(0.5625)


def test_v2_mbu():
    assert mbu(v2_steps(), V2_REQUESTS, 1000, 10, 1e6) == pytest.approx([0.644, 0.586, 0.48])
