import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_bench import percentile, request_latencies, throughput  # noqa: E402

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
