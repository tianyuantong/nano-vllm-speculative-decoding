import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from report_v2 import (crossover, format_range, invalid_reasons, timing_noise, tokens_equal,  # noqa: E402
                       verdict_a_beats_b, verdict_faster)

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
