import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gate_task_accuracy import (CI_LOW_MIN, MAX_UNPARSEABLE, MIN_K0_ACCURACY, extract_final_number,  # noqa: E402
                                gate_verdict, is_correct, load_gsm8k, paired_delta)


@pytest.mark.parametrize("text, expected", [
    ("so 42.\n#### 42", "42"),
    ("#### 1,234", "1234"),
    ("cost $5.50 then #### $18", "18"),
    ("7 apples and 3 pears", "3"),
    ("#### -3", "-3"),
    ("#### 12\n\nnote 99", "12"),
    ("#### 5 #### 6", "6"),
    ("no numbers", None),
])
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


@pytest.mark.parametrize("prediction, truth, expected", [("3.0", "3", True), ("1234", "1,234", True), (None, "3", False)])
def test_is_correct(prediction, truth, expected):
    assert is_correct(prediction, truth) is expected


def test_paired_delta():
    k0 = [True] * 92 + [False] * 5 + [True] * 3
    k3 = [True] * 92 + [True] * 5 + [False] * 3            # b = 5, c = 3, n = 100
    assert paired_delta(k0, k3) == pytest.approx((0.02, -0.0352984, 0.0752984), abs=1e-6)
    assert paired_delta([True, False] * 5, [True, False] * 5) == (0.0, 0.0, 0.0)
    assert paired_delta([True] * 4 + [False] * 6, [False] * 10) == pytest.approx((-0.4, -0.7036419, -0.0963581), abs=1e-6)


@pytest.mark.parametrize("arguments, expected", [
    ((0.91, 0.01, 0.01, -0.012), "PASS"),
    ((0.91, 0.01, 0.01, -0.04), "FAIL"),
    ((0.91, 0.03, 0.01, 0.0), "INVALID"),
    ((0.75, 0.0, 0.0, 0.0), "INVALID"),
])
def test_gate_verdict(arguments, expected):
    assert gate_verdict(*arguments) == expected


def test_thresholds_are_the_protocols():
    gate = json.loads((ROOT / "benchmarks" / "protocol-v2.json").read_text())["gates"]["G5"]
    assert (CI_LOW_MIN, MAX_UNPARSEABLE, MIN_K0_ACCURACY) == (gate["ci_low_min"], gate["max_unparseable"],
                                                              gate["min_k0_accuracy"])


def test_load_gsm8k(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "a #### b\n#### 1,234"}) + "\n")
    assert load_gsm8k(path) == [{"question": "q", "truth": "1234"}]
