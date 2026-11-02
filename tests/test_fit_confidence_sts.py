import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fit_confidence_sts import (COST_SCHEMA, LOG_SCHEMA, auroc, calibration, expected_calibration_error,  # noqa: E402
                                fit_temperatures, main, replay)
from nanovllm.engine.verify_schedule import VerifySchedule  # noqa: E402


def logit(probability) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.log(probability / (1 - probability))


def planted_rows(temperatures: np.ndarray, rows: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Logits and accepted counts where draft i is accepted, once the earlier ones were, with sigmoid(z_i / T_i)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, (rows, len(temperatures)))
    accepted_each = rng.random(logits.shape) < 1 / (1 + np.exp(-logits / temperatures))
    return logits, np.cumprod(accepted_each, axis=1).sum(axis=1)


def test_sts_recovers_planted_temperatures():
    planted = np.array([2.0, 0.5, 1.0])
    logits, accepted = planted_rows(planted, 20_000)
    fitted = fit_temperatures(logits, accepted, np.geomspace(0.1, 10.0, 401))
    assert fitted == pytest.approx(planted, rel=0.1)
    raw, sts = calibration(logits, accepted, np.ones(3)), calibration(logits, accepted, fitted)
    assert sts["ece_pooled"] < raw["ece_pooled"]


def test_auroc_and_ece_on_hand_examples():
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) == 0.75
    assert auroc(np.array([0.5, 0.5]), np.array([False, True])) == 0.5          # a tie counts half
    assert expected_calibration_error(np.full(10, 0.3), np.arange(10) < 3) == pytest.approx(0.0)
    assert expected_calibration_error(np.full(10, 0.9), np.arange(10) < 3) == pytest.approx(0.6)


def test_replay_counts_what_a_shorter_verification_accepts():
    # k = 3, 2 rows; round times 10, 12, 14, 16 ms for 2, 4, 6, 8 tokens. Survivals 0.9, 0.81, 0.04 and 0.1, 0.01,
    # 0.001: the walk admits row 0's first two drafts, 4 tokens are a graph, so the lengths are 3 and 1. Row 0
    # accepted all 3 drafts and keeps 2 of them, row 1 accepted 2 and keeps none: 4 tokens in 12 ms against 7 in 16.
    schedule = VerifySchedule([2], [[10.0, 12.0, 14.0, 16.0]], [1.0, 1.0, 1.0], max_num_seqs=2)
    rounds = [{"logits": logit([[0.9, 0.9, 0.05], [0.1, 0.1, 0.1]]).tolist(), "accepted": [3, 2]}]
    predicted = replay(rounds, schedule, concurrency=2)
    assert predicted["gain"] == pytest.approx((4 / 12) / (7 / 16) - 1)
    assert predicted["verified_per_row"] == 2.0 and predicted["rounds"] == 1


def test_the_written_schedule_loads_in_the_engine(tmp_path):
    protocol = {"seeds": {"dev": [1, 2]},
                "verify_schedule": {"calibration": {"line": "dspark_k7", "panel": "D48", "concurrency": 2, "seeds": "dev",
                                                    "grid": {"low": 0.5, "high": 2.0, "points": 9}},
                                    "replay": {"concurrency": 2, "stop_if_gain_below": -1.0}}}
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    cost = {"schema": COST_SCHEMA, "protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
            "manifest": {"node": "gpu-pro6000-8", "commit": "abc1234"}, "gpu_state_start": {"power.limit": "300.00"},
            "config": {"batch_sizes": [1, 2]}, "round_ms": [[10.0, 10.5, 11.0, 11.5], [12.0, 13.0, 14.0, 15.0]]}
    (tmp_path / "cost.json").write_text(json.dumps(cost))
    logs = []
    for seed in (1, 2):
        logits, accepted = planted_rows(np.ones(3), 200, seed)
        rounds = [{"logits": logits[i:i + 2].tolist(), "accepted": accepted[i:i + 2].tolist()} for i in range(0, 200, 2)]
        log = tmp_path / f"run{seed}.confidences.json"
        log.write_text(json.dumps({"schema": LOG_SCHEMA, "line": "dspark_k7", "panel": "D48", "concurrency": 2,
                                   "seed": seed, "commit": "abc1234", "rounds": rounds}))
        logs.append(str(log))
    out = tmp_path / "verify-schedule.json"
    assert main(["--protocol", str(protocol_file), "--cost", str(tmp_path / "cost.json"), "--logs", *logs,
                 "--out", str(out)]) == 0
    schedule = VerifySchedule.load(str(out), num_drafts=3, max_num_seqs=2)
    assert len(schedule.temperatures) == 3 and json.loads(out.read_text())["calibration"]["rows"] == 400
