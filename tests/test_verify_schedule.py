"""The verify schedule: DSpark's Algorithm 1 on an interpolated round time, then the graph filled in rank order."""
import json

import numpy as np
import pytest

from nanovllm.engine.verify_schedule import (
    SCHEMA,
    VerifySchedule,
    survival_probabilities,
    verify_lengths,
    verify_token_counts,
)


def test_interpolation_and_filling_reach_the_step_costs_best():
    # The design's example (§2.4): rows A and B, graphs of 2, 4, 6, 8 tokens at 10.0, 10.4, 12.4, 14.4 ms. On the
    # interpolated time the walk admits A1, A2, B1 and stops before A3; 5 tokens round up to the graph of 6, which
    # A3 fills: A verifies 4, B 2.
    survival = np.array([[0.9, 0.6, 0.3], [0.5, 0.2, 0.1]])
    round_ms = np.interp(np.arange(9), [2, 4, 6, 8], [10.0, 10.4, 12.4, 14.4])
    assert verify_lengths(survival, round_ms, [2, 4, 6, 8]) == [4, 2]


@pytest.mark.parametrize("second_survival", [0.72, 0.0], ids=["after_A", "after_B"])
def test_the_walk_stops_before_reading_a_later_drafts_confidence(second_survival):
    # The paper's appendix A: admitting draft 1 already lowers the throughput (0.9 < 1.0), so whatever draft 1
    # turns out to be, and so whatever draft 2's survival is, only the anchor is verified.
    round_ms = np.array([1.0, 1.0, 2.0, 1 / 0.45])
    assert verify_lengths(np.array([[0.8, second_survival]]), round_ms, [1, 2, 3]) == [1]


def test_zero_survival_positions_still_fill_the_graph():
    # Three rows, every survival 0: the walk admits nothing, 3 tokens round up to the graph of 4, and the first
    # position in rank order (row 0's, by the tie rule) fills it.
    round_ms = 10.0 + np.arange(25)
    assert verify_lengths(np.zeros((3, 7)), round_ms, verify_token_counts(3, 8)) == [2, 1, 1]


def test_a_flat_round_time_verifies_every_draft():
    assert verify_lengths(np.full((2, 7), 0.5), np.full(17, 12.0), verify_token_counts(2, 8)) == [8, 8]


def test_temperatures_can_reorder_rows():
    # Calibration rescales each position's confidence, so it can change which row a slot goes to: (0.9, 0.6) beats
    # (0.7, 0.7) at the second position, 0.540 to 0.490, until the first position's temperature is 10.
    logits = np.log(np.array([[0.9, 0.6], [0.7, 0.7]]) / (1 - np.array([[0.9, 0.6], [0.7, 0.7]])))
    raw = survival_probabilities(logits, np.array([1.0, 1.0]))[:, 1]
    calibrated = survival_probabilities(logits, np.array([10.0, 1.0]))[:, 1]
    assert raw[0] > raw[1] and calibrated[0] < calibrated[1]


def test_variable_length_graphs_hold_every_full_batch():
    assert verify_token_counts(4, 8) == [1, 2, 4, 8, 16, 24, 32]
    assert all(batch_size * 8 in verify_token_counts(64, 8) for batch_size in range(1, 65))


def write_schedule(tmp_path, **fields) -> str:
    table = {"schema": SCHEMA, "batch_sizes": [1, 2], "round_ms": [[10.0, 10.0, 11.0], [12.0, 14.0, 13.0]],
             "confidence_temperatures": [1.0, 1.0], **fields}
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(table))
    return str(path)


def test_the_frozen_table_is_interpolated_by_token_count(tmp_path):
    # k = 2. Batch 2's times at 2, 4, 6 tokens become 12, 14, 14 (never cheaper for more), linear in between; a
    # round of 1 row reads batch 1's.
    schedule = VerifySchedule.load(write_schedule(tmp_path), num_drafts=2, max_num_seqs=2)
    assert schedule.round_ms(2).tolist() == [12.0, 12.0, 12.0, 13.0, 14.0, 14.0, 14.0]
    assert schedule.round_ms(1).tolist() == [10.0, 10.0, 10.0, 11.0]
    assert schedule.token_counts == [1, 2, 3, 6]


def test_a_table_that_does_not_fit_the_engine_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        VerifySchedule.load(write_schedule(tmp_path), num_drafts=7, max_num_seqs=2)     # 2 temperatures for k = 7
    with pytest.raises(ValueError):
        VerifySchedule.load(write_schedule(tmp_path), num_drafts=2, max_num_seqs=4)     # no row for 4 sequences
    with pytest.raises(ValueError):
        VerifySchedule.load(write_schedule(tmp_path, schema="other"), num_drafts=2, max_num_seqs=2)


def test_non_finite_confidences_stop_the_round(tmp_path):
    schedule = VerifySchedule.load(write_schedule(tmp_path), num_drafts=2, max_num_seqs=2)
    with pytest.raises(AssertionError):
        schedule.verify_lengths(np.array([[0.0, np.nan]]))
