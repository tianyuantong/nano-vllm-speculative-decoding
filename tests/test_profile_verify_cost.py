import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from profile_verify_cost import EveryRow, round_totals  # noqa: E402


def test_a_round_is_its_phases_summed():
    phases = [{"phase": name, "duration_ms": ms} for name, ms in
              [("propose", 2.0), ("schedule", 0.5), ("verify", 10.0), ("ctx_store", 1.0), ("accept", 0.5),
               ("commit_copy", 0.25), ("propose", 2.5), ("schedule", 0.5), ("verify", 11.0), ("ctx_store", 1.0),
               ("accept", 0.5), ("commit_copy", 0.5)]]
    assert round_totals(phases) == [14.25, 16.0]


def test_every_row_verifies_the_same_length():
    assert EveryRow(5).verify_lengths(np.zeros((3, 7))) == [5, 5, 5]
    with pytest.raises(AssertionError):
        EveryRow(5).verify_lengths(np.full((1, 7), np.nan))
