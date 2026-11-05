"""G2-real's checks on CPU (tools/gate_sampler_distribution.py): the variable-length layout, and which rows qualify."""
import sys
from pathlib import Path

import gpu_import_stubs  # noqa: F401  (before the engine import)
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gate_sampler_distribution import check_real_position, row_first_positions  # noqa: E402

POSITION = {"p": [0.6, 0.3, 0.1], "q": [0.2, 0.5, 0.3], "accept_mass": 0.2 + 0.3 + 0.1}


def test_only_rows_that_verify_a_draft_and_stop_short_qualify():
    assert row_first_positions(1, None, 7) == [0]
    assert row_first_positions(5, [1, 2, 7, 8, 5], 7) == [1, 2, 4]


@pytest.mark.parametrize("num_drafts", [1, 7])
def test_both_layouts_reproduce_the_acceptance_rate_and_the_target(num_drafts):
    report = check_real_position(POSITION, 16384, 100000, 0.001, torch.device("cpu"), num_drafts)
    assert report["accept_error"] < 0.02
    assert report["worst_frequency_gap"] <= report["frequency_bound"]
