import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from samehost_report import gap_remains, report  # noqa: E402


def test_a_gap_remains_only_when_the_ranges_do_not_overlap():
    assert gap_remains([90, 95, 92], [96, 100, 98])
    assert gap_remains([96, 100], [90, 95])
    assert not gap_remains([90, 96, 92], [95, 100])


def test_report_pairs_the_lines_by_seed():
    nano = {("dspark_k7", 1): {1: {"q": 90.0, "tokens_per_round": 2.7}, 2: {"q": 91.0, "tokens_per_round": 2.8}}}
    vllm = {("dspark_k7", 1): {1: {"q": 100.0, "tokens_per_round": 2.8}, 2: {"q": 99.0, "tokens_per_round": 2.8}}}
    assert "| dspark_k7 | 1 | 2 | 90.5 (90.0–91.0) | 99.5 (99.0–100.0) | 0.910 | yes |" in report(nano, vllm)
