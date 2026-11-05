import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from samehost_report import gap_remains, nano_rows, report  # noqa: E402

GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
PROTOCOL = {"gpu": {"name": GPU}, "lines": {"dspark_k7": {"tree": "main", "k": 7}}}


def test_a_gap_remains_only_when_the_ranges_do_not_overlap():
    assert gap_remains([90, 95, 92], [96, 100, 98])
    assert gap_remains([96, 100], [90, 95])
    assert not gap_remains([90, 96, 92], [95, 100])


def rows(*values):
    return {seed: {"q": value, "tokens_per_round": 2.7} for seed, value in enumerate(values, 1)}


def test_report_pairs_the_lines_by_seed():
    text = report({("dspark_k7", 1): rows(90.0, 91.0)}, {("dspark_k7", 1): rows(100.0, 99.0)}, [1, 2], [])
    assert "| dspark_k7 | 1 | 2 | 90.5 (90.0–91.0) | 99.5 (99.0–100.0) | 0.910 | yes |" in text


def test_a_short_set_of_seeds_gets_no_verdict():
    text = report({("dspark_k7", 1): rows(90.0)}, {("dspark_k7", 1): rows(100.0)}, [1, 2, 3, 4, 5], [])
    assert "| incomplete |" in text and "| yes |" not in text


def run(seed, directory="job", spike=1.0):
    steps = [{"kind": "round", "duration_ms": 10.0}] * 5 + [{"kind": "round", "duration_ms": 10.0 * spike}]
    result = {"gpu_state_start": {"name": GPU}, "steps": steps, "wall_seconds": 10.0, "speculative_stats": None,
              "requests": [{"token_ids": [1] * 100, "deliveries": [[0, 1], [1, 99]]}], "protocol_sha256": "p",
              "manifest": {"commit": "a", "panel_sha256": "panel"},
              "config": {"panel": "D48", "concurrency": 8, "line": "dspark_k7", "seed": seed, "repeat": 1}}
    return f"{directory}/dspark_k7-C8-s{seed}-r1.json", result, {"preemptions": 0, "q_steady": 1.0, "q_full": 1.0}


def test_an_invalid_run_leaves_the_set_until_its_rerun_replaces_it():
    rows_, bad = nano_rows([run(1), run(2, spike=4.0)], PROTOCOL)
    assert set(rows_[("dspark_k7", 8)]) == {1} and len(bad) == 1
    rows_, bad = nano_rows([run(1), run(2, spike=4.0), run(2, directory="rerun")], {**PROTOCOL})
    assert set(rows_[("dspark_k7", 8)]) == {1, 2}
