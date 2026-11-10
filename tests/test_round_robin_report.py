import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from round_robin_report import spread, table  # noqa: E402


def one_line(itl_p99, q_steady=100.0):
    return {"q_steady": q_steady, "q_full": 90.0, "itl_p50": 20.0, "itl_p99": itl_p99, "prefill_gaps": 0.25,
            "ttft_p50": 50.0, "ttft_p99": 400.0}


def test_spread_is_the_median_and_the_range():
    assert spread([0.70, 0.60, 0.65]) == "0.65 (0.60–0.70)"


def test_table_pairs_each_seed_before_taking_the_median():
    groups = {(32, 1): {"k0": one_line(100.0), "k0_rr": one_line(70.0, q_steady=98.0)},
              (32, 2): {"k0": one_line(200.0), "k0_rr": one_line(120.0, q_steady=100.0)}}
    row = table(groups, {"k0": "k0_rr"}, [8, 32])[2]    # C = 8 has no group and gets no row
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:3] == ["k0_rr / k0", "32", "2"]
    assert cells[3] == "0.99 (0.98–1.00)"               # Q_steady: 98/100 and 100/100
    assert cells[5] == "0.65 (0.60–0.70)"               # ITL p99: 70/100 and 120/200, not 95/150
    assert cells[9] == "150.0 → 95.0"                   # the medians of each line's own milliseconds
