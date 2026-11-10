import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from round_robin_report import LATER_RATIOS, LATER_VALUES, load_finite_runs, measures, spread, standing_groups, table  # noqa: E402

LINES = ["k0", "k0_rr"]
PROTOCOL = {"gpu": {"name": "G4"}, "lines": {line: {"tree": "main", "k": 0, "k0_line": line} for line in LINES},
            "confirm": {"panel": "K128b", "concurrency": [8, 32], "versions": LINES}, "seeds": {"confirm": [1, 2]},
            "timing_noise": {"concurrency": [8], "seed": 1, "repeats": 2}}
SUMMARY = {"excess_prefill_tokens": 0, "q_steady": 100.0, "q_full": 90.0}


def bench_run(directory, line, concurrency, seed, repeat=1, spike=1.0):
    steps = [{"kind": "decode", "duration_ms": 10.0} for _ in range(5)] + [{"kind": "decode", "duration_ms": 10.0 * spike}]
    result = {"gpu_state_start": {"name": "G4"}, "steps": steps, "protocol_sha256": "protocol", "speculative_stats": None,
              "config": {"panel": "K128b", "concurrency": concurrency, "line": line, "seed": seed, "repeat": repeat},
              "manifest": {"commit": "aaaaaaa", "panel_sha256": "panel"},
              "requests": [{"token_ids": [1, 2], "deliveries": [[0, 1], [1, 1]]}]}
    return f"{directory}/{line}-C{concurrency}-s{seed}-r{repeat}.json", result, SUMMARY


def whole_group(directory, concurrency, seed, repeats):
    return [bench_run(directory, line, concurrency, seed, repeat) for line in LINES for repeat in range(1, repeats + 1)]


def test_a_group_stands_only_whole_with_the_repeats_the_protocol_asks_for():
    runs = whole_group("job", 8, 1, 2) + whole_group("job", 8, 2, 1) + whole_group("job", 32, 1, 1)
    assert set(standing_groups(runs, PROTOCOL)) == {(8, 1), (8, 2), (32, 1)}
    without_a_repeat = [run for run in runs if "k0_rr-C8-s1-r2" not in run[0]]
    assert set(standing_groups(without_a_repeat, PROTOCOL)) == {(8, 2), (32, 1)}
    without_a_line = [run for run in runs if "k0_rr-C32-s1-r1" not in run[0]]
    assert set(standing_groups(without_a_line, PROTOCOL)) == {(8, 1), (8, 2)}


def test_an_invalid_repeat_voids_its_group_and_a_whole_rerun_replaces_it():
    slow_repeat = [bench_run("job", "k0", 8, 1, 2, spike=4.0) if "k0-C8-s1-r2" in run[0] else run
                   for run in whole_group("job", 8, 1, 2)]
    assert standing_groups(slow_repeat, PROTOCOL) == {}
    assert set(standing_groups(slow_repeat + whole_group("rerun", 8, 1, 2), PROTOCOL)) == {(8, 1)}
    assert standing_groups(slow_repeat + whole_group("rerun", 8, 1, 2)[:2], PROTOCOL) == {}    # a rerun of one line only


def test_a_seed_outside_the_confirm_seeds_is_left_out():
    assert standing_groups(whole_group("job", 32, 7, 1), PROTOCOL) == {}


def test_a_result_file_with_a_non_finite_value_is_not_read(tmp_path):
    (tmp_path / "a.json").write_text('{"schema": "bench-v2", "phases": [NaN]}')
    (tmp_path / "b.json").write_text('{"schema": "bench-v2", "wall_seconds": 1e999}')
    assert load_finite_runs([tmp_path]) == []


def one_line(itl_p99, q_steady=100.0):
    return {"q_steady": q_steady, "q_full": 90.0, "itl_p50": 20.0, "itl_p99": itl_p99, "itl_max": 300.0,
            "prefill_gaps": 0.25, "ttft_p50": 50.0, "ttft_p99": 400.0}


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
    assert cells[13] == "400 → 400"                     # TTFT p99, both lines'


def test_measures_take_the_later_requests_and_the_longest_gap():
    durations = [("prefill", 600.0), ("decode", 20.0), ("prefill", 50.0), ("decode", 20.0), ("prefill", 100.0),
                 ("decode", 20.0)]
    steps, start = [], 0.0
    for index, (kind, duration) in enumerate(durations):
        steps.append({"step": index, "kind": kind, "batch_size": 1, "start_ms": start, "duration_ms": duration})
        start += duration
    requests = [{"submit_step": 0, "deliveries": [[0, 1], [1, 1], [3, 1], [5, 1]], "token_ids": [1, 2, 3, 4]},
                {"submit_step": 1, "deliveries": [[2, 1], [3, 1]], "token_ids": [1, 2]},      # 670 - 600
                {"submit_step": 2, "deliveries": [[2, 1]], "token_ids": [1]},                 # 670 - 620
                {"submit_step": 2, "deliveries": [[4, 1]], "token_ids": [1]}]                 # 790 - 620
    result = {"steps": steps, "requests": requests, "config": {"concurrency": 1}}
    values = measures(result, {"q_steady": 1.0, "q_full": 1.0, "ttft_ms": {"p50": 70.0, "p99": 600.0}})
    assert values["later_ttft_p50"] == 70.0      # of 50, 70 and 170: the request of the start, at 600, is left out
    assert values["later_ttft_p99"] > 160.0
    assert values["ttft_p99"] == 600.0           # the protocol's, over every request, is kept
    assert values["itl_max"] == 120.0            # the first request, across the 100 ms prefill step: 810 - 690


def test_table_takes_other_columns():
    groups = {(32, 1): {"k0": {"later_ttft_p50": 50.0, "later_ttft_p99": 100.0},
                        "k0_rr": {"later_ttft_p50": 50.0, "later_ttft_p99": 250.0}}}
    row = table(groups, {"k0": "k0_rr"}, [32], LATER_RATIOS, LATER_VALUES)[2]
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[3] == "2.50 (2.50–2.50)"
    assert cells[4:] == ["50 → 50", "100 → 250"]
