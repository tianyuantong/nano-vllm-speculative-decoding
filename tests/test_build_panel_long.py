import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_panel_long import bamboo_candidates, build_panel, embedded_candidates, stackselect_candidates  # noqa: E402

SPEC = {"salt": "s:", "window_fraction": 0.1, "quotas": {"code_completion": 2, "open_books": 2}}


def candidate(index, category, length):
    return {"question_id": f"q{index}", "category": category, "source": "pool", "src_id": str(index), "length": length}


def encode_by_length(prompt):
    return [0] * prompt


def prompt_of(candidate_row):
    return candidate_row["length"], "embedded_first_turn"


def pool(lengths):
    return [candidate(index, category, length) for index, (category, length) in enumerate(lengths)]


def test_the_final_length_decides_the_bucket():
    rows = pool([("code_completion", 7999), ("code_completion", 7200), ("code_completion", 8800), ("code_completion", 7199),
                 ("code_completion", 8801), ("open_books", 8000), ("open_books", 8100)])
    panel = build_panel(rows, 8000, SPEC, prompt_of, encode_by_length)
    assert sorted(entry["formatted_tokens"] for entry in panel if entry["category"] == "code_completion")[0] >= 7200
    assert all(7200 <= entry["formatted_tokens"] <= 8800 for entry in panel)
    assert len(panel) == 4 and {entry["recipe"] for entry in panel} == {"embedded_first_turn"}


def test_a_panels_prefix_keeps_the_shares():
    rows = pool([("code_completion", 8000)] * 2 + [("open_books", 8000)] * 2)
    panel = build_panel(rows, 8000, SPEC, prompt_of, encode_by_length)
    assert [entry["category"] for entry in panel[:2]].count("open_books") == 1


def test_a_category_short_of_its_quota_stops_the_build():
    rows = pool([("code_completion", 8000)] * 2 + [("open_books", 8000), ("open_books", 100)])
    with pytest.raises(ValueError):
        build_panel(rows, 8000, SPEC, prompt_of, encode_by_length)


def test_embedded_candidates_keep_the_panels_categories():
    rows = [{"sub_category": "open_books"}, {"sub_category": "code_completion"}, {"sub_category": "sort"}]
    assert [row["category"] for row in embedded_candidates(rows, {"open_books", "code_completion"})] == ["open_books", "code_completion"]


def test_recipe_candidates_name_their_source():
    assert stackselect_candidates([{"question_id": 7}])[0]["question_id"] == "stackselect_4k_7"
    assert bamboo_candidates([{"content": "x"}])[0]["category"] == "creative_writing"
