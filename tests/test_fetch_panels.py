import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_panels import A1_GROUPS, A2_GROUPS, GROUP_SIZE, select_panel  # noqa: E402


def samples(prefix, count):
    return [{"question_id": f"{prefix}{index}", "prompt_token_ids": [index]} for index in range(count)]


def test_select_panel_takes_the_performance_md_groups_in_order():
    panel = select_panel(samples("a", 48), samples("b", 24))
    assert len(panel) == (len(A1_GROUPS) + len(A2_GROUPS)) * GROUP_SIZE == 48
    assert panel[0] == {"id": "a8", "panel": "A1", "group": 2, "prompt_token_ids": [8]}
    assert panel[24] == {"id": "b0", "panel": "A2", "group": 0, "prompt_token_ids": [0]}
    assert len({entry["id"] for entry in panel}) == 48
