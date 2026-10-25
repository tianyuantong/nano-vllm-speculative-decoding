import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_panel_k128 import interleave, interleave_proportional, rank_key, select_candidates  # noqa: E402

SALT = "bench-v2-confirm:"


def candidate(question_id, category="A", source="s", src_id=None):
    return {"question_id": question_id, "category": category, "source": source, "src_id": src_id or question_id}


def test_rank_key():
    assert rank_key(SALT, "a") == hashlib.sha256(b"bench-v2-confirm:a").hexdigest()


def by_rank(ids):
    return sorted(ids, key=lambda question_id: rank_key(SALT, question_id))


def test_excluded_id_is_skipped():
    chosen = select_candidates([candidate(i) for i in "xyz"], {"y"}, set(), SALT, {"A": 2}, lambda c: True)
    assert [c["question_id"] for c in chosen["A"]] == by_rank(["x", "z"])


def test_excluded_source_is_skipped():
    pool = [candidate("x"), candidate("y", src_id="shared"), candidate("z")]
    chosen = select_candidates(pool, set(), {("s", "shared")}, SALT, {"A": 2}, lambda c: True)
    assert "y" not in [c["question_id"] for c in chosen["A"]]


def test_a_candidate_that_does_not_fit_is_replaced_by_the_next():
    ranked = by_rank(["p", "q", "r"])
    chosen = select_candidates([candidate(i) for i in "pqr"], set(), set(), SALT, {"A": 2},
                               lambda c: c["question_id"] != ranked[0])
    assert [c["question_id"] for c in chosen["A"]] == ranked[1:]


def test_too_few_candidates_raise():
    with pytest.raises(ValueError):
        select_candidates([candidate(i) for i in "abcd"], set(), set(), SALT, {"A": 5}, lambda c: True)


def test_quota_per_category_in_rank_order():
    pool = [candidate(f"a{i}", "A") for i in range(6)] + [candidate(f"b{i}", "B") for i in range(4)]
    chosen = select_candidates(pool, set(), set(), SALT, {"A": 3, "B": 2}, lambda c: True)
    assert [c["question_id"] for c in chosen["A"]] == by_rank([f"a{i}" for i in range(6)])[:3]
    assert [c["question_id"] for c in chosen["B"]] == by_rank([f"b{i}" for i in range(4)])[:2]


def test_interleave():
    assert interleave({"A": ["a1", "a2"], "B": ["b1"], "C": ["c1", "c2", "c3"]}, ["A", "B", "C"]) == \
        ["a1", "b1", "c1", "a2", "c2", "c3"]


def test_proportional_interleave_keeps_every_prefix_within_one_prompt_of_each_share():
    quotas = {"code_completion": 40, "NIAH": 43, "open_books": 37, "creative_writing": 8}
    total = sum(quotas.values())
    panel = interleave_proportional({c: [(c, i) for i in range(n)] for c, n in quotas.items()}, list(quotas))
    for c, n in quotas.items():
        assert [i for category, i in panel if category == c] == list(range(n))    # each list whole, in order
    for length in range(1, total + 1):
        prefix = [category for category, _ in panel[:length]]
        assert all(abs(prefix.count(c) - length * n / total) <= 1 for c, n in quotas.items())
