import hashlib
import json
import re
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
TOP_LEVEL = {"version", "frozen", "gpu", "models", "sampling", "lines", "max_model_len", "max_num_batched_tokens", "concurrency", "seeds",
             "order", "panels", "kv_budget", "warmup", "metrics", "timing_noise", "decision", "invalid_if",
             "gates", "external", "confirm"}
DERIVED = {"derived_from", "changes_from_v2"}    # v2.1 records what it changed from the frozen v2
# v2.2 (M4) adds the two source trees, the frozen verify schedule, the dev lines and the pre-registration; it runs no vLLM
TOP_LEVEL_V22 = (TOP_LEVEL - {"external"}) | {"derived_from", "changes_from_v2_1", "trees", "verify_schedule", "dev",
                                               "hypotheses", "report_only", "stop_if"}
TOP_LEVEL_V23 = (TOP_LEVEL_V22 - {"changes_from_v2_1"}) | {"changes_from_v2_2"}    # v2.3 moves M4 to the G4 runtime
PARENTS = {"protocol-v2.1.json": "protocol-v2.json", "protocol-v2.2.json": "protocol-v2.1.json",
           "protocol-v2.3.json": "protocol-v2.2.json"}
VLLM_LINES = ["k0_v1", "draft_model_k3", "k0_v2", "dflash_k7", "dspark_k7", "dspark_k7_triton", "dspark_k7_adaptive"]
K_OF_DRAFTER = {None: 0, "model_0p6b": 3, "dflash": 7, "dspark": 7}
SHA256 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture(params=["protocol-v2.json", "protocol-v2.1.json", "protocol-v2.2.json", "protocol-v2.3.json"])
def protocol(request):
    return json.loads((BENCHMARKS / request.param).read_text())


@pytest.fixture(params=["protocol-v2.2.json", "protocol-v2.3.json"])
def m4_protocol(request):
    return json.loads((BENCHMARKS / request.param).read_text())


def strictly_increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


def test_top_level_keys(protocol):
    if protocol["version"] == "2.2":
        assert set(protocol) == TOP_LEVEL_V22
    elif protocol["version"] == "2.3":
        assert set(protocol) == TOP_LEVEL_V23
    else:
        assert set(protocol) == (TOP_LEVEL | DERIVED if "derived_from" in protocol else TOP_LEVEL)


@pytest.mark.parametrize("child, parent", PARENTS.items())
def test_a_derived_protocol_pins_its_parent(child, parent):
    derived = json.loads((BENCHMARKS / child).read_text())["derived_from"]
    assert derived == {"file": f"benchmarks/{parent}", "sha256": hashlib.sha256((BENCHMARKS / parent).read_bytes()).hexdigest()}


def test_seed_relations(protocol):
    seeds = protocol["seeds"]
    dev = seeds["dev"]
    assert len(set(dev)) == 5 and all(isinstance(seed, int) for seed in dev)
    for name in set(seeds) - {"dev"}:
        assert set(seeds[name]) <= set(dev)


def test_concurrency_levels(protocol):
    levels = protocol["concurrency"]
    assert strictly_increasing(levels) and levels[0] == 1 and levels[-1] == 64


def test_lines_match_drafters(protocol):
    drafts = protocol["models"]["drafts"]
    for line in protocol["lines"].values():
        assert line["drafter"] is None or line["drafter"] in drafts
        assert line["k"] == K_OF_DRAFTER[line["drafter"]]


def test_frozen_protocol_pins_hashes(protocol):
    if not protocol["frozen"]:
        return
    for panel in protocol["panels"].values():
        assert SHA256.fullmatch(panel["sha256"] or "")
    g5 = protocol["gates"]["G5"]
    assert g5["commit"] and SHA256.fullmatch(g5["sha256"] or "")
    if "verify_schedule" in protocol:
        assert SHA256.fullmatch(protocol["verify_schedule"]["sha256"] or "")


def test_vllm_lines(protocol):
    if "external" in protocol:
        assert protocol["external"]["vllm"]["lines"] == VLLM_LINES


def test_cost_curve_sampling(protocol):
    curve = protocol["verify_schedule"]["cost_curve"] if "verify_schedule" in protocol else protocol["external"]["cost_curve"]
    assert curve["warmup"] >= 25 and curve["iters"] >= 100
    assert strictly_increasing(curve["batch"]) and strictly_increasing(curve["query_len"])


def test_every_line_divides_by_a_k0_of_its_own_tree(m4_protocol):
    lines = m4_protocol["lines"]
    for line in lines.values():
        k0 = lines[line["k0_line"]]
        assert line["tree"] in m4_protocol["trees"] and k0["k"] == 0 and k0["tree"] == line["tree"]
    assert set(m4_protocol["dev"]["lines"]) | set(m4_protocol["confirm"]["versions"]) <= set(lines)


def test_every_concurrency_has_a_row_of_the_cost_table(m4_protocol):
    # The bench runs with max_num_seqs = C, which the verify schedule needs among its batch sizes.
    assert set(m4_protocol["concurrency"]) <= set(m4_protocol["verify_schedule"]["cost_curve"]["batch"])
