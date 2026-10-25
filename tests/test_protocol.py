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
VLLM_LINES = ["k0_v1", "draft_model_k3", "k0_v2", "dflash_k7", "dspark_k7", "dspark_k7_triton", "dspark_k7_adaptive"]
K_OF_DRAFTER = {None: 0, "model_0p6b": 3, "dflash": 7, "dspark": 7}
SHA256 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture(params=["protocol-v2.json", "protocol-v2.1.json"])
def protocol(request):
    return json.loads((BENCHMARKS / request.param).read_text())


def strictly_increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


def test_top_level_keys(protocol):
    assert set(protocol) == (TOP_LEVEL | DERIVED if "derived_from" in protocol else TOP_LEVEL)


def test_v21_derives_from_the_frozen_v2():
    derived = json.loads((BENCHMARKS / "protocol-v2.1.json").read_text())["derived_from"]
    assert derived["sha256"] == hashlib.sha256((BENCHMARKS / "protocol-v2.json").read_bytes()).hexdigest()


def test_seed_relations(protocol):
    seeds = protocol["seeds"]
    dev = seeds["dev"]
    assert len(set(dev)) == 5 and all(isinstance(seed, int) for seed in dev)
    for name in ("confirm", "gate_g5", "external"):
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


def test_vllm_lines(protocol):
    assert protocol["external"]["vllm"]["lines"] == VLLM_LINES


def test_cost_curve_sampling(protocol):
    curve = protocol["external"]["cost_curve"]
    assert curve["warmup"] >= 25 and curve["iters"] >= 100
    assert strictly_increasing(curve["batch"]) and strictly_increasing(curve["query_len"])
