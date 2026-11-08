import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_vllm import LINES, engine_kwargs, panel_prompts, speculative_config  # noqa: E402

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "protocol-v2.json").read_text())
MODELS = "/models"
EXPECTED = {
    "draft_model_k3": ("draft_model", "/models/Qwen3-0.6B", 3),
    "dflash_k7": ("dspark", "/models/dflash_qwen3_8b_block7-9e44dbbb6cb6", 7),
    "dspark_k7": ("dspark", "/models/dspark_qwen3_8b_block7-03326e504381", 7),
    "dspark_k7_triton": ("dspark", "/models/dspark_qwen3_8b_block7-03326e504381", 7),
    "dspark_k7_adaptive": ("dspark", "/models/dspark_qwen3_8b_block7-03326e504381", 7),
}


def test_lines_match_protocol_and_each_runner_has_a_baseline():
    assert list(LINES) == PROTOCOL["external"]["vllm"]["lines"]
    baseline_runners = {line.runner for line in LINES.values() if line.method is None}
    assert {line.runner for line in LINES.values()} == baseline_runners


def test_speculative_configs():
    for line, (method, model, k) in EXPECTED.items():
        spec = speculative_config(line, PROTOCOL, MODELS)
        assert (spec["method"], spec["model"], spec["num_speculative_tokens"]) == (method, model, k)
        assert spec["draft_sample_method"] == "probabilistic"
    assert speculative_config("k0_v1", PROTOCOL, MODELS) is None and speculative_config("k0_v2", PROTOCOL, MODELS) is None


def test_engine_kwargs_hold_the_comparison_settings_fixed():
    for line in LINES:
        kwargs = engine_kwargs(line, 8, PROTOCOL, MODELS)
        assert kwargs["max_num_seqs"] == 8
        assert kwargs["enable_prefix_caching"] is False and kwargs["async_scheduling"] is False
        assert kwargs["kv_cache_memory_bytes"] == PROTOCOL["external"]["vllm"]["kv_cache_bytes"]
    adaptive = engine_kwargs("dspark_k7_adaptive", 8, PROTOCOL, MODELS)
    assert adaptive["speculative_config"].pop("enable_adaptive_verification") is True
    assert adaptive == engine_kwargs("dspark_k7_triton", 8, PROTOCOL, MODELS)
    assert adaptive["attention_backend"] == "TRITON_ATTN"


def test_batched_tokens_fit_every_slot_of_the_engine():
    assert engine_kwargs("k0_v2", 8, PROTOCOL, MODELS)["max_num_batched_tokens"] == PROTOCOL["max_num_batched_tokens"]
    assert engine_kwargs("k0_v2", 1, PROTOCOL, MODELS)["max_num_batched_tokens"] == PROTOCOL["max_model_len"]


def test_panel_prompts_use_whole_cycles():
    panel = [{"prompt_token_ids": [index]} for index in range(48)]
    assert [len(panel_prompts(panel, concurrency)) for concurrency in (1, 8, 32)] == [48, 48, 144]
    assert panel_prompts(panel, 32)[48] == [0]


def test_a_seed_replaces_the_protocols_external_seed():
    assert engine_kwargs("k0_v2", 8, PROTOCOL, MODELS)["seed"] == PROTOCOL["seeds"]["external"][0]
    assert engine_kwargs("k0_v2", 8, PROTOCOL, MODELS, seed=110017)["seed"] == 110017


# ---------------------------------------------------------------- protocol v2.7: M6 ③, the shared-prefix panels
V27 = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "protocol-v2.7.json").read_text())
SHARED_PANELS = ("S0", "S512", "S1024", "S2048")


def test_a_protocol_panel_gives_vllm_the_prompts_and_warm_up_bench_spec_takes():
    from bench_spec import ROOT as BENCH_ROOT, load_panel, warmup_entries
    from bench_vllm import protocol_panel_prompts
    for panel in SHARED_PANELS:
        for concurrency in V27["samehost_prefix"]["concurrency"]:
            measured, warmup = protocol_panel_prompts(V27, panel, concurrency, prefix_cache=True)
            entries = load_panel(BENCH_ROOT / V27["panels"][panel]["file"], concurrency, V27["panels"][panel]["take"])
            assert measured == [entry["prompt_token_ids"] for entry in entries]
            assert warmup == [entry["prompt_token_ids"] for entry in warmup_entries(V27, panel, entries, concurrency)]
            assert len(measured) == max(48, 4 * concurrency) and len(warmup) == concurrency


def test_a_protocol_panel_without_its_own_warm_up_cycles_as_before():
    from bench_vllm import protocol_panel_prompts
    measured, warmup = protocol_panel_prompts(V27, "D48", 32, prefix_cache=False)
    assert len(measured) == 144 and warmup == measured[:32]


def test_the_cache_on_refuses_a_cycled_panel():
    from bench_vllm import protocol_panel_prompts
    with pytest.raises(ValueError):
        protocol_panel_prompts(V27, "D48", 32, prefix_cache=True)


def test_shared_prefix_panels_hit_exactly_their_shared_part_at_vllms_block_size():
    """Every measured prompt's longest common prefix with any other prompt, the warm-up panel's included, rounds down
    to exactly the shared part at 16-token blocks: vLLM finds no hit beyond it."""
    root = Path(__file__).resolve().parents[1]
    for panel in SHARED_PANELS:
        spec = V27["panels"][panel]
        measured = [entry["prompt_token_ids"] for entry in json.loads((root / spec["file"]).read_text())]
        warmup = [entry["prompt_token_ids"] for entry in json.loads((root / V27["panels"][spec["warmup"]]["file"]).read_text())]
        prompts = warmup + measured
        order = sorted(range(len(prompts)), key=lambda index: prompts[index])
        longest = [0] * len(prompts)
        for a, b in zip(order, order[1:]):
            common = next((i for i, (x, y) in enumerate(zip(prompts[a], prompts[b])) if x != y), min(len(prompts[a]), len(prompts[b])))
            longest[a], longest[b] = max(longest[a], common), max(longest[b], common)
        assert {value // 16 * 16 for value in longest[len(warmup):]} == {spec["shared_tokens"]}


def test_the_prefix_cache_flag_reaches_the_engine_and_nothing_else_moves():
    for line in ("k0_v2", "dflash_k7", "dspark_k7"):
        on, off = engine_kwargs(line, 64, V27, MODELS, 17011, prefix_cache=True), engine_kwargs(line, 64, V27, MODELS, 17011)
        assert on.pop("enable_prefix_caching") is True and off.pop("enable_prefix_caching") is False
        assert on == off and on["kv_cache_memory_bytes"] == V27["external"]["vllm"]["kv_cache_bytes"]


def test_the_kv_pool_holds_the_longest_panel_at_c64_with_the_cache_off():
    """M5's logs: 32 GiB held 204,592 tokens for a drafting line, 233,008 for k0_v2."""
    root = Path(__file__).resolve().parents[1]
    longest = max(len(entry["prompt_token_ids"]) for name in ("S2048", "S2048w")
                  for entry in json.loads((root / V27["panels"][name]["file"]).read_text()))
    need = 64 * math.ceil((longest + V27["sampling"]["max_tokens"] + 8) / 16) * 16
    assert need == 247_808
    for tokens_at_32_gib in (204_592, 233_008):
        assert V27["external"]["vllm"]["kv_cache_bytes"] * tokens_at_32_gib / 2 ** 35 > need


def test_v2_7_samples_without_detokenizing_and_m5s_protocol_as_before():
    from bench_vllm import sampling_kwargs
    assert sampling_kwargs(V27, 512)["detokenize"] is False
    assert "detokenize" not in sampling_kwargs(json.loads((Path(__file__).resolve().parents[1] / "benchmarks" /
                                                           "protocol-v2.4.json").read_text()), 512)
