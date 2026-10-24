import json
import sys
from pathlib import Path

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


def test_panel_prompts_use_whole_cycles():
    panel = [{"prompt_token_ids": [index]} for index in range(48)]
    assert [len(panel_prompts(panel, concurrency)) for concurrency in (1, 8, 32)] == [48, 48, 144]
    assert panel_prompts(panel, 32)[48] == [0]
