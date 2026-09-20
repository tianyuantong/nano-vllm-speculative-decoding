import json

import pytest

from nanovllm.config import Config

TINY_QWEN3 = {
    "model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
    "max_position_embeddings": 1024, "torch_dtype": "bfloat16", "tie_word_embeddings": False,
}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(TINY_QWEN3))
    return str(tmp_path)


def test_defaults_have_no_draft(model_dir):
    config = Config(model_dir)
    assert config.draft_model is None and config.num_speculative_tokens == 0


def test_draft_requires_k_prefix_cache_off_and_budgets(model_dir):
    kwargs = dict(draft_model=model_dir, num_speculative_tokens=3, enable_prefix_cache=False,
                  kv_cache_memory_bytes=1 << 20, draft_kv_cache_memory_bytes=1 << 20)
    Config(model_dir, **kwargs)
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "num_speculative_tokens": 0})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "enable_prefix_cache": True})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "draft_kv_cache_memory_bytes": None})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "tensor_parallel_size": 2})


def test_speculative_tokens_without_draft_is_rejected(model_dir):
    with pytest.raises(ValueError):
        Config(model_dir, num_speculative_tokens=2)
