import json
from pathlib import Path

import pytest
import torch

import gpu_import_stubs  # noqa: F401
from bench_spec import engine_kwargs
from nanovllm.layers import attention
from nanovllm.layers.attention import Attention
from nanovllm.layers.fused_ops import SparseDraftProbs, VerifySplitBuffers
from nanovllm.utils.context import reset_context, set_context


def test_draft_noncausal_gqa_packs_and_unpacks_identically(monkeypatch):
    calls = []

    def fake_flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs):
        calls.append((q.clone(), kwargs))
        return q * 2.0 + 1.0

    monkeypatch.setattr(attention, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache)
    try:
        batch_size, q_len, num_heads, num_kv_heads, head_dim = 3, 7, 32, 8, 16
        q = torch.arange(
            batch_size * q_len * num_heads * head_dim, dtype=torch.float32
        ).reshape(batch_size * q_len, num_heads, head_dim)
        k = torch.zeros(batch_size * q_len, num_kv_heads, head_dim)
        v = torch.zeros_like(k)

        set_context(
            False,
            context_lens=torch.full((batch_size,), 64, dtype=torch.int32),
            query_len_per_seq=q_len,
            causal=False,
            fused_kernels=True,
        )
        out = Attention(num_heads, head_dim, 0.125, num_kv_heads)(q, k, v)
        assert len(calls) == 1
        q_packed, kwargs = calls[0]
        assert q_packed.shape == (batch_size, q_len * (num_heads // num_kv_heads), num_kv_heads, head_dim)
        assert kwargs["causal"] is False
        torch.testing.assert_close(out, q * 2.0 + 1.0)
    finally:
        reset_context()


def test_verify_split_buffers_and_sparse_draft_probs_slice():
    bufs = VerifySplitBuffers.allocate(16, 8, 32, 8, 128, torch.bfloat16, torch.device("cpu"))
    sub = bufs.slice(4, 21)
    assert sub.q_packed.shape == (4, 32, 8, 128)
    assert sub.prefix_lens.shape == (4,)
    assert sub.token_row_qi.shape == (21,)

    sp = SparseDraftProbs(
        cand_ids=torch.zeros(8, 7, 84, dtype=torch.int32),
        cand_probs=torch.zeros(8, 7, 84, dtype=torch.float32),
        q_selected=torch.zeros(8, 7, dtype=torch.float32),
    )
    sp_sub = sp.slice(3)
    assert sp_sub.cand_ids.shape == (3, 7, 84)
    assert sp_sub.q_selected.shape == (3, 7)


def test_bench_spec_engine_kwargs_passes_fused_kernels(tmp_path: Path):
    target_dir = tmp_path / "Qwen3-8B"
    draft_dir = tmp_path / "dspark_qwen3_8b_block7-03326e504381"
    target_dir.mkdir()
    draft_dir.mkdir()
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 4096,
                "head_dim": 128,
            }
        )
    )
    protocol_path = Path(__file__).resolve().parents[1] / "benchmarks" / "protocol-exp-fused.json"
    protocol = json.loads(protocol_path.read_text())
    panel = [{"id": "0", "category": "test", "prompt_token_ids": [1, 2, 3, 4]}]

    _, _, kwargs_base = engine_kwargs(protocol, protocol["lines"]["dspark_var"], panel, 8, 17011, str(tmp_path))
    _, _, kwargs_fused = engine_kwargs(protocol, protocol["lines"]["dspark_var_fused"], panel, 8, 17011, str(tmp_path))
    assert "fused_kernels" not in kwargs_base
    assert kwargs_fused["fused_kernels"] is True
