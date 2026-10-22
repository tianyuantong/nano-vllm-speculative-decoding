import pytest
import torch

import gpu_import_stubs  # noqa: F401  (before the attention import)
from nanovllm.layers import attention
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context

HEADS, HEAD_DIM = 4, 8


class Recorder:
    """Stands in for a flash-attn function: remembers the call and returns q unchanged."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append((q.clone(), kwargs))
        return q.clone()


@pytest.fixture
def kernels(monkeypatch):
    decode, prefill = Recorder(), Recorder()
    monkeypatch.setattr(attention, "flash_attn_with_kvcache", decode)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", prefill)
    yield decode, prefill
    reset_context()


def rows(n):
    """q, k, v of n tokens; every element of q's row r equals r."""
    q = torch.arange(n, dtype=torch.float32).view(n, 1, 1).expand(n, HEADS, HEAD_DIM).contiguous()
    return q, torch.zeros(n, HEADS, HEAD_DIM), torch.zeros(n, HEADS, HEAD_DIM)


def test_default_decode_is_unchanged(kernels):
    decode, _ = kernels
    set_context(False, context_lens=torch.ones(3, dtype=torch.int32))
    Attention(HEADS, HEAD_DIM, 1.0, HEADS)(*rows(3))
    q, kwargs = decode.calls[0]
    assert q.shape == (3, 1, HEADS, HEAD_DIM) and kwargs["causal"] is True


def test_block_mode(kernels):
    decode, _ = kernels
    set_context(False, context_lens=torch.ones(3, dtype=torch.int32), query_len_per_seq=7, causal=False)
    out = Attention(HEADS, HEAD_DIM, 1.0, HEADS)(*rows(21))
    q, kwargs = decode.calls[0]
    assert q.shape == (3, 7, HEADS, HEAD_DIM) and kwargs["causal"] is False
    assert out.shape == (21, HEADS, HEAD_DIM)


def test_block_rows_are_sequence_major(kernels):
    decode, _ = kernels
    set_context(False, context_lens=torch.ones(3, dtype=torch.int32), query_len_per_seq=7, causal=False)
    Attention(HEADS, HEAD_DIM, 1.0, HEADS)(*rows(21))
    q, _ = decode.calls[0]
    assert all(q[b, j, 0, 0].item() == b * 7 + j for b in range(3) for j in range(7))


def test_prefill_is_unaffected(kernels):
    _, prefill = kernels
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    set_context(True, cu_seqlens, cu_seqlens, 5, 5)
    Attention(HEADS, HEAD_DIM, 1.0, HEADS)(*rows(5))
    assert prefill.calls[0][1]["causal"] is True
