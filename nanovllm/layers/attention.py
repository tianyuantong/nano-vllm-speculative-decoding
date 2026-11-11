import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.fused_ops import fused_verify_paged_attention
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        *,
        kv_stored: bool = False,
    ):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if not kv_stored and k is not None and v is not None and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.verify_attn_buffers is not None:
                return fused_verify_paged_attention(
                    q, k_cache, v_cache, context.block_tables, context.slot_mapping,
                    self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
                    context.verify_attn_buffers, attn_with_kvcache_fn=flash_attn_with_kvcache,
                )
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode: query_len_per_seq queries per sequence, sequence-major rows
            q_len = context.query_len_per_seq
            if context.fused_kernels and not context.causal and q_len > 1 and self.num_heads > self.num_kv_heads:
                group = self.num_heads // self.num_kv_heads
                q_packed = (q.view(-1, q_len, self.num_kv_heads, group, self.head_dim)
                            .transpose(2, 3)
                            .contiguous()
                            .view(-1, q_len * group, self.num_kv_heads, self.head_dim))
                o_packed = flash_attn_with_kvcache(
                    q_packed, k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                    softmax_scale=self.scale, causal=False,
                )
                return (o_packed.view(-1, q_len, group, self.num_kv_heads, self.head_dim)
                        .transpose(2, 3)
                        .contiguous()
                        .view(-1, self.num_heads, self.head_dim))
            o = flash_attn_with_kvcache(q.view(-1, q_len, self.num_heads, self.head_dim),
                                        k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=context.causal)
            o = o.view(-1, self.num_heads, self.head_dim)
        return o
