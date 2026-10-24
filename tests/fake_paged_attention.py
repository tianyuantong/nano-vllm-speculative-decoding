"""CPU stand-ins for nano-vllm's paged attention, for tests.

store_kvcache writes K/V rows into the paged cache by slot. attention_forward replaces Attention.forward in
decode mode: it gathers each sequence's keys through its block table up to its context length and runs SDPA.
Rows are sequence-major, query_len_per_seq per sequence; with causal, query j of a sequence with context length L
sees keys 0 .. L - query_len_per_seq + j, which is FlashAttention's rule for queries at the end of the context.
"""
import torch
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    rows = slot_mapping >= 0
    k_cache.view(-1, *key.shape[1:])[slot_mapping[rows].long()] = key[rows]
    v_cache.view(-1, *value.shape[1:])[slot_mapping[rows].long()] = value[rows]


def attention_forward(self, q, k, v):
    context = get_context()
    assert not context.is_prefill
    store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
    block_size, queries_per_seq = self.k_cache.shape[1], context.query_len_per_seq
    all_keys = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)
    all_values = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)
    group = self.num_heads // self.num_kv_heads
    outputs = []
    for row, length in enumerate(context.context_lens.tolist()):
        positions = torch.arange(length)
        slots = context.block_tables[row, positions // block_size].long() * block_size + positions % block_size
        keys = all_keys[slots].repeat_interleave(group, dim=1)          # [L, heads, D]
        values = all_values[slots].repeat_interleave(group, dim=1)
        queries = q[row * queries_per_seq:(row + 1) * queries_per_seq]  # [n, heads, D]
        mask = None
        if context.causal:
            mask = positions[None, :] <= (length - queries_per_seq + torch.arange(queries_per_seq))[:, None]
        out = F.scaled_dot_product_attention(queries.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1),
                                             attn_mask=mask, scale=self.scale)
        outputs.append(out.transpose(0, 1))
    return torch.cat(outputs)
