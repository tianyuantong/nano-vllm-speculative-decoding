"""Hand-written Triton fused kernels for speculative decoding (experimental, Config.fused_kernels).

Two operator groups:
  1. Prefix-split GQA paged attention + fused causal tail & online LSE merge for multi-token verification.
  2. Sparse candidate draft probability representation + fused candidate truncation/sampling for DSpark's
     Markov head + fused sparse target acceptance and chunked exponential residual race.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.layers.spec_sampler import EXPONENTIAL_NOISE_FLOOR, SamplingTensors

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only test environments without triton installed
    class _TritonStub:
        @staticmethod
        def jit(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

        @staticmethod
        def next_power_of_2(n: int) -> int:
            p = 1
            while p < n:
                p <<= 1
            return p

    class _TLStub:
        constexpr = object()

    triton = _TritonStub()  # type: ignore[assignment]
    tl = _TLStub()  # type: ignore[assignment]

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


MIN_VERIFY_SPLIT_TOKENS = 96
MAX_SPARSE_CANDIDATES = 128
VOCAB_CHUNK_SIZE = 1024


# ==============================================================================
# Group 1 (Node B1): Prefix-Split GQA Verify Attention + Fused Causal Tail LSE
# ==============================================================================


@dataclass(slots=True)
class VerifySplitBuffers:
    q_packed: torch.Tensor       # [num_rows, max_q_len * group, num_kv_heads, head_dim]
    prefix_lens: torch.Tensor    # [num_rows] int32
    token_row_qi: torch.Tensor   # [num_tokens] int32: (row << 8) | qi
    max_q_len: int

    @classmethod
    def allocate(
        cls,
        max_rows: int,
        max_q_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> VerifySplitBuffers:
        group = num_heads // num_kv_heads
        return cls(
            q_packed=torch.zeros(
                (max_rows, max_q_len * group, num_kv_heads, head_dim), dtype=dtype, device=device
            ),
            prefix_lens=torch.zeros((max_rows,), dtype=torch.int32, device=device),
            token_row_qi=torch.zeros((max_rows * max_q_len,), dtype=torch.int32, device=device),
            max_q_len=max_q_len,
        )

    def slice(self, num_rows: int, num_tokens: int) -> VerifySplitBuffers:
        return VerifySplitBuffers(
            q_packed=self.q_packed[:num_rows],
            prefix_lens=self.prefix_lens[:num_rows],
            token_row_qi=self.token_row_qi[:num_tokens],
            max_q_len=self.max_q_len,
        )


@triton.jit
def _build_verify_split_meta_kernel(
    cu_q_ptr,
    cu_k_ptr,
    prefix_lens_ptr,
    token_row_qi_ptr,
    MAX_Q_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    q_start = tl.load(cu_q_ptr + row)
    q_end = tl.load(cu_q_ptr + row + 1)
    k_start = tl.load(cu_k_ptr + row)
    k_end = tl.load(cu_k_ptr + row + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    tl.store(prefix_lens_ptr + row, k_len - q_len)
    for qi in range(MAX_Q_LEN):
        if qi < q_len:
            tl.store(token_row_qi_ptr + q_start + qi, (row << 8) | qi)


def prepare_verify_split_meta(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    buffers: VerifySplitBuffers,
) -> None:
    """Populate prefix_lens [num_rows] and token_row_qi [num_tokens] once before layer 0."""
    num_rows = buffers.prefix_lens.shape[0]
    if num_rows == 0:
        return
    _build_verify_split_meta_kernel[(num_rows,)](
        cu_seqlens_q,
        cu_seqlens_k,
        buffers.prefix_lens,
        buffers.token_row_qi,
        MAX_Q_LEN=buffers.max_q_len,
    )


@triton.jit
def _pack_verify_q_kernel(
    q_ptr,
    q_stride_t,
    token_row_qi_ptr,
    qp_ptr,
    qp_stride_r,
    qp_stride_pq,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    meta = tl.load(token_row_qi_ptr + token_idx)
    row = (meta >> 8).to(tl.int64)
    qi = meta & 0xFF
    offs_d = tl.arange(0, HEAD_DIM)

    q_token_ptr = q_ptr + token_idx * q_stride_t
    qp_row_ptr = qp_ptr + row * qp_stride_r
    for g in tl.static_range(GROUP):
        head = kv_head * GROUP + g
        pq = qi * GROUP + g
        vec = tl.load(q_token_ptr + head * HEAD_DIM + offs_d)
        tl.store(qp_row_ptr + pq * qp_stride_pq + kv_head * HEAD_DIM + offs_d, vec)


@triton.jit
def _fused_verify_tail_lse_kernel(
    q_ptr,
    q_stride_t,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    token_row_qi_ptr,
    prefix_lens_ptr,
    out1_ptr,
    out1_stride_r,
    out1_stride_pq,
    out1_stride_kvh,
    lse1_ptr,
    lse1_stride_r,
    lse1_stride_kvh,
    lse1_stride_pq,
    o_ptr,
    o_stride_t,
    scale,
    KV_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_Q_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    out_dtype = o_ptr.dtype.element_ty
    offs_d = tl.arange(0, HEAD_DIM)

    slot_self = tl.load(slot_mapping_ptr + token_idx)
    if slot_self < 0:
        tl.store(o_ptr + token_idx * o_stride_t + head * HEAD_DIM + offs_d, tl.zeros([HEAD_DIM], dtype=out_dtype))
        return

    meta = tl.load(token_row_qi_ptr + token_idx)
    row = (meta >> 8).to(tl.int64)
    qi = meta & 0xFF
    kv_head = head // GROUP
    g = head % GROUP
    pq = qi * GROUP + g

    q_vec = tl.load(q_ptr + token_idx * q_stride_t + head * HEAD_DIM + offs_d).to(tl.float32) * scale
    prefix_len = tl.load(prefix_lens_ptr + row)

    if prefix_len > 0:
        m = tl.load(lse1_ptr + row * lse1_stride_r + kv_head * lse1_stride_kvh + pq * lse1_stride_pq).to(tl.float32)
        l = 1.0
        acc = tl.load(
            out1_ptr + row * out1_stride_r + pq * out1_stride_pq + kv_head * out1_stride_kvh + offs_d
        ).to(tl.float32)
    else:
        m = -1.0e30
        l = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    row_token_start = token_idx - qi
    for j in range(MAX_Q_LEN):
        if j <= qi:
            slot_j = tl.load(slot_mapping_ptr + row_token_start + j).to(tl.int64)
            kv_base = slot_j * KV_SIZE + kv_head * HEAD_DIM
            k_j = tl.load(k_cache_ptr + kv_base + offs_d).to(tl.float32)
            v_j = tl.load(v_cache_ptr + kv_base + offs_d).to(tl.float32)
            score = tl.sum(q_vec * k_j, axis=0)
            m_new = tl.maximum(m, score)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(score - m_new)
            l = l * alpha + beta
            acc = acc * alpha + beta * v_j
            m = m_new

    tl.store(o_ptr + token_idx * o_stride_t + head * HEAD_DIM + offs_d, (acc / l).to(out_dtype))


def fused_verify_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    buffers: VerifySplitBuffers,
    attn_with_kvcache_fn=None,
) -> torch.Tensor:
    """Prefix-split GQA paged attention + fused causal tail & online LSE merge."""
    n_tokens = q.shape[0]
    o_out = torch.empty((n_tokens, num_heads, head_dim), dtype=q.dtype, device=q.device)
    if n_tokens == 0:
        return o_out

    group = num_heads // num_kv_heads
    kv_size = num_kv_heads * head_dim
    _pack_verify_q_kernel[(n_tokens, num_kv_heads)](
        q,
        q.stride(0),
        buffers.token_row_qi,
        buffers.q_packed,
        buffers.q_packed.stride(0),
        buffers.q_packed.stride(1),
        GROUP=group,
        HEAD_DIM=head_dim,
    )

    fa_fn = attn_with_kvcache_fn or flash_attn_with_kvcache
    out1, lse1 = fa_fn(
        buffers.q_packed,
        k_cache,
        v_cache,
        cache_seqlens=buffers.prefix_lens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=False,
        return_softmax_lse=True,
    )

    _fused_verify_tail_lse_kernel[(n_tokens, num_heads)](
        q,
        q.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        buffers.token_row_qi,
        buffers.prefix_lens,
        out1,
        out1.stride(0),
        out1.stride(1),
        out1.stride(2),
        lse1,
        lse1.stride(0),
        lse1.stride(1),
        lse1.stride(2),
        o_out,
        o_out.stride(0),
        scale,
        KV_SIZE=kv_size,
        GROUP=group,
        HEAD_DIM=head_dim,
        MAX_Q_LEN=buffers.max_q_len,
    )
    return o_out


# ==============================================================================
# Group 2 (Node A): Sparse Candidate Draft Sampling + Fused Target Acceptance
# ==============================================================================


@dataclass(slots=True)
class SparseDraftProbs:
    """Sparse representation of DSpark's truncated draft distribution per (row, draft_step)."""
    cand_ids: torch.Tensor     # [B, k, C] int32
    cand_probs: torch.Tensor   # [B, k, C] fp32
    q_selected: torch.Tensor   # [B, k] fp32

    def slice(self, batch_size: int) -> SparseDraftProbs:
        return SparseDraftProbs(
            cand_ids=self.cand_ids[:batch_size],
            cand_probs=self.cand_probs[:batch_size],
            q_selected=self.q_selected[:batch_size],
        )


@triton.jit
def _fused_candidate_sample_kernel(
    values_ptr,
    ids_ptr,
    temperatures_ptr,
    top_k_ptr,
    top_p_ptr,
    noise_ptr,
    noise_stride_b,
    cand_ids_ptr,
    cand_ids_stride_b,
    cand_ids_stride_k,
    cand_probs_ptr,
    cand_probs_stride_b,
    cand_probs_stride_k,
    drafts_ptr,
    drafts_stride_b,
    q_selected_ptr,
    q_selected_stride_b,
    next_prev_ptr,
    row_idx,
    NUM_CANDS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_C)
    valid = offs < NUM_CANDS

    temp = tl.load(temperatures_ptr + b).to(tl.float32)
    raw_k = tl.load(top_k_ptr + b).to(tl.int32)
    eff_k = tl.minimum(tl.maximum(raw_k, 1), NUM_CANDS)

    val = tl.load(values_ptr + b * NUM_CANDS + offs, mask=valid, other=-float("inf")).to(tl.float32)
    tok_id = tl.load(ids_ptr + b * NUM_CANDS + offs, mask=valid, other=0).to(tl.int64)
    scaled = tl.where(valid, val / temp, -float("inf"))

    k_thresh = tl.sum(tl.where(offs == eff_k - 1, scaled, 0.0), axis=0)
    thresh = k_thresh

    if HAS_TOP_P:
        p_val = tl.load(top_p_ptr + b).to(tl.float32)
        if p_val < 1.0:
            cand = tl.where(offs < eff_k, scaled, -float("inf"))
            c_max = tl.max(cand, axis=0)
            c_exp = tl.where(offs < eff_k, tl.exp(cand - c_max), 0.0)
            nucleus = c_exp / tl.sum(c_exp, axis=0)
            mass_before = tl.cumsum(nucleus, axis=0) - nucleus
            kept_mask = (offs < eff_k) & (mass_before < p_val)
            num_kept = tl.maximum(tl.sum(kept_mask.to(tl.int32), axis=0), 1)
            p_thresh = tl.sum(tl.where(offs == num_kept - 1, scaled, 0.0), axis=0)
            thresh = tl.maximum(k_thresh, p_thresh)

    keep = valid & (scaled >= thresh)
    trunc = tl.where(keep, scaled, -float("inf"))
    m = tl.max(trunc, axis=0)
    probs = tl.where(keep, tl.exp(trunc - m), 0.0)
    probs = probs / tl.sum(probs, axis=0)

    noise = tl.load(noise_ptr + b * noise_stride_b + offs, mask=valid, other=1.0).to(tl.float32)
    race_scores = tl.where(valid, probs / noise, -float("inf"))
    win_idx = tl.argmax(race_scores, axis=0)
    win_tok = tl.sum(tl.where(offs == win_idx, tok_id, 0), axis=0)
    win_prob = tl.sum(tl.where(offs == win_idx, probs, 0.0), axis=0)

    cid_base = cand_ids_ptr + b * cand_ids_stride_b + row_idx * cand_ids_stride_k
    cprob_base = cand_probs_ptr + b * cand_probs_stride_b + row_idx * cand_probs_stride_k
    tl.store(cid_base + offs, tok_id.to(tl.int32), mask=valid)
    tl.store(cprob_base + offs, probs, mask=valid)
    tl.store(drafts_ptr + b * drafts_stride_b + row_idx, win_tok)
    tl.store(q_selected_ptr + b * q_selected_stride_b + row_idx, win_prob)
    tl.store(next_prev_ptr + b, win_tok)


def sample_markov_candidate_row(
    row_logits: torch.Tensor,
    candidates: int,
    sampling: SamplingTensors,
    noise_row: torch.Tensor,
    row_idx: int,
    drafts: torch.Tensor,
    sparse_probs: SparseDraftProbs,
    next_prev: torch.Tensor,
) -> torch.Tensor:
    """Top-C selection + fused temperature/top-k/top-p/softmax/exponential-race for one Markov draft row."""
    batch_size = row_logits.shape[0]
    values, ids = row_logits.topk(candidates, dim=-1)
    block_c = triton.next_power_of_2(candidates)
    _fused_candidate_sample_kernel[(batch_size,)](
        values,
        ids,
        sampling.temperatures,
        sampling.top_k,
        sampling.top_p,
        noise_row,
        noise_row.stride(0),
        sparse_probs.cand_ids,
        sparse_probs.cand_ids.stride(0),
        sparse_probs.cand_ids.stride(1),
        sparse_probs.cand_probs,
        sparse_probs.cand_probs.stride(0),
        sparse_probs.cand_probs.stride(1),
        drafts,
        drafts.stride(0),
        sparse_probs.q_selected,
        sparse_probs.q_selected.stride(0),
        next_prev,
        row_idx,
        NUM_CANDS=candidates,
        BLOCK_C=block_c,
        HAS_TOP_P=sampling.has_top_p,
    )
    return next_prev


@triton.jit
def _build_accept_token_owner_kernel(
    verified_lens_ptr,
    starts_ptr,
    token_owner_ptr,
    MAX_VLEN: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(starts_ptr + b)
    vlen = tl.load(verified_lens_ptr + b)
    for i in range(MAX_VLEN):
        if i < vlen:
            tl.store(token_owner_ptr + start + i, b)


@triton.jit
def _target_threshold_kernel(
    top_vals_ptr,
    top_vals_stride_t,
    token_owner_ptr,
    temperatures_ptr,
    top_k_ptr,
    top_p_ptr,
    cut_ptr,
    max_s_ptr,
    MAX_TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    b = tl.load(token_owner_ptr + t).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    valid = offs < MAX_TOP_K

    temp = tl.load(temperatures_ptr + b).to(tl.float32)
    raw_k = tl.load(top_k_ptr + b).to(tl.int32)
    eff_k = tl.minimum(tl.maximum(raw_k, 1), MAX_TOP_K)

    vals = tl.load(top_vals_ptr + t * top_vals_stride_t + offs, mask=valid, other=-float("inf")).to(tl.float32)
    scaled = tl.where(valid, vals / temp, -float("inf"))
    max_s = tl.max(scaled, axis=0)
    k_thresh = tl.sum(tl.where(offs == eff_k - 1, scaled, 0.0), axis=0)
    thresh = k_thresh

    if HAS_TOP_P:
        p_val = tl.load(top_p_ptr + b).to(tl.float32)
        if p_val < 1.0:
            cand = tl.where(offs < eff_k, scaled, -float("inf"))
            c_exp = tl.where(offs < eff_k, tl.exp(cand - max_s), 0.0)
            nucleus = c_exp / tl.sum(c_exp, axis=0)
            mass_before = tl.cumsum(nucleus, axis=0) - nucleus
            kept_mask = (offs < eff_k) & (mass_before < p_val)
            num_kept = tl.maximum(tl.sum(kept_mask.to(tl.int32), axis=0), 1)
            p_thresh = tl.sum(tl.where(offs == num_kept - 1, scaled, 0.0), axis=0)
            thresh = tl.maximum(k_thresh, p_thresh)

    tl.store(cut_ptr + t, thresh)
    tl.store(max_s_ptr + t, max_s)


@triton.jit
def _target_partition_chunk_kernel(
    logits_ptr,
    logits_stride_t,
    token_owner_ptr,
    temperatures_ptr,
    cut_ptr,
    max_s_ptr,
    chunk_sums_ptr,
    NUM_CHUNKS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1)
    cols = chunk_id * BLOCK_V + tl.arange(0, BLOCK_V)
    valid = cols < VOCAB_SIZE

    b = tl.load(token_owner_ptr + t).to(tl.int64)
    temp = tl.load(temperatures_ptr + b).to(tl.float32)
    cut_t = tl.load(cut_ptr + t).to(tl.float32)
    max_s_t = tl.load(max_s_ptr + t).to(tl.float32)

    raw = tl.load(logits_ptr + t * logits_stride_t + cols, mask=valid, other=-float("inf")).to(tl.float32)
    scaled = raw / temp
    keep = valid & (scaled >= cut_t)
    part = tl.sum(tl.where(keep, tl.exp(scaled - max_s_t), 0.0), axis=0)
    tl.store(chunk_sums_ptr + t * NUM_CHUNKS + chunk_id, part)


@triton.jit
def _prefix_accept_kernel(
    logits_ptr,
    logits_stride_t,
    starts_ptr,
    verified_lens_ptr,
    temperatures_ptr,
    cut_ptr,
    max_s_ptr,
    z_ptr,
    drafts_ptr,
    drafts_stride_b,
    q_selected_ptr,
    q_selected_stride_b,
    uniforms_ptr,
    uniforms_stride_b,
    n_accept_ptr,
    t_next_ptr,
    NUM_DRAFTS: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    start = tl.load(starts_ptr + b).to(tl.int64)
    vlen = tl.load(verified_lens_ptr + b).to(tl.int64)
    max_drafts = vlen - 1
    temp = tl.load(temperatures_ptr + b).to(tl.float32)

    n_acc = start - start
    still_accepting = 1
    for j in range(NUM_DRAFTS):
        if (j < max_drafts) & (still_accepting == 1):
            pos = start + j
            d_tok = tl.load(drafts_ptr + b * drafts_stride_b + j).to(tl.int64)
            q_prob = tl.load(q_selected_ptr + b * q_selected_stride_b + j).to(tl.float32)
            u = tl.load(uniforms_ptr + b * uniforms_stride_b + j).to(tl.float32)

            raw = tl.load(logits_ptr + pos * logits_stride_t + d_tok).to(tl.float32)
            scaled = raw / temp
            cut_p = tl.load(cut_ptr + pos).to(tl.float32)
            max_s_p = tl.load(max_s_ptr + pos).to(tl.float32)
            z_p = tl.load(z_ptr + pos).to(tl.float32)
            p_prob = tl.where(scaled >= cut_p, tl.exp(scaled - max_s_p) / z_p, 0.0)
            if u * q_prob < p_prob:
                n_acc += 1
            else:
                still_accepting = 0

    tl.store(n_accept_ptr + b, n_acc)
    tl.store(t_next_ptr + b, start + n_acc)


@triton.jit
def _residual_race_chunk_kernel(
    logits_ptr,
    logits_stride_t,
    verified_lens_ptr,
    n_accept_ptr,
    t_next_ptr,
    temperatures_ptr,
    cut_ptr,
    max_s_ptr,
    z_ptr,
    cand_ids_ptr,
    cand_ids_stride_b,
    cand_ids_stride_k,
    cand_probs_ptr,
    cand_probs_stride_b,
    cand_probs_stride_k,
    race_seeds_ptr,
    chunk_scores_ptr,
    chunk_ids_ptr,
    noise_floor,
    NUM_CANDS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1)
    chunk_start = chunk_id * BLOCK_V
    cols = chunk_start + tl.arange(0, BLOCK_V)
    valid = cols < VOCAB_SIZE

    pos = tl.load(t_next_ptr + b).to(tl.int64)
    temp = tl.load(temperatures_ptr + b).to(tl.float32)
    cut_p = tl.load(cut_ptr + pos).to(tl.float32)

    raw = tl.load(logits_ptr + pos * logits_stride_t + cols, mask=valid, other=-float("inf")).to(tl.float32)
    scaled = raw / temp
    keep = valid & (scaled >= cut_p)

    # Any token with max(p - q, 0) > 0 or p > 0 must have p > 0 (scaled >= cut_p).
    if tl.max(keep.to(tl.int32), axis=0) == 0:
        tl.store(chunk_scores_ptr + b * NUM_CHUNKS + chunk_id, -1.0)
        tl.store(chunk_ids_ptr + b * NUM_CHUNKS + chunk_id, chunk_start)
        return

    max_s_p = tl.load(max_s_ptr + pos).to(tl.float32)
    z_p = tl.load(z_ptr + pos).to(tl.float32)
    w = tl.where(keep, tl.exp(scaled - max_s_p) / z_p, 0.0)

    n_acc = tl.load(n_accept_ptr + b).to(tl.int64)
    vlen = tl.load(verified_lens_ptr + b).to(tl.int64)
    if n_acc < vlen - 1:
        cid_base = cand_ids_ptr + b * cand_ids_stride_b + n_acc * cand_ids_stride_k
        cprob_base = cand_probs_ptr + b * cand_probs_stride_b + n_acc * cand_probs_stride_k
        for c in range(NUM_CANDS):
            cid = tl.load(cid_base + c)
            if (cid >= chunk_start) & (cid < chunk_start + BLOCK_V):
                cprob = tl.load(cprob_base + c).to(tl.float32)
                w = tl.where(cols == cid, tl.maximum(w - cprob, 0.0), w)

    seed = tl.load(race_seeds_ptr + b).to(tl.int32)
    u = tl.rand(seed, cols)
    exp_noise = tl.maximum(-tl.log(tl.maximum(1.0 - u, noise_floor)), noise_floor)
    scores = tl.where(w > 0.0, w / exp_noise, -1.0)

    local_idx = tl.argmax(scores, axis=0)
    local_best = tl.max(scores, axis=0)
    tl.store(chunk_scores_ptr + b * NUM_CHUNKS + chunk_id, local_best)
    tl.store(chunk_ids_ptr + b * NUM_CHUNKS + chunk_id, chunk_start + local_idx)


def accept_sparse_varlen(
    target_logits: torch.Tensor,
    sparse_probs: SparseDraftProbs,
    drafts: torch.Tensor,
    sampling: SamplingTensors,
    verified_lens: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lossless rejection sampling directly over flat [T, V] target logits and SparseDraftProbs."""
    assert sampling.temperatures is not None and sampling.all_top_k
    num_tokens, vocab_size = target_logits.shape
    batch_size, num_drafts = drafts.shape
    num_cands = sparse_probs.cand_ids.shape[-1]
    device = target_logits.device

    starts = verified_lens.cumsum(0) - verified_lens
    token_owner = torch.empty((num_tokens,), dtype=torch.int32, device=device)
    _build_accept_token_owner_kernel[(batch_size,)](
        verified_lens,
        starts,
        token_owner,
        MAX_VLEN=num_drafts + 1,
    )

    max_top_k = min(sampling.max_top_k, vocab_size)
    top_vals = target_logits.topk(max_top_k, dim=-1).values
    cut = torch.empty((num_tokens,), dtype=torch.float32, device=device)
    max_s = torch.empty((num_tokens,), dtype=torch.float32, device=device)
    _target_threshold_kernel[(num_tokens,)](
        top_vals,
        top_vals.stride(0),
        token_owner,
        sampling.temperatures,
        sampling.top_k,
        sampling.top_p,
        cut,
        max_s,
        MAX_TOP_K=max_top_k,
        BLOCK_K=triton.next_power_of_2(max_top_k),
        HAS_TOP_P=sampling.has_top_p,
    )

    num_chunks = (vocab_size + VOCAB_CHUNK_SIZE - 1) // VOCAB_CHUNK_SIZE
    chunk_sums = torch.empty((num_tokens, num_chunks), dtype=torch.float32, device=device)
    _target_partition_chunk_kernel[(num_tokens, num_chunks)](
        target_logits,
        target_logits.stride(0),
        token_owner,
        sampling.temperatures,
        cut,
        max_s,
        chunk_sums,
        NUM_CHUNKS=num_chunks,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=VOCAB_CHUNK_SIZE,
    )
    z_part = chunk_sums.sum(dim=-1)

    uniforms = torch.rand((batch_size, num_drafts), dtype=torch.float32, device=device, generator=generator)
    race_seeds = torch.empty((batch_size,), dtype=torch.int64, device=device).random_(0, 2**31 - 1, generator=generator)
    n_accept = torch.empty((batch_size,), dtype=torch.int64, device=device)
    t_next = torch.empty((batch_size,), dtype=torch.int64, device=device)
    _prefix_accept_kernel[(batch_size,)](
        target_logits,
        target_logits.stride(0),
        starts,
        verified_lens,
        sampling.temperatures,
        cut,
        max_s,
        z_part,
        drafts,
        drafts.stride(0),
        sparse_probs.q_selected,
        sparse_probs.q_selected.stride(0),
        uniforms,
        uniforms.stride(0),
        n_accept,
        t_next,
        NUM_DRAFTS=num_drafts,
    )

    chunk_scores = torch.empty((batch_size, num_chunks), dtype=torch.float32, device=device)
    chunk_ids = torch.empty((batch_size, num_chunks), dtype=torch.int64, device=device)
    _residual_race_chunk_kernel[(batch_size, num_chunks)](
        target_logits,
        target_logits.stride(0),
        verified_lens,
        n_accept,
        t_next,
        sampling.temperatures,
        cut,
        max_s,
        z_part,
        sparse_probs.cand_ids,
        sparse_probs.cand_ids.stride(0),
        sparse_probs.cand_ids.stride(1),
        sparse_probs.cand_probs,
        sparse_probs.cand_probs.stride(0),
        sparse_probs.cand_probs.stride(1),
        race_seeds,
        chunk_scores,
        chunk_ids,
        EXPONENTIAL_NOISE_FLOOR,
        NUM_CANDS=num_cands,
        NUM_CHUNKS=num_chunks,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=VOCAB_CHUNK_SIZE,
    )
    win_chunk = chunk_scores.argmax(dim=-1, keepdim=True)
    tail = chunk_ids.gather(-1, win_chunk).squeeze(-1)
    return n_accept, tail
