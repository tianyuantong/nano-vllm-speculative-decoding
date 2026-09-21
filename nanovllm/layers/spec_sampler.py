"""Sampling primitives shared by ordinary decoding and speculative verification.

Pure tensor code; runs on CPU or CUDA. A batch has B rows. Logits and probabilities
are [B, V] (one position per row) or [B, n, V] (n positions per row); per-row
parameters are [B] and broadcast over the middle dimensions.
"""
from dataclasses import dataclass

import torch

# Upstream clamps the Exp(1) noise of the exponential race to avoid dividing by zero.
EXPONENTIAL_NOISE_FLOOR = 1e-10
NEGATIVE_INFINITY = float("-inf")


@dataclass
class SamplingTensors:
    temperatures: torch.Tensor | None  # [B] fp32; None when every row is greedy
    top_k: torch.Tensor                # [B] int64; 0 disables top-k for that row
    top_p: torch.Tensor                # [B] fp32; 1.0 disables top-p for that row
    max_top_k: int                     # host copy of max(top_k); 0 means no row truncates
    has_top_p: bool                    # host: some row has top_p < 1


def sampling_tensors(seqs, device: torch.device) -> SamplingTensors:
    temperatures = [seq.temperature for seq in seqs]
    is_greedy = temperatures[0] == 0
    if any((temperature == 0) != is_greedy for temperature in temperatures):
        raise ValueError("mixed greedy and random sampling is not supported")
    top_k = [seq.top_k for seq in seqs]
    top_p = [seq.top_p for seq in seqs]
    return SamplingTensors(
        temperatures=None if is_greedy else _to_device(temperatures, torch.float32, device),
        top_k=_to_device(top_k, torch.int64, device),
        top_p=_to_device(top_p, torch.float32, device),
        max_top_k=max(top_k),
        has_top_p=any(value < 1.0 for value in top_p),
    )


def _to_device(values: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


def _per_row(values: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a [B] tensor so it broadcasts against a tensor with `ndim` dims and B rows."""
    return values.reshape(values.shape[0], *([1] * (ndim - 1)))


def probs_from_logits(logits: torch.Tensor, sampling: SamplingTensors) -> torch.Tensor:
    """fp32 probabilities after temperature, top-k and top-p; greedy batches have none."""
    assert sampling.temperatures is not None
    scaled = logits.float() / _per_row(sampling.temperatures, logits.ndim)
    return torch.softmax(truncate_logits(scaled, sampling), dim=-1)


def truncate_logits(scaled: torch.Tensor, sampling: SamplingTensors) -> torch.Tensor:
    """-inf outside each row's top-k, then outside the top-p nucleus of those candidates.

    HuggingFace order: temperature (already applied), top-k, top-p. Ties at the k-th
    value are kept. Rows with top_k == 0 are returned unchanged.
    """
    if sampling.max_top_k == 0:
        return scaled
    ndim = scaled.ndim
    top_k = _per_row(sampling.top_k, ndim)
    effective_k = top_k.clamp(min=1)                      # rows with top_k == 0 are masked out below
    num_candidates = min(sampling.max_top_k, scaled.shape[-1])
    candidates = scaled.topk(num_candidates, dim=-1).values          # [..., K], descending
    ranks = torch.arange(num_candidates, device=scaled.device)
    candidates = candidates.masked_fill(ranks >= effective_k, NEGATIVE_INFINITY)
    k_index = (effective_k - 1).clamp(max=num_candidates - 1).expand(*candidates.shape[:-1], 1)
    threshold = candidates.gather(-1, k_index)                        # k-th largest value per row
    if sampling.has_top_p:
        nucleus = torch.softmax(candidates, dim=-1)                   # over the row's own top-k
        mass_before = nucleus.cumsum(dim=-1) - nucleus
        num_kept = (mass_before < _per_row(sampling.top_p, ndim)).sum(dim=-1, keepdim=True)
        p_threshold = candidates.gather(-1, num_kept - 1)
        threshold = torch.maximum(threshold, p_threshold)
    threshold = torch.where(top_k > 0, threshold, torch.full_like(threshold, NEGATIVE_INFINITY))
    return scaled.masked_fill(scaled < threshold, NEGATIVE_INFINITY)


def sample(weights: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Exponential race: argmax(w / E) with E ~ Exp(1) picks index i with probability w_i / sum(w)."""
    noise = torch.empty_like(weights).exponential_(1.0, generator=generator).clamp_min_(EXPONENTIAL_NOISE_FLOOR)
    return (weights / noise).argmax(dim=-1)


def accept_random(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor,
                  generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection sampling of k drafts per row.

    p: [B, k+1, V] target probabilities at the k draft positions and the bonus position.
    q: [B, k, V] draft probabilities the drafts were sampled from.
    drafts: [B, k] int64.
    Returns n_accept [B] in 0..k and tail [B]: the token following the accepted prefix,
    drawn from normalize(max(p - q, 0)) at the first rejected position, or from the
    bonus row when every draft was accepted. The output distribution equals the target's.
    """
    batch_size, num_drafts = drafts.shape
    draft_index = drafts.unsqueeze(-1)
    p_draft = p[:, :num_drafts].gather(-1, draft_index).squeeze(-1)      # [B, k]
    q_draft = q.gather(-1, draft_index).squeeze(-1)                      # [B, k]
    uniforms = torch.rand(batch_size, num_drafts, device=p.device, generator=generator)
    accepted = uniforms * q_draft < p_draft                              # u < min(1, p/q) without dividing
    n_accept = accepted.long().cumprod(dim=-1).sum(dim=-1)              # length of the accepted prefix
    rows = torch.arange(batch_size, device=p.device)
    p_next = p[rows, n_accept]                                          # [B, V] first rejected or bonus position
    q_next = q[rows, n_accept.clamp(max=num_drafts - 1)]
    residual = (p_next - q_next).clamp_(min=0)
    is_bonus = (n_accept == num_drafts).unsqueeze(-1)
    tail = sample(torch.where(is_bonus, p_next, residual), generator)  # the race needs no normalization
    return n_accept, tail


def accept_greedy(target_logits: torch.Tensor, drafts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy verification: a draft is accepted iff it equals the target's argmax at its position."""
    target_tokens = target_logits.argmax(dim=-1)                         # [B, k+1]
    num_drafts = drafts.shape[1]
    accepted = target_tokens[:, :num_drafts] == drafts
    n_accept = accepted.long().cumprod(dim=-1).sum(dim=-1)
    rows = torch.arange(drafts.shape[0], device=drafts.device)
    return n_accept, target_tokens[rows, n_accept]
