"""Shared tensor primitives for the planned standard random B/N/S0/S1 path.

Not yet connected to LLMEngine. CUDA acceptance must exercise these functions,
not a test-only sampler. Values stay on device until require_valid_boundary().
Callers accumulate every returned invalid mask and check it before publishing
tokens. Metadata/index safety is enforced before a potentially unsafe access.
"""

from dataclasses import dataclass

import torch


# Input normalization check only; never an acceptance margin or denominator fix.
NORMALIZATION_ATOL = 1e-5


@dataclass(frozen=True)
class Probabilities:
    values: torch.Tensor  # [batch, vocab]; FP32 p/q, FP64 residual weights
    mass: torch.Tensor  # [batch], FP64; effective distribution is values / mass
    invalid: torch.Tensor  # [batch], bool, on the same device


@dataclass(frozen=True)
class Sample:
    token_ids: torch.Tensor
    invalid: torch.Tensor


@dataclass(frozen=True)
class Acceptance:
    accepted: torch.Tensor
    invalid: torch.Tensor


def _matrix(values: torch.Tensor) -> None:
    # Shape/dtype/device metadata is available on the host without reading CUDA.
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("expected a nonempty [batch, vocab] tensor")
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("probability weights must be FP32 or FP64")


def _vector(values, batch, device, dtype) -> None:
    if values.shape != (batch,) or values.device != device or values.dtype != dtype:
        raise ValueError("vector shape, device or dtype does not match probabilities")


def _pair(target: Probabilities, draft: Probabilities) -> None:
    _matrix(target.values)
    _matrix(draft.values)
    if target.values.shape != draft.values.shape or target.values.device != draft.values.device:
        raise ValueError("target and draft must have the same layout and device")
    for value in (target, draft):
        _vector(value.invalid, value.values.shape[0], value.values.device, torch.bool)
        _vector(value.mass, value.values.shape[0], value.values.device, torch.float64)


def _placeholder(values: torch.Tensor) -> torch.Tensor:
    # An in-vocabulary token for safe termination only; never a published fallback.
    result = torch.zeros_like(values)
    result[:, 0] = 1.0
    return result


def from_probs(values: torch.Tensor) -> Probabilities:
    """Borrow near-normalized weights, retaining their effective normalizer.

    Exponential sampling is invariant to row scale. Acceptance and residual must
    use that same normalized distribution, not assume an approximately-one mass
    equals one. FP64 arithmetic reduces this error; it is not exact real math.
    The caller owns this storage and must not modify it until verification and
    any residual draw finish; frozen dataclasses do not freeze tensor storage.
    """
    _matrix(values)
    valid = torch.isfinite(values).all(dim=-1) & (values >= 0).all(dim=-1)
    mass = values.double().sum(dim=-1)
    valid = valid & torch.isfinite(mass) & ((mass - 1.0).abs() <= NORMALIZATION_ATOL)
    return Probabilities(values, mass, ~valid)


def from_logits(logits: torch.Tensor, temperatures: torch.Tensor) -> Probabilities:
    """Allocate a distinct FP32 probability tensor; never overwrite logits/q."""
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("expected nonempty floating [batch, vocab] logits")
    _vector(temperatures, logits.shape[0], logits.device, torch.float32)
    bad_temperature = ~torch.isfinite(temperatures) | (temperatures <= 0)
    safe_temperature = torch.where(bad_temperature, torch.ones_like(temperatures), temperatures)
    probabilities = from_probs(torch.softmax(logits.float() / safe_temperature[:, None], dim=-1))
    return Probabilities(probabilities.values, probabilities.mass, probabilities.invalid | bad_temperature)


def draw(probabilities: Probabilities, *, generator: torch.Generator) -> Sample:
    """Exponential-race sample on device, preserving the original probabilities.

    Each call consumes a full [batch, vocab] noise tensor, including masked rows.
    No value-dependent compaction, multinomial dispatch or host scalar read.
    """
    values = probabilities.values
    _matrix(values)
    _vector(probabilities.invalid, values.shape[0], values.device, torch.bool)
    safe = torch.where(probabilities.invalid[:, None], _placeholder(values), values)
    noise = torch.empty_like(values).exponential_(1.0, generator=generator)
    bad_noise = ~torch.isfinite(noise) | (noise <= 0)
    safe_noise = torch.where(bad_noise, torch.ones_like(noise), noise)
    scores = safe / safe_noise  # out of place: q remains available to verification
    invalid = probabilities.invalid | bad_noise.any(dim=-1) | ~torch.isfinite(scores).all(dim=-1)
    return Sample(scores.argmax(dim=-1), invalid)


def accept(
    target: Probabilities,
    draft: Probabilities,
    token_ids: torch.Tensor,
    uniforms: torch.Tensor,
) -> Acceptance:
    """Per-position decision; the scheduler must keep only the accepted prefix.

    Uniforms are explicit FP64 [0,1) values from the independent acceptance role.
    Invalid token IDs are flagged and clamped before gather, never after it.
    """
    _pair(target, draft)
    batch, vocab = target.values.shape
    _vector(token_ids, batch, target.values.device, torch.int64)
    _vector(uniforms, batch, target.values.device, torch.float64)
    bad_ids = (token_ids < 0) | (token_ids >= vocab)
    indices = token_ids.clamp(0, vocab - 1)[:, None]
    p = target.values.gather(1, indices).squeeze(1).double()
    q = draft.values.gather(1, indices).squeeze(1).double()
    p = p / torch.where(target.mass > 0, target.mass, torch.ones_like(target.mass))
    q = q / torch.where(draft.mass > 0, draft.mass, torch.ones_like(draft.mass))
    invalid = target.invalid | draft.invalid | bad_ids | (q <= 0)
    invalid = invalid | ~torch.isfinite(uniforms) | (uniforms < 0) | (uniforms >= 1)
    # This placeholder denominator only makes invalid rows safe to terminate.
    # Their invalid flag is retained and no token may be committed from them.
    ratio = p / torch.where(q > 0, q, torch.ones_like(q))
    threshold = torch.minimum(ratio, torch.ones_like(ratio))
    return Acceptance((uniforms < threshold) & ~invalid, invalid)


def residual(
    target: Probabilities,
    draft: Probabilities,
    rejected: torch.Tensor,
) -> Probabilities:
    """Residual rows for actual first rejections; inactive rows use token 0.

    A fixed row layout permits sampling without a device-to-host nonzero().
    Inactive draws are discarded but still consume RNG. The caller must retain
    all earlier error masks even when no rejection occurs in a particular row.
    """
    _pair(target, draft)
    _vector(rejected, target.values.shape[0], target.values.device, torch.bool)
    p_mass = torch.where(target.mass > 0, target.mass, torch.ones_like(target.mass))
    q_mass = torch.where(draft.mass > 0, draft.mass, torch.ones_like(draft.mass))
    weights = (target.values.double() / p_mass[:, None] - draft.values.double() / q_mass[:, None]).clamp_min(0.0)
    mass = weights.sum(dim=-1)
    good_mass = torch.isfinite(mass) & (mass > 0)
    invalid = target.invalid | draft.invalid | (rejected & ~good_mass)
    denominator = torch.where(good_mass, mass, torch.ones_like(mass))
    values = weights / denominator[:, None]
    safe_active = rejected & ~invalid
    values = torch.where(safe_active[:, None], values, _placeholder(values))
    checked = from_probs(values)
    return Probabilities(checked.values, checked.mass, checked.invalid | invalid)


def require_valid_boundary(invalid: torch.Tensor) -> None:
    """The intentional host read: stage boundary only, always before commit.

    Accumulate invalid masks from proposal, accept and residual/draw first.
    Do not call from each inner draft step. This is not a KV/index preflight.
    """
    if invalid.dtype != torch.bool or invalid.ndim != 1 or invalid.numel() == 0:
        raise ValueError("expected a nonempty per-request invalid mask")
    if invalid.any().item():
        raise FloatingPointError("invalid probability/sample state; abort before commit")
