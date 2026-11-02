import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from random_sampling_cases import CASES  # noqa: E402

from nanovllm.layers.spec_sampler import (  # noqa: E402
    accept_greedy,
    accept_greedy_varlen,
    accept_random,
    accept_random_varlen,
    probs_from_logits,
    sample,
    sampling_per_token,
    sampling_tensors,
    truncate_logits,
)

CPU = torch.device("cpu")
SINGLE_STEP_CASES = [case for case in CASES if "p" in case]
NUM_DRAWS = 200_000


def rows(*params):
    """Build SamplingTensors from (temperature, top_k, top_p) triples."""
    seqs = [SimpleNamespace(temperature=t, top_k=k, top_p=p) for t, k, p in params]
    return sampling_tensors(seqs, CPU)


def test_sampling_tensors_greedy_batch_has_no_temperatures():
    sampling = rows((0, 0, 1.0), (0, 0, 1.0))
    assert sampling.temperatures is None
    assert sampling.max_top_k == 0
    assert sampling.has_top_p is False


def test_sampling_tensors_rejects_mixed_greedy_and_random():
    with pytest.raises(ValueError):
        rows((0, 0, 1.0), (0.7, 0, 1.0))


def test_temperature_scales_logits_before_softmax():
    sampling = rows((0.5, 0, 1.0))
    logits = torch.tensor([[1.0, 0.0]])
    probs = probs_from_logits(logits, sampling)
    expected = torch.softmax(torch.tensor([[2.0, 0.0]]), dim=-1)
    torch.testing.assert_close(probs, expected)


def test_top_k_keeps_only_the_k_largest():
    sampling = rows((1.0, 2, 1.0))
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    probs = probs_from_logits(logits, sampling)
    assert probs[0, 0] == 0 and probs[0, 3] == 0
    torch.testing.assert_close(probs[0, 1:3], torch.softmax(torch.tensor([3.0, 2.0]), dim=-1))


def test_top_k_keeps_ties_at_the_threshold():
    sampling = rows((1.0, 1, 1.0))
    logits = torch.tensor([[2.0, 2.0, 1.0, 0.0]])
    probs = probs_from_logits(logits, sampling)
    torch.testing.assert_close(probs, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))


def test_top_p_keeps_the_smallest_prefix_reaching_p():
    sampling = rows((1.0, 4, 0.5))
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
    probs = probs_from_logits(weights.log().unsqueeze(0), sampling)
    torch.testing.assert_close(probs, torch.tensor([[4 / 7, 3 / 7, 0.0, 0.0]]))


def test_top_p_is_computed_within_the_top_k_candidates():
    sampling = rows((1.0, 2, 0.99))
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
    probs = probs_from_logits(weights.log().unsqueeze(0), sampling)
    torch.testing.assert_close(probs, torch.tensor([[4 / 7, 3 / 7, 0.0, 0.0]]))


def test_rows_without_truncation_are_untouched_in_a_mixed_batch():
    sampling = rows((1.0, 0, 1.0), (1.0, 1, 1.0))
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    probs = probs_from_logits(logits, sampling)
    torch.testing.assert_close(probs[0], torch.softmax(logits[0], dim=-1))
    torch.testing.assert_close(probs[1], torch.tensor([0.0, 0.0, 1.0]))


def test_truncation_broadcasts_over_a_middle_positions_dimension():
    sampling = rows((1.0, 1, 1.0))
    logits = torch.tensor([[[1.0, 5.0, 2.0], [9.0, 5.0, 2.0]]])   # [B=1, n=2, V=3]
    truncated = truncate_logits(logits, sampling)
    assert truncated[0, 0].tolist() == [-math.inf, 5.0, -math.inf]
    assert truncated[0, 1].tolist() == [9.0, -math.inf, -math.inf]


def test_sample_frequencies_follow_the_weights():
    generator = torch.Generator().manual_seed(0)
    weights = torch.tensor([0.1, 0.6, 0.3]).expand(200_000, 3)
    counts = torch.bincount(sample(weights, generator), minlength=3).float() / 200_000
    torch.testing.assert_close(counts, torch.tensor([0.1, 0.6, 0.3]), atol=0.005, rtol=0)


def test_sample_never_picks_zero_weight():
    generator = torch.Generator().manual_seed(1)
    weights = torch.tensor([0.0, 1.0, 0.0]).expand(10_000, 3)
    assert (sample(weights, generator) == 1).all()


@pytest.mark.parametrize("case", SINGLE_STEP_CASES, ids=[case["name"] for case in SINGLE_STEP_CASES])
def test_accept_random_emits_the_target_distribution(case):
    generator = torch.Generator().manual_seed(1234)
    p_row = torch.tensor(case["p"], dtype=torch.float32)
    q_row = torch.tensor(case["q"], dtype=torch.float32)
    vocab = p_row.numel()
    drafts = sample(q_row.expand(NUM_DRAWS, vocab), generator).unsqueeze(1)          # [N, 1] ~ q
    p = p_row.expand(NUM_DRAWS, 2, vocab)                                             # verified position + bonus
    q = q_row.expand(NUM_DRAWS, 1, vocab)
    n_accept, tail = accept_random(p, q, drafts, generator)
    emitted = torch.where(n_accept == 1, drafts[:, 0], tail)
    frequencies = torch.bincount(emitted, minlength=vocab).float() / NUM_DRAWS
    total_variation = 0.5 * (frequencies - p_row).abs().sum().item()
    assert total_variation < 0.01, (case["name"], frequencies.tolist())


def test_accept_random_all_accepted_uses_the_bonus_row():
    generator = torch.Generator().manual_seed(0)
    vocab = 4
    p = torch.zeros(1, 3, vocab)
    p[0, :2] = torch.tensor([0.25, 0.25, 0.25, 0.25])
    p[0, 2, 3] = 1.0                                              # bonus position is a point mass on token 3
    q = p[:, :2].clone()                                          # identical proposal: always accepted
    drafts = torch.tensor([[0, 1]])
    n_accept, tail = accept_random(p, q, drafts, generator)
    assert n_accept.tolist() == [2]
    assert tail.tolist() == [3]


def test_accept_random_rejects_where_target_has_zero_mass():
    generator = torch.Generator().manual_seed(0)
    p = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])                  # target never emits token 1 at position 0
    q = torch.tensor([[[0.0, 1.0]]])
    drafts = torch.tensor([[1]])
    n_accept, tail = accept_random(p, q, drafts, generator)
    assert n_accept.tolist() == [0]
    assert tail.tolist() == [0]                                   # residual (p - q)+ is a point mass on token 0


@pytest.mark.parametrize("num_drafts", [1, 2, 4])
@pytest.mark.parametrize("first_mismatch", [0, 1, 2, 3, 4])
def test_accept_greedy_counts_the_matching_prefix(num_drafts, first_mismatch):
    if first_mismatch > num_drafts:
        pytest.skip("mismatch position beyond the draft length")
    vocab = 16
    drafts = torch.arange(1, num_drafts + 1).unsqueeze(0)                     # [1, k] = 1..k
    target_tokens = list(range(1, num_drafts + 2))                            # target agrees: 1..k, bonus k+1
    if first_mismatch < num_drafts:
        target_tokens[first_mismatch] = 9
    logits = torch.full((1, num_drafts + 1, vocab), -10.0)
    logits[0, torch.arange(num_drafts + 1), torch.tensor(target_tokens)] = 10.0
    n_accept, tail = accept_greedy(logits, drafts)
    expected_count = min(first_mismatch, num_drafts)
    assert n_accept.tolist() == [expected_count]
    assert tail.tolist() == [target_tokens[expected_count]]


def test_accept_greedy_is_per_row():
    logits = torch.full((2, 2, 4), -10.0)
    logits[0, 0, 1] = logits[0, 1, 2] = 10.0        # row 0 agrees with drafts [1] -> bonus 2
    logits[1, 0, 3] = logits[1, 1, 2] = 10.0        # row 1 disagrees at position 0 -> tail 3
    n_accept, tail = accept_greedy(logits, torch.tensor([[1], [1]]))
    assert n_accept.tolist() == [1, 0]
    assert tail.tolist() == [2, 3]


# ---------------------------------------------------------------- variable-length verification
def m3_accept_random(p, q, drafts, generator):
    """accept_random's body as it was at 7bf0466, before the variable-length rewrite."""
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


def m3_accept_greedy(target_logits, drafts):
    """accept_greedy's body as it was at 7bf0466."""
    target_tokens = target_logits.argmax(dim=-1)                         # [B, k+1]
    num_drafts = drafts.shape[1]
    accepted = target_tokens[:, :num_drafts] == drafts
    n_accept = accepted.long().cumprod(dim=-1).sum(dim=-1)
    rows = torch.arange(drafts.shape[0], device=drafts.device)
    return n_accept, target_tokens[rows, n_accept]


@pytest.mark.parametrize("seed", range(5))
def test_every_row_verifying_all_drafts_reproduces_the_m3_rule(seed):
    # The fixed-length path now runs through accept_random_varlen: same accepts, same tails, same generator state.
    torch.manual_seed(seed)
    batch_size, num_drafts, vocab_size = 6, 7, 5
    p = torch.softmax(torch.randn(batch_size, num_drafts + 1, vocab_size), dim=-1)
    q = torch.softmax(torch.randn(batch_size, num_drafts, vocab_size), dim=-1)
    drafts = torch.randint(vocab_size, (batch_size, num_drafts))
    ours, theirs = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    for got, expected in zip(accept_random(p, q, drafts, ours), m3_accept_random(p, q, drafts, theirs)):
        assert torch.equal(got, expected)
    assert torch.equal(ours.get_state(), theirs.get_state())
    for got, expected in zip(accept_greedy(p.log(), drafts), m3_accept_greedy(p.log(), drafts)):
        assert torch.equal(got, expected)


def varlen_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three rows verifying 1, 2 and 4 positions; every row's drafts are 1, 2, 3 and every p, q a point mass, so a
    verified draft is always accepted. Flat p: row 0 at 0, row 1 at 1-2, row 2 at 3-6; each row's last position,
    its bonus, holds token 3, 2, 0. Row 1's bonus is its unverified draft d2, which must not count as accepted."""
    point = torch.eye(4)
    p = point[[3, 1, 2, 1, 2, 3, 0]]
    drafts = torch.tensor([[1, 2, 3]] * 3)
    return p, drafts, torch.tensor([1, 2, 4])


def test_a_row_accepts_at_most_its_verified_drafts():
    p, drafts, verified_lens = varlen_case()
    q = torch.eye(4)[drafts]                                     # [3, 3, 4]: point masses on the drafts
    n_accept, tail = accept_random_varlen(p, q, drafts, verified_lens, torch.Generator().manual_seed(0))
    assert (n_accept.tolist(), tail.tolist()) == ([0, 1, 3], [3, 2, 0])


def test_greedy_accepts_at_most_the_verified_drafts():
    p, drafts, verified_lens = varlen_case()
    n_accept, tail = accept_greedy_varlen(p, drafts, verified_lens)
    assert (n_accept.tolist(), tail.tolist()) == ([0, 1, 3], [3, 2, 0])


def test_per_token_parameters_give_each_row_its_own_probabilities():
    # Rows verifying every position: the flat probabilities are the per-row ones, position by position.
    sampling = rows((0.7, 3, 0.9), (1.0, 0, 1.0))
    logits = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(3))
    verified_lens = torch.tensor([4, 4])
    flat = probs_from_logits(logits.flatten(0, 1), sampling_per_token(sampling, verified_lens, 8))
    assert torch.equal(flat, probs_from_logits(logits, sampling).flatten(0, 1))
