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
    accept_random,
    probs_from_logits,
    sample,
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
