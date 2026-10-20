"""G2-multi: the output distribution of k drafts per round through the production sample() and accept_random().

Each frozen Markov case of random_sampling_cases runs N independent rows round by round until they stop; the
frequency of every output sequence must lie within a Hoeffding bound of the target's exact probability.
"""
import math
from collections import Counter

import pytest
import torch

from nanovllm.layers.spec_sampler import accept_random, sample
from random_sampling_cases import CASES, K7_CASE, expected_distribution

TRIALS, FAILURE_PROBABILITY = 100_000, 0.001
MULTISTEP = [case for case in CASES if "p_rows" in case]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def frequency_bound(num_categories: int, num_trials: int, failure_probability: float) -> float:
    """√(ln(2M/δ) / 2N): with probability 1 − δ, no category's frequency is further than this from its probability."""
    return math.sqrt(math.log(2 * num_categories / failure_probability) / (2 * num_trials))


def simulate_rounds(case: dict, num_drafts: int, num_trials: int, generator: torch.Generator,
                    device: torch.device) -> tuple[list[tuple[int, ...]], float]:
    """Rows of the case run round by round until each stops; returns their outputs and the acceptance rate.

    Markov state: row 0 before any token, row x + 1 after token x. Draft i is sampled from q_rows at the state
    before it; p at position i uses that same state, and the bonus position the state after draft k.
    Tensors: p [n, k+1, V], q [n, k, V], drafts [n, k]. A round's output is the accepted drafts and the tail,
    cut after EOS (inclusive) or at max_tokens, which also stops the row.
    """
    p_rows = torch.tensor(case["p_rows"], dtype=torch.float32, device=device)
    q_rows = torch.tensor(case["q_rows"], dtype=torch.float32, device=device)
    eos, max_tokens = case["eos_token_id"], case["max_tokens"]
    outputs = [[] for _ in range(num_trials)]
    state = torch.zeros(num_trials, dtype=torch.long, device=device)
    active = list(range(num_trials))
    accepted = proposed = 0
    while active:
        index = torch.tensor(active, device=device)
        states, drafts = [state[index]], []
        for _ in range(num_drafts):
            drafts.append(sample(q_rows[states[-1]], generator))
            states.append(drafts[-1] + 1)
        p = torch.stack([p_rows[s] for s in states], dim=1)
        q = torch.stack([q_rows[s] for s in states[:-1]], dim=1)
        n_accept, tail = accept_random(p, q, torch.stack(drafts, dim=1), generator)
        accepted += int(n_accept.sum())
        proposed += len(active) * num_drafts
        rounds = torch.cat([torch.stack(drafts, dim=1), tail[:, None]], dim=1).tolist()
        still_active = []
        for row, tokens, count in zip(active, rounds, n_accept.tolist()):
            for token in tokens[:count] + [tokens[num_drafts]]:
                outputs[row].append(token)
                if token == eos or len(outputs[row]) == max_tokens:
                    break
            if outputs[row][-1] != eos and len(outputs[row]) < max_tokens:
                still_active.append(row)
                state[row] = outputs[row][-1] + 1
        active = still_active
    return [tuple(output) for output in outputs], accepted / proposed


def assert_matches_target(case: dict, outputs: list[tuple[int, ...]]) -> None:
    expected = expected_distribution(case)
    counts = Counter(outputs)
    assert set(counts) <= set(expected), f"{case['name']}: outputs the target cannot produce"
    bound = frequency_bound(len(expected), len(outputs), FAILURE_PROBABILITY)
    worst = max(abs(counts[sequence] / len(outputs) - probability) for sequence, probability in expected.items())
    assert worst <= bound, f"{case['name']}: frequency off by {worst:.5f} > {bound:.5f}"


def test_bound():
    assert frequency_bound(13, 100_000, 0.001) == pytest.approx(0.0071295, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case, num_drafts", [(case, 3) for case in MULTISTEP] + [(K7_CASE, 7)],
                         ids=lambda value: value["name"] if isinstance(value, dict) else f"k{value}")
def test_self_check_q_equals_p(case, num_drafts, device):
    """A perfect draft: every draft is accepted and the outputs are the target's. A failure here is in this test."""
    perfect = {**case, "q_rows": case["p_rows"]}
    generator = torch.Generator(device=device).manual_seed(7)
    outputs, acceptance = simulate_rounds(perfect, num_drafts, TRIALS, generator, torch.device(device))
    assert acceptance == 1.0
    assert_matches_target(perfect, outputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case, num_drafts", [(case, k) for case in MULTISTEP for k in (2, 3)] + [(K7_CASE, 7)],
                         ids=lambda value: value["name"] if isinstance(value, dict) else f"k{value}")
def test_gate(case, num_drafts, device):
    generator = torch.Generator(device=device).manual_seed(17011)
    outputs, _ = simulate_rounds(case, num_drafts, TRIALS, generator, torch.device(device))
    assert_matches_target(case, outputs)
