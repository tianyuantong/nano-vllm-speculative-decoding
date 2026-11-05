"""G2-multi: the output distribution of k drafts per round through the production sample() and accept_random(),
through accept_random_varlen() when each row verifies a length of its own, and with the production verify schedule
choosing the lengths of rows scheduled together.

Each frozen Markov case of random_sampling_cases runs N independent rows round by round until they stop; the
frequency of every output sequence must lie within a Hoeffding bound of the target's exact probability.
"""
import itertools
import math
from collections import Counter

import numpy as np
import pytest
import torch

from nanovllm.engine.verify_schedule import VerifySchedule, survival_probabilities
from nanovllm.layers.spec_sampler import accept_random, accept_random_varlen, sample
from random_sampling_cases import CASES, K7_CASE, expected_distribution

TRIALS, FAILURE_PROBABILITY = 100_000, 0.001
MULTISTEP = [case for case in CASES if "p_rows" in case]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def frequency_bound(num_categories: int, num_trials: int, failure_probability: float) -> float:
    """√(ln(2M/δ) / 2N): with probability 1 − δ, no category's frequency is further than this from its probability."""
    return math.sqrt(math.log(2 * num_categories / failure_probability) / (2 * num_trials))


def simulate_rounds(case: dict, num_drafts: int, num_trials: int, generator: torch.Generator, device: torch.device,
                    choose_lengths=None) -> tuple[list[tuple[int, ...]], float]:
    """Rows of the case run round by round until each stops; returns their outputs and the acceptance rate over
    the verified drafts (nan when none was verified).

    Markov state: row 0 before any token, row x + 1 after token x. Draft i is sampled from q_rows at the state
    before it; p at position i uses that same state, and the bonus position the state after draft k.
    Tensors: p [n, k+1, V], q [n, k, V], drafts [n, k]. A round's output is the accepted drafts and the tail,
    cut after EOS (inclusive) or at max_tokens, which also stops the row. With choose_lengths(active rows, states
    before each draft), every round each row verifies the 1..k+1 positions it gives, through accept_random_varlen
    on p's verified positions; without it, all k drafts through accept_random.
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
        drafts = torch.stack(drafts, dim=1)
        if choose_lengths is None:
            n_accept, tail = accept_random(p, q, drafts, generator)
            proposed += len(active) * num_drafts
        else:
            verified_lens = choose_lengths(active, states)
            verified = torch.arange(num_drafts + 1, device=device) < verified_lens.unsqueeze(1)
            n_accept, tail = accept_random_varlen(p[verified], q, drafts, verified_lens, generator)
            proposed += int(verified_lens.sum()) - len(active)
        accepted += int(n_accept.sum())
        rounds = torch.cat([drafts, tail[:, None]], dim=1).tolist()
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
    return [tuple(output) for output in outputs], accepted / proposed if proposed else math.nan


def random_lengths(lengths_generator: torch.Generator, num_drafts: int):
    """A length per row, uniform in 1..k+1, from a generator of its own: apart from the drafts."""
    return lambda active, states: torch.randint(1, num_drafts + 2, (len(active),), generator=lengths_generator,
                                                device=states[0].device)


def scheduled_lengths(schedule, confidence_logits_by_state: torch.Tensor, batch_size: int):
    """Rows in batches of batch_size consecutive indices, each batch's active rows scheduled together. Draft i's
    confidence logit is confidence_logits_by_state at the state before it, so it reads the token before the draft,
    as DSpark's confidence head does."""
    def choose(active, states):
        logits = confidence_logits_by_state[torch.stack(states[:-1], dim=1)].cpu().numpy()       # [n, k]
        lengths = []
        for _, members in itertools.groupby(range(len(active)), key=lambda position: active[position] // batch_size):
            members = list(members)
            lengths += schedule(logits[members[0]:members[-1] + 1])
        return torch.tensor(lengths, device=states[0].device)
    return choose


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case, num_drafts", [(case, 3) for case in MULTISTEP] + [(K7_CASE, 7)],
                         ids=lambda value: value["name"] if isinstance(value, dict) else f"k{value}")
def test_self_check_varlen_q_equals_p(case, num_drafts, device):
    """A perfect draft with lengths of their own: every verified draft is accepted. A failure here is in this test."""
    perfect = {**case, "q_rows": case["p_rows"]}
    generator = torch.Generator(device=device).manual_seed(7)
    lengths_generator = torch.Generator(device=device).manual_seed(8)
    outputs, acceptance = simulate_rounds(perfect, num_drafts, TRIALS, generator, torch.device(device),
                                          random_lengths(lengths_generator, num_drafts))
    assert acceptance == 1.0
    assert_matches_target(perfect, outputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case, num_drafts", [(case, k) for case in MULTISTEP for k in (2, 3)] + [(K7_CASE, 7)],
                         ids=lambda value: value["name"] if isinstance(value, dict) else f"k{value}")
def test_gate_varlen(case, num_drafts, device):
    generator = torch.Generator(device=device).manual_seed(17011)
    lengths_generator = torch.Generator(device=device).manual_seed(110017)
    outputs, _ = simulate_rounds(case, num_drafts, TRIALS, generator, torch.device(device),
                                 random_lengths(lengths_generator, num_drafts))
    assert_matches_target(case, outputs)


# ---------------------------------------------------------------- the verify schedule in the loop
SCHEDULED_BATCH_SIZE = 4
APPENDIX_A_TRIALS = 20_000
# DSpark's appendix A: target (0.7, 0.3) and draft (0.5, 0.5) at every state, one token out. Draft 1's confidence is
# 0.8; draft 2's is 0.9 after A and about 0 after B. Round times 1, 2 and 1 / 0.45 for 1, 2 and 3 tokens.
APPENDIX_A = {"name": "appendix_a", "p_rows": [[0.7, 0.3]] * 3, "q_rows": [[0.5, 0.5]] * 3,
              "max_tokens": 1, "eos_token_id": None}
APPENDIX_A_LOGITS = [math.log(0.8 / 0.2), math.log(0.9 / 0.1), -30.0]
APPENDIX_A_SCHEDULE = VerifySchedule([1], [[1.0, 2.0, 1 / 0.45]], [1.0, 1.0], max_num_seqs=1)


def batch_schedule(num_drafts: int) -> VerifySchedule:
    """Batches of 4 rows, fewer once rows stop, a batch of 3 reading the table of 4: 3 ms more per token over 10,
    12 and 16 ms for 1, 2 and 4 rows."""
    return VerifySchedule([1, 2, 4], [[base + 3 * batch_size * extra for extra in range(num_drafts + 1)]
                                      for batch_size, base in [(1, 10.0), (2, 12.0), (4, 16.0)]],
                          [1.0] * num_drafts, max_num_seqs=SCHEDULED_BATCH_SIZE)


def assert_pairs_match_target(case: dict, pairs: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
    """Each pair's outputs jointly follow the product of the target's distributions: scheduling a pair together
    must not correlate what its rows emit."""
    expected = expected_distribution(case)
    joint = {(a, b): p_a * p_b for a, p_a in expected.items() for b, p_b in expected.items()}
    counts = Counter(pairs)
    bound = frequency_bound(len(joint), len(pairs), FAILURE_PROBABILITY)
    worst = max(abs(counts[pair] / len(pairs) - probability) for pair, probability in joint.items())
    assert worst <= bound, f"{case['name']}: pair frequency off by {worst:.5f} > {bound:.5f}"


def global_maximum(logits: np.ndarray) -> list[int]:
    """Appendix A's biased schedule: the admitted count with the highest throughput over all counts. Whether draft
    1 is verified then depends on draft 2's confidence, which reads draft 1."""
    survival = survival_probabilities(logits, np.ones(logits.shape[1]))
    rows, steps = np.indices(survival.shape)
    order = np.lexsort((rows.ravel(), steps.ravel(), -survival.ravel()))
    expected = len(survival) + np.concatenate(([0.0], np.cumsum(survival.ravel()[order])))
    round_ms = APPENDIX_A_SCHEDULE.round_ms(len(survival))[len(survival):len(survival) + len(expected)]
    admitted = int(np.argmax(expected / round_ms))
    return (1 + np.bincount(rows.ravel()[order[:admitted]], minlength=len(survival))).tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case, num_drafts", [(case, 3) for case in MULTISTEP] + [(K7_CASE, 7)],
                         ids=lambda value: value["name"] if isinstance(value, dict) else f"k{value}")
def test_gate_scheduled(case, num_drafts, device):
    """Rows scheduled together in batches of 4 by the production verify schedule, on confidences that read the
    token before each draft: the outputs of every slot of a batch follow the target, and so do the first two
    slots' jointly, where the pair outcomes are few enough for the bound to mean something."""
    generator = torch.Generator(device=device).manual_seed(17011)
    by_state = 2.0 * torch.sin(torch.arange(len(case["p_rows"]), dtype=torch.float32, device=device) + 1)
    schedule = batch_schedule(num_drafts).verify_lengths
    outputs, _ = simulate_rounds(case, num_drafts, TRIALS, generator, torch.device(device),
                                 scheduled_lengths(schedule, by_state, SCHEDULED_BATCH_SIZE))
    slots = [outputs[slot::SCHEDULED_BATCH_SIZE] for slot in range(SCHEDULED_BATCH_SIZE)]
    for slot_outputs in slots:
        assert_matches_target(case, slot_outputs)
    if len(expected_distribution(case)) <= 16:
        assert_pairs_match_target(case, list(zip(slots[0], slots[1])))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("schedule, frequency_of_a", [(APPENDIX_A_SCHEDULE.verify_lengths, 0.70),
                                                      (global_maximum, 0.85)], ids=["early_stop", "global_maximum"])
def test_appendix_a(schedule, frequency_of_a, device):
    """The paper's counterexample through the same rounds: the early stop emits A with the target's 0.70; the
    global maximum emits it with 0.85, the biased value the paper computes, which the gate would fail."""
    generator = torch.Generator(device=device).manual_seed(17011)
    by_state = torch.tensor(APPENDIX_A_LOGITS, device=device)
    outputs, _ = simulate_rounds(APPENDIX_A, 2, APPENDIX_A_TRIALS, generator, torch.device(device),
                                 scheduled_lengths(schedule, by_state, 1))
    frequency = sum(output == (0,) for output in outputs) / len(outputs)
    assert abs(frequency - frequency_of_a) <= frequency_bound(2, len(outputs), FAILURE_PROBABILITY)
