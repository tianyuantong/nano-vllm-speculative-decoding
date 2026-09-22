import pytest

from nanovllm.engine.sequence import KVState, Sequence
from nanovllm.sampling_params import SamplingParams


def test_upstream_names_address_the_target_kv():
    seq = Sequence([1, 2, 3])
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.block_table.append(7)
    assert seq.target_kv == KVState(num_cached_tokens=2, num_scheduled_tokens=1, block_table=[7])


def test_draft_kv_is_independent_of_target_kv():
    seq = Sequence([1, 2, 3])
    seq.draft_kv.num_cached_tokens = 5
    assert seq.num_cached_tokens == 0
    assert seq.kv("draft") is seq.draft_kv
    assert seq.kv("target") is seq.target_kv


def test_sampling_parameters_are_copied_onto_the_sequence():
    seq = Sequence([1], SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=3))
    assert (seq.temperature, seq.top_k, seq.top_p, seq.max_tokens) == (0.7, 20, 0.8, 3)


def test_step_indices_start_unset():
    seq = Sequence([1])
    assert (seq.first_token_step, seq.finish_step) == (-1, -1)


def test_top_p_requires_top_k():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.8)
    SamplingParams(top_p=0.8, top_k=20)


def test_pickle_round_trip_keeps_target_kv_only():
    seq = Sequence([1, 2, 3])
    seq.num_cached_tokens = 3
    seq.block_table.append(4)
    seq.is_prefill = False
    restored = Sequence.__new__(Sequence)
    restored.__setstate__(seq.__getstate__())
    assert restored.block_table == [4]
    assert restored.num_cached_tokens == 3
    assert restored.draft_kv == KVState()
