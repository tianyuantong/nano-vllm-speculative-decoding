from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
EOS = 7


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def make_scheduler(num_speculative_tokens, num_blocks=16, max_num_seqs=4):
    config = SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=64, eos=EOS,
                             kvcache_block_size=BLOCK_SIZE, num_kvcache_blocks=num_blocks,
                             enable_prefix_cache=False, num_speculative_tokens=num_speculative_tokens)
    return Scheduler(config)


def prefilled(scheduler, prompt, first_token, **params):
    """Add a sequence and run its prefill step so it sits at the round-entry invariants."""
    seq = Sequence(prompt, SamplingParams(**params))
    scheduler.add(seq)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [seq]
    scheduler.postprocess(seqs, [first_token], True)
    return seq


def test_prefill_allocates_and_advances_the_kv_state():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    assert seq.target_kv.block_table == [0, 1] and seq.target_kv.num_cached_tokens == 5
    assert len(seq) == 6 and seq.num_cached_tokens == len(seq) - 1


def test_decode_scheduling_reserves_len_plus_k():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)   # len 6 -> 9 tokens
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs == [seq]
    assert len(seq.target_kv.block_table) == 3


def test_k0_decode_reserves_exactly_like_upstream():
    scheduler = make_scheduler(num_speculative_tokens=0)
    seq = prefilled(scheduler, [1, 2, 3, 4], first_token=9, max_tokens=8)        # len 5 -> block for position 4
    scheduler.schedule()
    assert len(seq.block_table) == 2


def test_decode_preempts_the_last_running_sequence_when_the_pool_is_short():
    scheduler = make_scheduler(num_speculative_tokens=3, num_blocks=6)
    first = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)    # 2 blocks each
    second = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    seqs, is_prefill = scheduler.schedule()                  # len 6: a round needs 9 tokens = 3 blocks, all 6 used
    assert not is_prefill and seqs == [first, second]
    scheduler.postprocess_speculative(seqs, [[10, 11, 12, 13], [10, 11, 12, 13]])   # len 10: 13 tokens = 4 blocks
    assert scheduler.num_preemptions == 0
    seqs, is_prefill = scheduler.schedule()                  # no free block: the last running seq is preempted
    assert scheduler.num_preemptions == 1
    assert not is_prefill and seqs == [first]
    assert second.status == SequenceStatus.WAITING and second.target_kv.block_table == []
    assert len(first.target_kv.block_table) == 4


def test_postprocess_speculative_keeps_the_kv_invariant():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11, 12, 13]])          # all three accepted + bonus
    assert seq.completion_token_ids == [9, 10, 11, 12, 13]
    assert seq.target_kv.num_cached_tokens == len(seq) - 1
    assert seq in scheduler.running


def test_postprocess_speculative_after_partial_acceptance():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11]])                   # one accepted + correction
    assert seq.target_kv.num_cached_tokens == len(seq) - 1


def test_postprocess_speculative_stops_at_eos_and_frees_the_blocks():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, EOS, 12, 13]])
    assert seq.completion_token_ids == [9, 10, EOS]
    assert seq.is_finished and seq not in scheduler.running
    assert len(scheduler.block_manager.free_block_ids) == 16


def test_postprocess_speculative_stops_at_max_tokens():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=3)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11, 12, 13]])
    assert seq.completion_token_ids == [9, 10, 11]
    assert seq.is_finished


def test_postprocess_speculative_ignore_eos_continues():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8, ignore_eos=True)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[EOS, 11]])
    assert seq.completion_token_ids == [9, EOS, 11] and not seq.is_finished


def admit(scheduler, num_tokens=2):
    seq = Sequence(list(range(1, num_tokens + 1)), SamplingParams(max_tokens=8))
    scheduler.add(seq)
    return seq


def test_admission_stops_at_max_num_seqs_counting_each_request_once():
    scheduler = make_scheduler(num_speculative_tokens=3, max_num_seqs=4, num_blocks=32)
    first = prefilled(scheduler, [1, 2], first_token=9, max_tokens=8)
    second = prefilled(scheduler, [1, 2], first_token=9, max_tokens=8)
    arrivals = [admit(scheduler) for _ in range(3)]
    seqs, is_prefill = scheduler.schedule()              # 2 running, slots for 2 more in one step: not 3 + 1 > 4
    assert is_prefill and seqs == arrivals[:2]
    scheduler.postprocess(seqs, [9, 9], True)
    seqs, is_prefill = scheduler.schedule()              # the third waits; the four running decode
    assert not is_prefill and seqs == [first, second, *arrivals[:2]]
    assert list(scheduler.waiting) == arrivals[2:]


def test_a_finished_request_frees_its_slot():
    scheduler = make_scheduler(num_speculative_tokens=3, max_num_seqs=1, num_blocks=32)
    running = prefilled(scheduler, [1, 2], first_token=9, max_tokens=8)
    arrival = admit(scheduler)
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs == [running]
    scheduler.postprocess_speculative(seqs, [[EOS]])
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [arrival]


def test_a_chunked_prefill_keeps_its_slot_and_needs_no_new_one():
    scheduler = make_scheduler(num_speculative_tokens=3, max_num_seqs=2, num_blocks=32)
    scheduler.max_num_batched_tokens = 4
    running = prefilled(scheduler, [1, 2], first_token=9, max_tokens=8)
    chunked = admit(scheduler, num_tokens=6)
    late = admit(scheduler)
    seqs, is_prefill = scheduler.schedule()              # the first chunk: 4 of 6 tokens, so it stays in waiting
    assert is_prefill and seqs == [chunked] and chunked.num_scheduled_tokens == 4
    scheduler.postprocess(seqs, [9], True)
    assert list(scheduler.waiting) == [chunked, late]
    seqs, is_prefill = scheduler.schedule()              # running + chunked hold both slots; chunked continues, late waits
    assert is_prefill and seqs == [chunked] and chunked.num_scheduled_tokens == 2
    scheduler.postprocess(seqs, [9], True)
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs == [running, chunked]
    assert list(scheduler.waiting) == [late]


def test_a_closed_loop_never_reaches_the_cap():
    # C requests in flight at most: every waiting request that holds no block leaves a free slot for itself.
    scheduler = make_scheduler(num_speculative_tokens=3, max_num_seqs=3, num_blocks=32)
    running = prefilled(scheduler, [1, 2], first_token=9, max_tokens=8)
    arrivals = [admit(scheduler) for _ in range(2)]
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == arrivals
    scheduler.postprocess(seqs, [9, 9], True)
    assert list(scheduler.running) == [running, *arrivals]
