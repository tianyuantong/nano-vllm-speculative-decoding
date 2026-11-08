"""Speculative rounds with a real BlockManager and the prefix cache on, around test_dflash_round's fake models.

Every prompt follows the rule right(p), so a shorter prompt is a prefix of a longer one and their whole blocks hit.
Every write, the target's and the draft's, in prefill and in rounds, is recorded by slot, so a write of the value a
slot already holds still counts: no write may touch a block whose content another sequence relies on, one the hash
table holds or two sequences share. The fake DFlash also checks that each block forward reads, below its anchor, the
context K/V of the committed token at every position, the cached ones included.
"""
import random

import pytest

from test_dflash_round import BLOCK_SIZE, K, VOCAB, commit, make_decoder, right
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


class Writes(dict):
    """A dict that remembers every key written since the last clear of `written`."""

    def __init__(self, *args):
        super().__init__(*args)
        self.written = set()

    def __setitem__(self, key, value):
        self.written.add(key)
        super().__setitem__(key, value)


def protected(manager: BlockManager) -> set[int]:
    """Blocks whose content must stay as it is: registered in the hash table, or held by two sequences or more."""
    return set(manager.hash_to_block_id.values()) | {block.block_id for block in manager.blocks if block.ref_count > 1}


def written_blocks(target, draft) -> set[int]:
    blocks = {slot // BLOCK_SIZE for slot in target.kv.written | draft.cache.written if slot >= 0}
    target.kv.written.clear()
    draft.cache.written.clear()
    return blocks


def seq_of(length: int) -> Sequence:
    return Sequence([right(p) for p in range(length)], SamplingParams(temperature=0.0, max_tokens=512))


def make(num_blocks: int = 64):
    rng = random.Random(num_blocks)
    decoder, target, draft = make_decoder(lambda position: rng.random() < 0.6)
    target.kv, draft.cache = Writes(target.kv), Writes(draft.cache)
    return decoder, target, draft, BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE, enable_prefix_cache=True)


def admit(manager, decoder, target, draft, seq) -> int:
    """What Scheduler.schedule, LLMEngine.step and Scheduler.postprocess do for one whole prefill; returns the
    blocks it hit. Its writes must stay off the blocks protected once its own are allocated."""
    cached = manager.can_allocate(seq)
    assert cached >= 0
    manager.allocate(seq, cached)
    assert seq.prefix_hit_tokens == cached * BLOCK_SIZE
    seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
    before = protected(manager)
    written_blocks(target, draft)
    aux = decoder.prefill_aux([seq])
    target.prefill([seq], aux)
    decoder.prefill_draft([seq], aux)
    assert not written_blocks(target, draft) & before
    manager.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    seq.append_token(right(len(seq)))
    return cached


def rounds(manager, decoder, target, draft, seqs, n, verified_lens=None) -> None:
    for _ in range(n):
        for seq in seqs:
            manager.reserve(seq, len(seq) + K)
        before = protected(manager)
        written_blocks(target, draft)
        lengths = verified_lens(seqs) if verified_lens else None
        commit(seqs, decoder.run_round(seqs, lengths))
        assert not written_blocks(target, draft) & before


@pytest.mark.parametrize("n_a,n_b", [(8, 12), (9, 12), (12, 12), (8, 8), (11, 23), (16, 17)])
def test_a_hit_keeps_the_shared_blocks_and_the_draft_context(n_a, n_b):
    decoder, target, draft, manager = make()
    a = seq_of(n_a)
    assert admit(manager, decoder, target, draft, a) == 0
    rounds(manager, decoder, target, draft, [a], 2)
    b = seq_of(n_b)
    assert admit(manager, decoder, target, draft, b) > 0
    assert any(block.ref_count > 1 for block in manager.blocks)
    rounds(manager, decoder, target, draft, [a, b], 6)       # A may run past its prompt; B reads the shared blocks
    assert draft.context_reads > 0 and not draft.bad_reads, draft.bad_reads[:3]


def test_rows_of_their_own_lengths_keep_the_shared_blocks():
    rng = random.Random(7)
    decoder, target, draft, manager = make()
    donor = seq_of(10)
    admit(manager, decoder, target, draft, donor)
    seqs = [donor] + [seq_of(10 + index) for index in range(1, 6)]
    for seq in seqs[1:]:
        assert admit(manager, decoder, target, draft, seq) >= 2    # each also hits the whole blocks of the one before
    rounds(manager, decoder, target, draft, seqs, 10, lambda batch: [rng.randint(1, K + 1) for _ in batch])
    assert not draft.bad_reads


def test_a_hashed_block_held_by_one_sequence_is_not_written():
    decoder, target, draft, manager = make()
    a = seq_of(12)
    admit(manager, decoder, target, draft, a)                 # blocks 0 and 1 hashed, held by A alone
    assert all(manager.blocks[block_id].ref_count == 1 for block_id in a.block_table[:2])
    rounds(manager, decoder, target, draft, [a], 8)


def test_a_freed_block_is_reused_by_a_hit():
    decoder, target, draft, manager = make()
    a = seq_of(12)
    admit(manager, decoder, target, draft, a)
    rounds(manager, decoder, target, draft, [a], 3)
    manager.deallocate(a)
    b = seq_of(12)
    assert admit(manager, decoder, target, draft, b) == 2
    rounds(manager, decoder, target, draft, [b], 5)
    assert not draft.bad_reads


def test_an_evicted_block_is_not_a_hit():
    decoder, target, draft, manager = make(num_blocks=8)
    a = seq_of(12)
    admit(manager, decoder, target, draft, a)
    rounds(manager, decoder, target, draft, [a], 2)
    manager.deallocate(a)
    other = Sequence([(right(p) + 1) % (VOCAB - 1) for p in range(32)], SamplingParams(temperature=0.0, max_tokens=512))
    assert manager.can_allocate(other) == 0
    manager.allocate(other, 0)                                # takes every block, A's among them
    assert all(manager.blocks[block_id].hash == h for h, block_id in manager.hash_to_block_id.items())
    manager.deallocate(other)
    assert manager.can_allocate(seq_of(12)) == 0              # A's blocks were reused, so no stale hit
