import random

from nanovllm.engine.batch_metadata import (
    decode_metadata,
    padded_block_tables,
    prefill_metadata,
    slot_of,
    slots_for_range,
    verify_metadata,
)
from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 4


def make_sequence(num_tokens, target_blocks, draft_blocks=()):
    seq = Sequence(list(range(num_tokens)))
    seq.target_kv.block_table = list(target_blocks)
    seq.draft_kv.block_table = list(draft_blocks)
    return seq


def test_slot_of_maps_position_through_the_block_table():
    assert slot_of([5, 2], BLOCK_SIZE, 0) == 20
    assert slot_of([5, 2], BLOCK_SIZE, 6) == 10


def test_slots_for_range_equals_per_position_slots():
    rng = random.Random(0)
    for _ in range(200):
        table = rng.sample(range(50), 6)
        start = rng.randrange(0, 20)
        end = rng.randrange(start + 1, 24)
        expected = [slot_of(table, BLOCK_SIZE, position) for position in range(start, end)]
        assert slots_for_range(table, BLOCK_SIZE, start, end) == expected


def test_prefill_metadata_reads_the_requested_role():
    seq = make_sequence(6, target_blocks=[3, 4], draft_blocks=[8, 9])
    seq.target_kv.num_cached_tokens, seq.target_kv.num_scheduled_tokens = 4, 2
    seq.draft_kv.num_cached_tokens, seq.draft_kv.num_scheduled_tokens = 0, 6
    target = prefill_metadata([seq], "target", BLOCK_SIZE)
    draft = prefill_metadata([seq], "draft", BLOCK_SIZE)
    assert target.input_ids == [4, 5] and target.positions == [4, 5]
    assert target.cu_seqlens_q == [0, 2] and target.cu_seqlens_k == [0, 6]
    assert target.slot_mapping == [16, 17] and target.has_cached_prefix
    assert draft.input_ids == list(range(6)) and draft.slot_mapping == [32, 33, 34, 35, 36, 37]
    assert not draft.has_cached_prefix


def test_prefill_metadata_skips_slots_without_blocks():
    seq = Sequence([1, 2, 3])
    seq.num_scheduled_tokens = 3
    metadata = prefill_metadata([seq], "target", BLOCK_SIZE)
    assert metadata.slot_mapping == [] and metadata.max_seqlen_q == 3


def test_decode_metadata_covers_consecutive_steps():
    seq = make_sequence(5, target_blocks=[0, 1], draft_blocks=[2, 3])   # positions 4, 5, 6 -> block index 1
    metadata = decode_metadata([seq], "draft", BLOCK_SIZE, num_steps=3)
    assert metadata.input_ids == [4] and metadata.positions == [4] and metadata.context_lens == [5]
    assert metadata.slot_mapping == [[12], [13], [14]]


def test_padded_block_tables_pad_with_minus_one():
    seqs = [make_sequence(1, [1]), make_sequence(1, [2, 3])]
    assert padded_block_tables(seqs, "target") == [[1, -1], [2, 3]]


def test_verify_metadata_layout_and_padding_rows():
    seq = make_sequence(5, target_blocks=[7, 8, 9])        # len 5, k = 2 -> positions 4, 5, 6, cu_k = 7
    metadata = verify_metadata([seq], num_drafts=2, block_size=BLOCK_SIZE,
                               padded_batch_size=2, pad_block_id=99, width=3)
    assert metadata.input_ids == [4, 0, 0, 0, 0, 0]
    assert metadata.positions == [4, 5, 6, 0, 1, 2]
    assert metadata.slot_mapping == [32, 33, 34, -1, -1, -1]
    assert metadata.cu_seqlens_q == [0, 3, 6]
    assert metadata.cu_seqlens_k == [0, 7, 10]
    assert metadata.block_tables == [[7, 8, 9], [99, 99, 99]]


def test_verify_metadata_pads_short_block_tables_with_the_pad_block():
    seq = make_sequence(2, target_blocks=[4])
    metadata = verify_metadata([seq], num_drafts=1, block_size=BLOCK_SIZE,
                               padded_batch_size=1, pad_block_id=99, width=2)
    assert metadata.block_tables == [[4, 99]]


def test_decode_metadata_position_offset_targets_the_previous_position():
    seq = make_sequence(5, target_blocks=[0, 1], draft_blocks=[2, 3])   # tokens 0..4; catch-up recomputes position 3
    metadata = decode_metadata([seq], "draft", BLOCK_SIZE, num_steps=1, position_offset=-1)
    assert metadata.input_ids == [3] and metadata.positions == [3] and metadata.context_lens == [4]
    assert metadata.slot_mapping == [[11]]
