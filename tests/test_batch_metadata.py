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


def make_sequence(num_tokens, target_blocks):
    seq = Sequence(list(range(num_tokens)))
    seq.target_kv.block_table = list(target_blocks)
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


def test_prefill_metadata_covers_the_scheduled_chunk():
    seq = make_sequence(6, target_blocks=[3, 4])
    seq.target_kv.num_cached_tokens, seq.target_kv.num_scheduled_tokens = 4, 2
    metadata = prefill_metadata([seq], BLOCK_SIZE)
    assert metadata.input_ids == [4, 5] and metadata.positions == [4, 5]
    assert metadata.cu_seqlens_q == [0, 2] and metadata.cu_seqlens_k == [0, 6]
    assert metadata.slot_mapping == [16, 17] and metadata.has_cached_prefix


def test_prefill_metadata_skips_slots_without_blocks():
    seq = Sequence([1, 2, 3])
    seq.num_scheduled_tokens = 3
    metadata = prefill_metadata([seq], BLOCK_SIZE)
    assert metadata.slot_mapping == [] and metadata.max_seqlen_q == 3


def test_decode_metadata_is_one_step_at_the_last_position():
    seq = make_sequence(5, target_blocks=[2, 3])      # position 4 -> block index 1, slot 3 * 4 + 0
    metadata = decode_metadata([seq], BLOCK_SIZE)
    assert metadata.input_ids == [4] and metadata.positions == [4] and metadata.context_lens == [5]
    assert metadata.slot_mapping == [12]


def test_padded_block_tables_pad_with_minus_one():
    seqs = [make_sequence(1, [1]), make_sequence(1, [2, 3])]
    assert padded_block_tables(seqs) == [[1, -1], [2, 3]]


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

