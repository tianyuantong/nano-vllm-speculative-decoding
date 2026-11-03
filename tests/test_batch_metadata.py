import random

from nanovllm.engine.batch_metadata import (
    PAD_SLOT,
    VerifyMetadata,
    decode_metadata,
    padded_block_tables,
    prefill_metadata,
    slot_of,
    slots_for_range,
    verify_metadata,
    verify_varlen_metadata,
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


def m3_verify_metadata(seqs, num_drafts, block_size, padded_batch_size, pad_block_id, width):
    """verify_metadata's body as it was at 7bf0466, before the variable-length rewrite."""
    query_len = num_drafts + 1
    metadata = VerifyMetadata()
    for seq in seqs:
        block_table = seq.target_kv.block_table
        start = len(seq) - 1
        assert len(block_table) <= width and len(block_table) * block_size >= start + query_len
        metadata.input_ids.extend([seq.last_token] + [0] * num_drafts)
        metadata.positions.extend(range(start, start + query_len))
        metadata.slot_mapping.extend(slot_of(block_table, block_size, position) for position in range(start, start + query_len))
        metadata.cu_seqlens_q.append(metadata.cu_seqlens_q[-1] + query_len)
        metadata.cu_seqlens_k.append(metadata.cu_seqlens_k[-1] + start + query_len)
        metadata.block_tables.append(block_table + [pad_block_id] * (width - len(block_table)))
    for _ in range(padded_batch_size - len(seqs)):
        metadata.input_ids.extend([0] * query_len)
        metadata.positions.extend(range(query_len))
        metadata.slot_mapping.extend([PAD_SLOT] * query_len)
        metadata.cu_seqlens_q.append(metadata.cu_seqlens_q[-1] + query_len)
        metadata.cu_seqlens_k.append(metadata.cu_seqlens_k[-1] + query_len)
        metadata.block_tables.append([pad_block_id] * width)
    return metadata


def test_fixed_length_metadata_is_the_m3_metadata():
    # The fixed-length verification is now the variable-length layout with every row k + 1 long: same lists.
    rng = random.Random(1)
    for _ in range(50):
        num_drafts = rng.randint(1, 7)
        seqs = [make_sequence(rng.randint(1, 30), target_blocks=rng.sample(range(40), 10)) for _ in range(rng.randint(0, 5))]
        padded = len(seqs) + rng.randint(0, 3)
        ours = verify_metadata(seqs, num_drafts, BLOCK_SIZE, padded, 99, 12)
        theirs = m3_verify_metadata(seqs, num_drafts, BLOCK_SIZE, padded, 99, 12)
        for name in ("input_ids", "positions", "slot_mapping", "cu_seqlens_q", "cu_seqlens_k", "block_tables"):
            assert getattr(ours, name) == getattr(theirs, name)


def test_varlen_metadata_lays_out_rows_of_their_own_length():
    # The design's example, block size 256: row A (len 100, block [7]) verifies 5 tokens, row B (len 37, block [3])
    # 3, in a graph of 8 tokens and 2 row slots. With k = 7, A's drafts are 0..6 of the flat drafts, B's 7..13.
    rows = [make_sequence(100, target_blocks=[7]), make_sequence(37, target_blocks=[3])]
    metadata = verify_varlen_metadata(rows, [5, 3], num_drafts=7, block_size=256, num_rows=2, num_tokens=8,
                                      pad_block_id=99, width=1)
    assert metadata.input_ids == [99, 0, 0, 0, 0, 36, 0, 0]
    assert metadata.positions == [99, 100, 101, 102, 103, 36, 37, 38]
    assert metadata.slot_mapping == [1891, 1892, 1893, 1894, 1895, 804, 805, 806]
    assert metadata.cu_seqlens_q == [0, 5, 8] and metadata.cu_seqlens_k == [0, 104, 143]
    assert metadata.draft_index == [1, 2, 3, 4, 6, 7] and metadata.draft_source == [0, 1, 2, 3, 7, 8]


def test_varlen_metadata_fills_padding_rows_then_leaves_slots_empty():
    # One row of 3 in a graph of 9 tokens and 4 slots, k = 3: the 6 tokens left fill a padding row of 4 and one
    # of 2, both on the pad block from position 0; the last slot is an empty row.
    seq = make_sequence(5, target_blocks=[7, 8])           # positions 4, 5, 6
    metadata = verify_varlen_metadata([seq], [3], num_drafts=3, block_size=BLOCK_SIZE, num_rows=4, num_tokens=9,
                                      pad_block_id=99, width=2)
    assert metadata.positions == [4, 5, 6, 0, 1, 2, 3, 0, 1]
    assert metadata.slot_mapping == [32, 33, 34] + [PAD_SLOT] * 6
    assert metadata.cu_seqlens_q == [0, 3, 7, 9, 9] and metadata.cu_seqlens_k == [0, 7, 11, 13, 13]
    assert metadata.block_tables == [[7, 8], [99, 99], [99, 99], [99, 99]]

