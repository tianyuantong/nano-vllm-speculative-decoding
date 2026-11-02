"""Host-side metadata for every forward shape, built from sequences as plain lists.

ModelRunner and SpeculativeDecoder turn these into device tensors; keeping the
construction tensor-free makes it testable without a GPU.
"""
from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence

PAD_SLOT = -1          # store_kvcache_kernel skips slot -1
PAD_BLOCK_ENTRY = -1   # unused block-table entries of decode batches (never read)


def slot_of(block_table: list[int], block_size: int, position: int) -> int:
    return block_table[position // block_size] * block_size + position % block_size


def slots_for_range(block_table: list[int], block_size: int, start: int, end: int) -> list[int]:
    """Slots of positions start..end-1, one range per block (upstream's prefill loop)."""
    slots: list[int] = []
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots.extend(range(slot_start, slot_end))
    return slots


@dataclass
class PrefillMetadata:
    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    cu_seqlens_q: list[int] = field(default_factory=lambda: [0])
    cu_seqlens_k: list[int] = field(default_factory=lambda: [0])
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: list[int] = field(default_factory=list)

    @property
    def has_cached_prefix(self) -> bool:
        return self.cu_seqlens_k[-1] > self.cu_seqlens_q[-1]


def prefill_metadata(seqs: list[Sequence], block_size: int) -> PrefillMetadata:
    metadata = PrefillMetadata()
    for seq in seqs:
        kv = seq.target_kv
        start = kv.num_cached_tokens
        seqlen_q = kv.num_scheduled_tokens
        end = start + seqlen_q
        metadata.input_ids.extend(seq[start:end])
        metadata.positions.extend(range(start, end))
        metadata.cu_seqlens_q.append(metadata.cu_seqlens_q[-1] + seqlen_q)
        metadata.cu_seqlens_k.append(metadata.cu_seqlens_k[-1] + end)
        metadata.max_seqlen_q = max(seqlen_q, metadata.max_seqlen_q)
        metadata.max_seqlen_k = max(end, metadata.max_seqlen_k)
        if kv.block_table:    # warm-up runs without KV blocks
            metadata.slot_mapping.extend(slots_for_range(kv.block_table, block_size, start, end))
    return metadata


@dataclass
class DecodeMetadata:
    input_ids: list[int] = field(default_factory=list)      # last committed token per row
    positions: list[int] = field(default_factory=list)      # len(seq) - 1 per row
    context_lens: list[int] = field(default_factory=list)   # len(seq) per row
    slot_mapping: list[int] = field(default_factory=list)   # slot of position len - 1 per row


def decode_metadata(seqs: list[Sequence], block_size: int) -> DecodeMetadata:
    """One single-token step: each row's last token, at position len-1."""
    metadata = DecodeMetadata()
    for seq in seqs:
        position = len(seq) - 1
        metadata.input_ids.append(seq.last_token)
        metadata.positions.append(position)
        metadata.context_lens.append(position + 1)
        metadata.slot_mapping.append(slot_of(seq.target_kv.block_table, block_size, position))
    return metadata


@dataclass
class DraftBlockMetadata:
    """One DFlash draft block per row: the last committed token (the anchor, at position len-1) and
    query_len - 1 mask tokens after it, in the target's block table. Padding rows store nothing and read
    query_len positions of the pad block."""
    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    slot_mapping: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)   # len-1 + query_len: the context below the anchor and the block
    block_tables: list[list[int]] = field(default_factory=list)


def draft_block_metadata(seqs: list[Sequence], block_size: int, query_len: int, mask_token_id: int,
                         padded_batch_size: int, pad_block_id: int, width: int) -> DraftBlockMetadata:
    metadata = DraftBlockMetadata()
    for seq in seqs:
        block_table = seq.target_kv.block_table
        anchor = len(seq) - 1
        metadata.input_ids.extend([seq.last_token] + [mask_token_id] * (query_len - 1))
        metadata.positions.extend(range(anchor, anchor + query_len))
        metadata.slot_mapping.extend(slot_of(block_table, block_size, position)
                                     for position in range(anchor, anchor + query_len))
        metadata.context_lens.append(anchor + query_len)
        metadata.block_tables.append(block_table + [pad_block_id] * (width - len(block_table)))
    for _ in range(padded_batch_size - len(seqs)):
        metadata.input_ids.extend([mask_token_id] * query_len)
        metadata.positions.extend(range(query_len))
        metadata.slot_mapping.extend([PAD_SLOT] * query_len)
        metadata.context_lens.append(query_len)
        metadata.block_tables.append([pad_block_id] * width)
    return metadata


def padded_block_tables(seqs: list[Sequence]) -> list[list[int]]:
    tables = [seq.target_kv.block_table for seq in seqs]
    width = max(len(table) for table in tables)
    return [table + [PAD_BLOCK_ENTRY] * (width - len(table)) for table in tables]


@dataclass
class VerifyMetadata:
    """One verification forward over its row slots, the sequences' rows first, then the padding rows, which read a
    zero block, then the empty ones."""
    input_ids: list[int] = field(default_factory=list)      # draft positions hold 0, filled on device
    positions: list[int] = field(default_factory=list)
    slot_mapping: list[int] = field(default_factory=list)
    cu_seqlens_q: list[int] = field(default_factory=lambda: [0])
    cu_seqlens_k: list[int] = field(default_factory=lambda: [0])
    block_tables: list[list[int]] = field(default_factory=list)
    draft_index: list[int] = field(default_factory=list)    # where each verified draft goes in input_ids
    draft_source: list[int] = field(default_factory=list)   # and where it is in the drafts [B, k], flattened


def verify_metadata(seqs: list[Sequence], num_drafts: int, block_size: int,
                    padded_batch_size: int, pad_block_id: int, width: int) -> VerifyMetadata:
    """Every row verifies [last token, its num_drafts drafts]; padded to padded_batch_size rows of as many tokens."""
    query_len = num_drafts + 1
    return verify_varlen_metadata(seqs, [query_len] * len(seqs), num_drafts, block_size, padded_batch_size,
                                  padded_batch_size * query_len, pad_block_id, width)


def verify_varlen_metadata(seqs: list[Sequence], verified_lens: list[int], num_drafts: int, block_size: int,
                           num_rows: int, num_tokens: int, pad_block_id: int, width: int) -> VerifyMetadata:
    """Row r verifies verified_lens[r] tokens, its last token and its first verified_lens[r] - 1 drafts, at
    positions len-1 onward. The tokens left of num_tokens fill padding rows of at most num_drafts + 1 tokens, at
    positions 0 onward of the zero block; the row slots left of num_rows after them are empty."""
    query_len = num_drafts + 1
    metadata = VerifyMetadata()
    for row, (seq, length) in enumerate(zip(seqs, verified_lens)):
        block_table = seq.target_kv.block_table
        start = len(seq) - 1
        assert 1 <= length <= query_len
        assert len(block_table) <= width and len(block_table) * block_size >= start + length
        first = len(metadata.input_ids)
        metadata.input_ids.extend([seq.last_token] + [0] * (length - 1))
        metadata.positions.extend(range(start, start + length))
        metadata.slot_mapping.extend(slot_of(block_table, block_size, position) for position in range(start, start + length))
        metadata.draft_index.extend(range(first + 1, first + length))
        metadata.draft_source.extend(range(row * num_drafts, row * num_drafts + length - 1))
        metadata.cu_seqlens_q.append(metadata.cu_seqlens_q[-1] + length)
        metadata.cu_seqlens_k.append(metadata.cu_seqlens_k[-1] + start + length)
        metadata.block_tables.append(block_table + [pad_block_id] * (width - len(block_table)))
    left = num_tokens - len(metadata.input_ids)
    for _ in range(num_rows - len(seqs)):
        length = min(left, query_len)
        left -= length
        metadata.input_ids.extend([0] * length)
        metadata.positions.extend(range(length))
        metadata.slot_mapping.extend([PAD_SLOT] * length)
        metadata.cu_seqlens_q.append(metadata.cu_seqlens_q[-1] + length)
        metadata.cu_seqlens_k.append(metadata.cu_seqlens_k[-1] + length)
        metadata.block_tables.append([pad_block_id] * width)
    assert left == 0, "the padding rows cannot hold the tokens left"
    return metadata
