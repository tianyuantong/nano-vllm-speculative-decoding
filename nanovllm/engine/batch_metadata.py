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


def prefill_metadata(seqs: list[Sequence], role: str, block_size: int) -> PrefillMetadata:
    metadata = PrefillMetadata()
    for seq in seqs:
        kv = seq.kv(role)
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
    slot_mapping: list[list[int]] = field(default_factory=list)   # [num_steps][B]: slot of position len-1+step


def decode_metadata(seqs: list[Sequence], role: str, block_size: int, num_steps: int = 1,
                    position_offset: int = 0) -> DecodeMetadata:
    """Single-token steps. Offset 0: the last token at position len-1 (ordinary decode / draft step 0).
    Offset -1: the token before it, at position len-2 (the draft's catch-up of one missing KV entry)."""
    metadata = DecodeMetadata(slot_mapping=[[] for _ in range(num_steps)])
    for seq in seqs:
        block_table = seq.kv(role).block_table
        position = len(seq) - 1 + position_offset
        metadata.input_ids.append(seq.last_token if position_offset == 0 else seq[position])
        metadata.positions.append(position)
        metadata.context_lens.append(position + 1)
        for step in range(num_steps):
            metadata.slot_mapping[step].append(slot_of(block_table, block_size, position + step))
    return metadata


def padded_block_tables(seqs: list[Sequence], role: str) -> list[list[int]]:
    tables = [seq.kv(role).block_table for seq in seqs]
    width = max(len(table) for table in tables)
    return [table + [PAD_BLOCK_ENTRY] * (width - len(table)) for table in tables]


@dataclass
class VerifyMetadata:
    """One verification forward: every row has k + 1 query tokens; padding rows read a zero block."""
    input_ids: list[int] = field(default_factory=list)      # draft positions hold 0, filled on device
    positions: list[int] = field(default_factory=list)
    slot_mapping: list[int] = field(default_factory=list)
    cu_seqlens_q: list[int] = field(default_factory=lambda: [0])
    cu_seqlens_k: list[int] = field(default_factory=lambda: [0])
    block_tables: list[list[int]] = field(default_factory=list)


def verify_metadata(seqs: list[Sequence], num_drafts: int, block_size: int,
                    padded_batch_size: int, pad_block_id: int, width: int) -> VerifyMetadata:
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
