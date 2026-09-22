from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import KVState, Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Block allocation for one model's KV cache; `role` selects which KVState of a sequence it owns."""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool, role: str = "target"):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.role = role
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _kv(self, seq: Sequence) -> KVState:
        return seq.kv(self.role)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if not self.enable_prefix_cache:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1

        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        kv = self._kv(seq)
        assert not kv.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            kv.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            kv.block_table.append(self._allocate_block())
        kv.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        kv = self._kv(seq)
        for block_id in reversed(kv.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        kv.num_cached_tokens = 0
        kv.block_table.clear()

    def _blocks_needed(self, seq: Sequence, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size - len(self._kv(seq).block_table)

    def can_reserve(self, seq: Sequence, num_tokens: int) -> bool:
        return len(self.free_block_ids) >= self._blocks_needed(seq, num_tokens)

    def reserve(self, seq: Sequence, num_tokens: int):
        """Grow the block table until it covers `num_tokens` positions."""
        kv = self._kv(seq)
        for _ in range(self._blocks_needed(seq, num_tokens)):
            kv.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache:
            return
        kv = self._kv(seq)
        start = kv.num_cached_tokens // self.block_size
        end = (kv.num_cached_tokens + kv.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[kv.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[kv.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
