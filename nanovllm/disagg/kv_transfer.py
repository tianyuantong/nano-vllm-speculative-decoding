"""Block-remapped KV cache transfer between a Prefill engine and a Decode engine."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import threading
import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class KVCacheTransfer:
    """Copies per-sequence KV blocks from Prefill (P) to Decode (D) and remaps `seq.target_kv.block_table`.

    Both Target KV (`[2, L_target, num_blocks, block_size, H, D]`) and, when speculative decoding is active,
    Draft context KV (`[2, L_draft, num_blocks, block_size, H, D]`) share the same block table and are copied
    block-by-block on `transfer_stream`.

    Stream Ordering (Three-Stage Pipeline without CPU blocking):
    1. `transfer_stream.wait_event(prefill_done_event)` ensures P has finished writing `p_blocks`.
    2. `transfer_stream.wait_event(d_alloc_event)` ensures D's previous decode step is no longer reading `d_blocks`.
    3. `copy_(non_blocking=True)` copies `p_blocks -> d_blocks` via the GPU Copy Engine (PCIe P2P / NVLink / D2D).
    4. `copy_done_event` is recorded on `transfer_stream`:
       - D's `d_stream.wait_event(copy_done_event)` waits on the GPU before decoding `seq`.
       - P's `p_blocks` are held in `_in_flight` and returned to `p_bm` once `copy_done_event.query()` succeeds.
    """

    def __init__(
        self,
        p_kv_cache: torch.Tensor,
        d_kv_cache: torch.Tensor,
        p_bm: BlockManager,
        d_bm: BlockManager,
        *,
        p_draft_kv_cache: torch.Tensor | None = None,
        d_draft_kv_cache: torch.Tensor | None = None,
        transfer_stream: torch.cuda.Stream | None = None,
        d_transfer_stream: torch.cuda.Stream | None = None,
        event_factory=None,
    ) -> None:
        if (p_draft_kv_cache is None) != (d_draft_kv_cache is None):
            raise ValueError("p_draft_kv_cache and d_draft_kv_cache must both be provided or both be None")
        self.p_kv_cache = p_kv_cache
        self.d_kv_cache = d_kv_cache
        self.p_draft_kv_cache = p_draft_kv_cache
        self.d_draft_kv_cache = d_draft_kv_cache
        self.p_bm = p_bm
        self.d_bm = d_bm
        self.transfer_stream = transfer_stream
        self.d_transfer_stream = d_transfer_stream
        if event_factory is not None:
            self._new_event = event_factory
        elif p_kv_cache.device.type == "cuda":
            self._new_event = lambda: torch.cuda.Event()
        else:
            self._new_event = _NoopEvent
        self._lock = threading.Lock()
        self._in_flight: deque[tuple[object, list[int]]] = deque()

    def can_accept(self, seq: Sequence, num_speculative_tokens: int = 0) -> bool:
        """Whether D's BlockManager has enough free blocks to receive `seq` and run its first decode round."""
        needed = (len(seq) + num_speculative_tokens + self.d_bm.block_size - 1) // self.d_bm.block_size
        return len(self.d_bm.free_block_ids) >= max(len(seq.target_kv.block_table), needed)

    def reclaim_completed(self, *, wait_all: bool = False) -> int:
        """Return P blocks whose async copy has completed back to `p_bm` (P thread / teardown only). Returns blocks freed."""
        freed = 0
        while True:
            with self._lock:
                if not self._in_flight:
                    break
                event, p_blocks = self._in_flight[0]
                if not wait_all and not event.query():
                    break
                self._in_flight.popleft()
            if wait_all:
                event.synchronize()
            self._free_p_blocks(p_blocks)
            freed += len(p_blocks)
        return freed

    def handoff(
        self,
        seq: Sequence,
        prefill_done_event=None,
        d_alloc_event=None,
    ) -> object:
        """Copy `seq`'s KV blocks from P to D, remap `seq.target_kv.block_table` to D's blocks, and return `copy_done_event`."""
        p_blocks = list(seq.target_kv.block_table)
        cached_tokens = seq.target_kv.num_cached_tokens
        assert p_blocks and cached_tokens == len(seq) - 1

        # Detach `seq` from P's block table without freeing `p_blocks` until the async copy completes.
        # When `len(prompt)` is an exact multiple of `block_size`, the first token appended after prefill
        # pushes `len(seq)` into one extra block on D (`len(d_blocks) == len(p_blocks) + 1`).
        seq.target_kv.block_table.clear()
        seq.target_kv.num_cached_tokens = 0
        self.d_bm.allocate(seq, 0)
        d_blocks = list(seq.target_kv.block_table)
        assert len(p_blocks) <= len(d_blocks) <= len(p_blocks) + 1
        seq.target_kv.num_cached_tokens = cached_tokens
        seq.is_prefill = False

        p_ctx = torch.cuda.stream(self.transfer_stream) if self.transfer_stream is not None else nullcontext()
        d_ctx = torch.cuda.stream(self.d_transfer_stream) if self.d_transfer_stream is not None else nullcontext()
        dst_stream = self.d_transfer_stream or self.transfer_stream
        with p_ctx, d_ctx:
            if self.transfer_stream is not None and prefill_done_event is not None:
                self.transfer_stream.wait_event(prefill_done_event)
            if dst_stream is not None and d_alloc_event is not None:
                dst_stream.wait_event(d_alloc_event)
            if self.d_transfer_stream is not None and self.transfer_stream is not None:
                self.d_transfer_stream.wait_stream(self.transfer_stream)
            for p_blk, d_blk in zip(p_blocks, d_blocks):
                self.d_kv_cache[:, :, d_blk].copy_(self.p_kv_cache[:, :, p_blk], non_blocking=True)
                if self.p_draft_kv_cache is not None:
                    self.d_draft_kv_cache[:, :, d_blk].copy_(self.p_draft_kv_cache[:, :, p_blk], non_blocking=True)
            if self.d_transfer_stream is not None and self.transfer_stream is not None:
                self.transfer_stream.wait_stream(self.d_transfer_stream)
            copy_done = self._new_event()
            if dst_stream is not None:
                copy_done.record(dst_stream)
            else:
                copy_done.record()

        with self._lock:
            self._in_flight.append((copy_done, p_blocks))
        return copy_done

    def _free_p_blocks(self, p_blocks: list[int]) -> None:
        for block_id in reversed(p_blocks):
            block = self.p_bm.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.p_bm._deallocate_block(block_id)


class _NoopEvent:
    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass
