"""Block-remapped KV cache transfer between a Prefill engine and a Decode engine."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import threading

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

DEFAULT_STAGE_BLOCKS_PEER = 96
DEFAULT_STAGE_BLOCKS_COLOCATED = 64
INITIAL_PACK_BLOCKS = 16          # covers default max_model_len = 4096 without runtime growth


def _index_tensor(block_ids: list[int], device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.tensor(block_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(block_ids, dtype=torch.int64, device=device)


@dataclass
class StagedHandoff:
    """One sequence's KV blocks packed on P and DMA-transferred into D's contiguous staging ring."""

    slot_start: int
    num_blocks: int
    cached_tokens: int
    dma_done: object

    def query(self) -> bool:
        return self.dma_done.query()

    def synchronize(self) -> None:
        self.dma_done.synchronize()


class KVCacheTransfer:
    """Transfers per-sequence KV blocks from Prefill (P) to Decode (D) via a contiguous staging ring.

    Thread & Device Ownership (eliminating cross-thread `cuda:0` context contention and 32-bit slice splitting):
    1. `begin_handoff` runs on P's thread right after `prefill_step`:
       - `transfer_stream.wait_event(prefill_done_event)` waits on GPU 0 for P's prefill writes.
       - One `torch.index_select` per cache packs scattered `p_blocks` into a contiguous buffer on P (`_p_pack`),
         followed by a single contiguous `copy_(non_blocking=True)` (`cudaMemcpyPeerAsync` DMA) into D's staging ring
         (`_d_stage`). When P and D share a device, `index_select` writes directly into `_d_stage`.
       - `p_done` is recorded on `transfer_stream` (`p_device`) so P reclaims `p_blocks` on its own device without
         waiting for D to admit the sequence; `dma_done` is recorded on `d_transfer_stream` (`d_device`) so D queries
         and destroys only `d_device` events.
    2. `end_handoff` runs on D's thread inside `decode_step` once `staged.query()` and `can_accept` succeed:
       - Allocates `d_blocks` from `d_bm` exclusively on D's thread (never touching `p_device` or `p_bm`).
       - Unpacks `_d_stage -> d_kv_cache` with one `index_copy_` per cache on `d_stream`, then records `unpack_done`
         on `d_stream` so D reclaims the staging slot once D's unpack kernel retires.
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
        max_stage_blocks: int | None = None,
        max_pack_blocks: int | None = None,
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
        self.p_device = p_kv_cache.device
        self.d_device = d_kv_cache.device
        if event_factory is not None:
            self._new_event = event_factory
        elif self.p_device.type == "cuda":
            self._new_event = lambda: torch.cuda.Event()
        else:
            self._new_event = _NoopEvent

        self._cache_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [(p_kv_cache, d_kv_cache)]
        if p_draft_kv_cache is not None and d_draft_kv_cache is not None:
            self._cache_pairs.append((p_draft_kv_cache, d_draft_kv_cache))
        self._block_elems = [p_cache[:, :, 0].numel() for p_cache, _ in self._cache_pairs]

        default_stage = DEFAULT_STAGE_BLOCKS_PEER if self.p_device != self.d_device else DEFAULT_STAGE_BLOCKS_COLOCATED
        self._stage_cap = min(d_kv_cache.shape[2], max_stage_blocks or default_stage)
        d_ctx = torch.cuda.stream(self.d_transfer_stream) if self.d_transfer_stream is not None else nullcontext()
        with d_ctx:
            if self.d_device.type == "cuda":
                torch.cuda.set_device(self.d_device)
            self._d_stage = [
                torch.empty(self._stage_cap * elems, dtype=d_cache.dtype, device=self.d_device)
                for (_, d_cache), elems in zip(self._cache_pairs, self._block_elems)
            ]

        pack_blocks = max(INITIAL_PACK_BLOCKS, max_pack_blocks or 0)
        self._pack_cap = min(p_kv_cache.shape[2], pack_blocks) if self.p_device != self.d_device else 0
        p_ctx = torch.cuda.stream(self.transfer_stream) if self.transfer_stream is not None else nullcontext()
        with p_ctx:
            if self.p_device.type == "cuda":
                torch.cuda.set_device(self.p_device)
            self._p_pack = (
                [
                    torch.empty(self._pack_cap * elems, dtype=p_cache.dtype, device=self.p_device)
                    for (p_cache, _), elems in zip(self._cache_pairs, self._block_elems)
                ]
                if self._pack_cap > 0
                else None
            )

        self._lock = threading.Lock()
        self._write_pos = 0
        self._occupied_stage: list[tuple[int, int]] = []
        self._in_flight_p: deque[tuple[object, list[int]]] = deque()
        self._in_flight_stage: deque[tuple[object, tuple[int, int]]] = deque()

    def _find_stage_slot_unlocked(self, num_blocks: int) -> int | None:
        if num_blocks > self._stage_cap:
            raise ValueError(f"Sequence requires {num_blocks} KV blocks, exceeding staging capacity {self._stage_cap}")
        for cand in (self._write_pos, 0):
            if cand + num_blocks <= self._stage_cap and all(
                end <= cand or start >= cand + num_blocks for start, end in self._occupied_stage
            ):
                return cand
        return None

    def can_stage(self, num_blocks: int) -> bool:
        """Whether D's contiguous staging ring currently has room for `num_blocks` contiguous blocks."""
        with self._lock:
            if self._find_stage_slot_unlocked(num_blocks) is not None:
                return True
            has_in_flight = bool(self._in_flight_stage)
        if not has_in_flight:
            return False
        self.reclaim_stage()
        with self._lock:
            return self._find_stage_slot_unlocked(num_blocks) is not None

    def can_accept(self, seq: Sequence, num_speculative_tokens: int = 0) -> bool:
        """Whether D's BlockManager has enough free blocks to receive `seq` and run its first decode round."""
        needed = (len(seq) + num_speculative_tokens + self.d_bm.block_size - 1) // self.d_bm.block_size
        return len(self.d_bm.free_block_ids) >= max(len(seq.target_kv.block_table), needed)

    def reclaim_p(self, *, wait_all: bool = False) -> int:
        """Return P blocks whose `p_device` DMA-source events have completed (P thread / teardown)."""
        freed = 0
        while True:
            with self._lock:
                if not self._in_flight_p:
                    break
                event, p_blocks = self._in_flight_p[0]
                if not wait_all and not event.query():
                    break
                self._in_flight_p.popleft()
            if wait_all:
                event.synchronize()
            self._free_p_blocks(p_blocks)
            freed += len(p_blocks)
        return freed

    def reclaim_stage(self, *, wait_all: bool = False) -> int:
        """Return D staging-ring slots whose `d_device` unpack events have completed (D thread / teardown)."""
        freed = 0
        while True:
            with self._lock:
                if not self._in_flight_stage:
                    break
                event, interval = self._in_flight_stage[0]
                if not wait_all and not event.query():
                    break
                self._in_flight_stage.popleft()
            if wait_all:
                event.synchronize()
            with self._lock:
                if interval in self._occupied_stage:
                    self._occupied_stage.remove(interval)
                    freed += interval[1] - interval[0]
        return freed

    def reclaim_completed(self, *, wait_all: bool = False) -> int:
        """Reclaim both P blocks and D staging-ring slots whose GPU events have completed."""
        freed = self.reclaim_p(wait_all=wait_all)
        self.reclaim_stage(wait_all=wait_all)
        return freed

    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._in_flight_p or self._in_flight_stage or self._occupied_stage)

    def reset_staging(self) -> None:
        """Clear staging-ring occupancy after all in-flight transfers have been synchronized."""
        with self._lock:
            self._in_flight_stage.clear()
            self._occupied_stage.clear()
            self._write_pos = 0

    def _stage_view(self, flat: torch.Tensor, template: torch.Tensor, start: int, num_blocks: int, elems: int) -> torch.Tensor:
        chunk = flat[start * elems : (start + num_blocks) * elems]
        return chunk.view(template.shape[0], template.shape[1], num_blocks, *template.shape[3:])

    def _ensure_pack_capacity(self, num_blocks: int) -> None:
        if self._p_pack is None or num_blocks <= self._pack_cap:
            return
        self._pack_cap = num_blocks
        self._p_pack = [
            torch.empty(num_blocks * elems, dtype=p_cache.dtype, device=self.p_device)
            for (p_cache, _), elems in zip(self._cache_pairs, self._block_elems)
        ]

    def begin_handoff(self, seq: Sequence, prefill_done_event=None) -> StagedHandoff:
        """Pack `seq`'s blocks on P and launch one contiguous P2P DMA per cache into D's staging ring (P thread)."""
        p_blocks = list(seq.target_kv.block_table)
        num_blocks = len(p_blocks)
        cached_tokens = seq.target_kv.num_cached_tokens
        assert p_blocks and cached_tokens == len(seq) - 1

        with self._lock:
            slot_start = self._find_stage_slot_unlocked(num_blocks)
            if slot_start is None:
                raise RuntimeError(f"Staging ring full for {num_blocks} blocks; check can_stage() before begin_handoff()")
            self._occupied_stage.append((slot_start, slot_start + num_blocks))
            self._write_pos = slot_start + num_blocks

        seq.target_kv.block_table.clear()
        seq.target_kv.num_cached_tokens = 0
        seq.is_prefill = False

        p_ctx = torch.cuda.stream(self.transfer_stream) if self.transfer_stream is not None else nullcontext()
        d_ctx = torch.cuda.stream(self.d_transfer_stream) if self.d_transfer_stream is not None else nullcontext()
        dst_stream = self.d_transfer_stream or self.transfer_stream
        with p_ctx, d_ctx:
            if self.p_device.type == "cuda":
                torch.cuda.set_device(self.p_device)
            if self.transfer_stream is not None and prefill_done_event is not None:
                self.transfer_stream.wait_event(prefill_done_event)
            self._ensure_pack_capacity(num_blocks)
            p_idx = _index_tensor(p_blocks, self.p_device)
            for idx, ((p_cache, d_cache), elems) in enumerate(zip(self._cache_pairs, self._block_elems)):
                d_view = self._stage_view(self._d_stage[idx], d_cache, slot_start, num_blocks, elems)
                if self._p_pack is None:
                    torch.index_select(p_cache, 2, p_idx, out=d_view)
                else:
                    p_view = self._stage_view(self._p_pack[idx], p_cache, 0, num_blocks, elems)
                    torch.index_select(p_cache, 2, p_idx, out=p_view)
                    d_view.copy_(p_view, non_blocking=True)
            if self.d_transfer_stream is not None and self.transfer_stream is not None:
                self.transfer_stream.wait_stream(self.d_transfer_stream)
            if self.d_device != self.p_device:
                p_done = self._new_event()
                if self.transfer_stream is not None:
                    p_done.record(self.transfer_stream)
                else:
                    p_done.record()
                if dst_stream is None and self.d_device.type == "cuda":
                    torch.cuda.set_device(self.d_device)
                dma_done = self._new_event()
                if dst_stream is not None:
                    dma_done.record(dst_stream)
                else:
                    dma_done.record()
            else:
                dma_done = self._new_event()
                if dst_stream is not None:
                    dma_done.record(dst_stream)
                else:
                    dma_done.record()
                p_done = dma_done
        if self.p_device.type == "cuda":
            torch.cuda.set_device(self.p_device)

        with self._lock:
            self._in_flight_p.append((p_done, p_blocks))
        return StagedHandoff(slot_start=slot_start, num_blocks=num_blocks, cached_tokens=cached_tokens, dma_done=dma_done)

    def end_handoff(
        self,
        seq: Sequence,
        staged: StagedHandoff,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> object:
        """Allocate D blocks for `seq` and unpack `staged` into D's KV cache on D's device (D thread only)."""
        self.d_bm.allocate(seq, 0)
        d_blocks = seq.target_kv.block_table[: staged.num_blocks]
        assert staged.num_blocks <= len(seq.target_kv.block_table) <= staged.num_blocks + 1
        seq.target_kv.num_cached_tokens = staged.cached_tokens

        unpack_stream = stream if stream is not None else (self.d_transfer_stream or self.transfer_stream)
        u_ctx = torch.cuda.stream(unpack_stream) if unpack_stream is not None else nullcontext()
        with u_ctx:
            if self.d_device.type == "cuda":
                torch.cuda.set_device(self.d_device)
            if unpack_stream is not None:
                unpack_stream.wait_event(staged.dma_done)
            d_idx = _index_tensor(d_blocks, self.d_device)
            for idx, ((_, d_cache), elems) in enumerate(zip(self._cache_pairs, self._block_elems)):
                d_view = self._stage_view(self._d_stage[idx], d_cache, staged.slot_start, staged.num_blocks, elems)
                d_cache.index_copy_(2, d_idx, d_view)
            unpack_done = self._new_event()
            if unpack_stream is not None:
                unpack_done.record(unpack_stream)
            else:
                unpack_done.record()

        with self._lock:
            self._in_flight_stage.append((unpack_done, (staged.slot_start, staged.slot_start + staged.num_blocks)))
        return unpack_done

    def handoff(
        self,
        seq: Sequence,
        prefill_done_event=None,
    ) -> object:
        """Single-call convenience wrapper executing `begin_handoff` followed immediately by `end_handoff`."""
        staged = self.begin_handoff(seq, prefill_done_event)
        return self.end_handoff(seq, staged)

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
