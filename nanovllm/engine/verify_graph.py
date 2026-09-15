"""Bounded, exact-shape TP1 paged VERIFY graphs (no sampling or scheduler).

Only uniform q=2..5 and B=1..4 are admitted. Mixed lengths use the old path;
there are NO dummy requests/tokens and no changed candidate budget. The FA
max_seqlen_k and page-table stride are fixed, while cu_k contains ACTUAL lengths.
Each entry owns a PRIVATE graph pool. Entries may replay in arbitrary order.
Returned logits are cloned: callers must never retain a reusable graph output.

First use may capture and is explicitly counted/timed. Call freeze() after
natural warmup to forbid captures in a measured generate. Capture errors abort;
only unsupported shapes, frozen misses and pre-admission resource limits fall
back. No graph/compile failure is silently retried through another backend.
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import torch

from nanovllm.utils.context import get_context, set_context, reset_context


@dataclass(frozen=True)
class VerifyMetadata:
    batch: int
    query_len: int
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slots: tuple[int, ...]
    cu_q: tuple[int, ...]
    cu_k: tuple[int, ...]
    block_tables: tuple[int, ...]  # row-major with FIXED page stride
    page_stride: int


def exact_verify_key(queries, *, max_batch=4):
    """Metadata only. Rejection here means use the existing eager path."""
    if not queries or len(queries) > max_batch:
        return None
    lengths = [q.num_scheduled_tokens for q in queries]
    if lengths[0] not in (2, 3, 4, 5) or any(n != lengths[0] for n in lengths):
        return None
    if any(q.num_cached_tokens <= 0 for q in queries):
        return None
    return len(queries), lengths[0]


def build_verify_metadata(queries, *, block_size, max_model_len, num_blocks, vocab_size):
    key = exact_verify_key(queries)
    if key is None:
        raise ValueError("exact VERIFY needs B1..4, uniform q2..5, cached prefixes")
    if min(block_size, max_model_len, num_blocks, vocab_size) <= 0:
        raise ValueError("invalid VERIFY capacities")
    b, qlen = key
    stride = (max_model_len + block_size - 1) // block_size
    ids, positions, slots, tables = [], [], [], []
    cu_q, cu_k = [0], [0]
    for query in queries:
        start, end = query.num_cached_tokens, len(query)
        if not 0 < start < end <= max_model_len or end - start != qlen:
            raise ValueError("invalid VERIFY prefix/query length")
        needed = (end + block_size - 1) // block_size
        table = query.block_table
        if len(table) < needed or len(table) > stride:
            raise ValueError("VERIFY block-table capacity mismatch")
        if any(type(p) is not int or not 0 <= p < num_blocks for p in table):
            raise ValueError("invalid physical KV block")
        tail = list(query[start:end])
        if len(tail) != qlen or any(type(t) is not int or not 0 <= t < vocab_size for t in tail):
            raise ValueError("invalid VERIFY input token")
        ids.extend(tail)
        positions.extend(range(start, end))
        slots.extend(table[p // block_size] * block_size + p % block_size for p in range(start, end))
        # Unused entries are legal block 0. FA must not read past actual cu_k.
        tables.extend(list(table) + [0] * (stride - len(table)))
        cu_q.append(cu_q[-1] + qlen)
        cu_k.append(cu_k[-1] + end)  # NOT qlen, max_model_len, or physical capacity
    if cu_k[-1] >= 2**31:
        raise ValueError("int32 cumulative KV length overflow")
    return VerifyMetadata(b, qlen, tuple(ids), tuple(positions), tuple(slots),
                          tuple(cu_q), tuple(cu_k), tuple(tables), stride)


def _restore_context(c):
    set_context(c.is_prefill, c.cu_seqlens_q, c.cu_seqlens_k,
                c.max_seqlen_q, c.max_seqlen_k, c.slot_mapping,
                c.context_lens, c.block_tables)


class _Entry:
    def __init__(self, runner, metadata):
        self.batch, self.query_len = metadata.batch, metadata.query_len
        self.max_seqlen_k = runner.config.max_model_len
        self.page_stride = metadata.page_stride
        param = runner.model.lm_head.weight
        self.device, self.dtype = param.device, param.dtype
        self.kv_ptr = runner.kv_cache.data_ptr()
        self.kv_shape = tuple(runner.kv_cache.shape)
        self.kv_stride = tuple(runner.kv_cache.stride())
        self.head_ptr = param.data_ptr()
        n, b = self.batch * self.query_len, self.batch
        # Explicit device/dtype: lazy capture happens after global defaults reset.
        self.long_buffer = torch.empty(2 * n, device=self.device, dtype=torch.int64)
        self.ids, self.positions = self.long_buffer[:n], self.long_buffer[n:]
        self.int_buffer = torch.empty(n + 2 * (b + 1) + b * self.page_stride,
                                      device=self.device, dtype=torch.int32)
        at = 0
        self.slots = self.int_buffer[at:at+n]; at += n
        self.cu_q = self.int_buffer[at:at+b+1]; at += b+1
        self.cu_k = self.int_buffer[at:at+b+1]; at += b+1
        self.tables = self.int_buffer[at:].view(b, self.page_stride)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None
        self.capture_seconds = 0.0
        self.update(metadata)

    def update(self, m):
        if (m.batch, m.query_len, m.page_stride) != (self.batch, self.query_len, self.page_stride):
            raise ValueError("VERIFY graph metadata shape/stride changed")
        # Fresh pinned allocations avoid reusing a host buffer before its H2D
        # completed. PyTorch's pinned allocator tracks outstanding async copies.
        long_host = torch.tensor(m.input_ids + m.positions, dtype=torch.int64,
                                 device="cpu", pin_memory=True)
        int_host = torch.tensor(m.slots + m.cu_q + m.cu_k + m.block_tables,
                                dtype=torch.int32, device="cpu", pin_memory=True)
        self.long_buffer.copy_(long_host, non_blocking=True)
        self.int_buffer.copy_(int_host, non_blocking=True)

    def context(self):
        set_context(True, self.cu_q, self.cu_k, self.query_len,
                    self.max_seqlen_k, self.slots, None, self.tables)

    def check_binding(self, runner):
        p, kv = runner.model.lm_head.weight, runner.kv_cache
        if (p.device != self.device or p.dtype != self.dtype or p.data_ptr() != self.head_ptr
                or kv.device != self.device or kv.dtype != self.dtype or kv.data_ptr() != self.kv_ptr
                or tuple(kv.shape) != self.kv_shape or tuple(kv.stride()) != self.kv_stride
                or runner.config.max_model_len != self.max_seqlen_k):
            raise RuntimeError("VERIFY graph storage/device/dtype binding changed")

    def compute(self, runner):
        self.context()
        return runner.model.compute_logits(runner.model(self.ids, self.positions), all_logits=True)

    def close(self):
        self.output = None
        self.graph.reset()


class VerifyGraphCache:
    def __init__(self, runner, *, reserve_budget_bytes=2 << 30, min_free_bytes=1 << 30):
        if runner.world_size != 1 or runner.enforce_eager or not runner._cache_initialized:
            raise ValueError("VERIFY graphs require an initialized TP1 Graph-capable runner")
        if any(type(x) is not int or x <= 0 for x in (reserve_budget_bytes, min_free_bytes)):
            raise ValueError("positive VERIFY graph memory guard required")
        if runner.model.lm_head.weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("VERIFY graph package supports FP16/BF16 FA2 inference only")
        if (runner.kv_cache.device != runner.model.lm_head.weight.device
                or runner.kv_cache.dtype != runner.model.lm_head.weight.dtype):
            raise ValueError("VERIFY model/KV device or dtype mismatch")
        self.runner = runner
        self.device = runner.kv_cache.device
        self.max_batch = min(4, runner.config.max_num_seqs)
        self.reserve_budget_bytes = reserve_budget_bytes
        self.min_free_bytes = min_free_bytes
        # A process-reserved-growth admission ceiling, NOT an exact per-graph
        # allocator fence or a bound on capture peak/driver memory. OOM is fatal.
        self.reserved_at_enable = torch.cuda.memory_reserved(self.device)
        self.entries = {}
        self.blocked = {}
        self.frozen = False
        self.enabled = True
        self.counts = Counter()
        self.capture_seconds = 0.0
        self.memory_checks = []  # capture-only evidence; no replay-time polling
        self.owner_stream = torch.cuda.current_stream(self.device).cuda_stream

    def freeze(self):
        """No new captures after warmup; a missing key uses original eager."""
        self.frozen = True

    def statistics(self):
        return {"enabled": self.enabled, "frozen": self.frozen,
                "keys": [list(k) for k in sorted(self.entries)],
                "blocked": {str(k): v for k, v in self.blocked.items()},
                "counts": dict(self.counts), "capture_seconds": self.capture_seconds,
                "max_seqlen_k": self.runner.config.max_model_len,
                "page_stride": (self.runner.config.max_model_len + self.runner.block_size - 1) // self.runner.block_size,
                "reserve_budget_bytes": self.reserve_budget_bytes,
                "min_free_bytes": self.min_free_bytes,
                "reserved_at_enable": self.reserved_at_enable,
                "memory_checks": list(self.memory_checks)}

    def _memory_ok(self, *, key, phase):
        # The budget is PROCESS reserved growth, not exact Graph-pool usage.
        # A released test/prefill tensor may still occupy allocator cache.
        # Only on a capture admission failure, reclaim unused cache ONCE and
        # recheck the SAME limits. Never run this on replay or a frozen miss.
        def snapshot():
            free, total = torch.cuda.mem_get_info(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            limit = self.reserved_at_enable + self.reserve_budget_bytes
            violations = []
            if free < self.min_free_bytes:
                violations.append("device_free_floor")
            if reserved > limit:
                violations.append("reserved_growth_limit")
            return {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes": reserved, "device_free_bytes": free,
                    "device_total_bytes": total, "reserved_limit_bytes": limit,
                    "min_free_bytes": self.min_free_bytes, "violations": violations}

        before = snapshot()
        after = None
        if before["violations"]:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                # Does not release live tensors or live Graph-private pools.
                torch.cuda.empty_cache()
            self.counts["memory_guard_reclaims"] += 1
            after = snapshot()
        final = before if after is None else after
        ok = not final["violations"]
        self.memory_checks.append({"key": list(key), "phase": phase,
                                   "before": before, "after_reclaim": after,
                                   "ok": ok})
        return ok

    @torch.inference_mode()
    def run(self, queries):
        key = exact_verify_key(queries, max_batch=self.max_batch)
        if not self.enabled or key is None:
            self.counts["unsupported_or_disabled"] += 1
            return None
        if torch.cuda.current_stream(self.device).cuda_stream != self.owner_stream:
            raise RuntimeError("VERIFY cache is single-stream; cross-stream replay is not supported")
        runner = self.runner
        metadata = build_verify_metadata(
            queries, block_size=runner.block_size, max_model_len=runner.config.max_model_len,
            num_blocks=runner.config.num_kvcache_blocks, vocab_size=runner.config.hf_config.vocab_size)
        entry = self.entries.get(key)
        if entry is None:
            if self.frozen or key in self.blocked:
                self.counts["frozen_or_blocked_miss"] += 1
                return None
            if not self._memory_ok(key=key, phase="pre_capture"):
                self.blocked[key] = "pre_capture_memory_guard"
                self.counts["resource_fallback"] += 1
                return None
            before_context = get_context()
            started = perf_counter()
            entry = _Entry(runner, metadata)
            try:
                # Warm on current stream before side-stream capture. All repeated
                # writes are to THIS query's uncached suffix. Valid prefixes and
                # block ownership do not change; no pool allocation/commit here.
                for _ in range(3):
                    scratch = entry.compute(runner)
                    del scratch
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        scratch = entry.compute(runner)
                        del scratch
                stream.synchronize()
                # No shared graph_pool: arbitrary B/q replay order is safe.
                with torch.cuda.graph(entry.graph, stream=stream):
                    entry.output = entry.compute(runner)
                stream.synchronize()
                if (entry.output.shape != (key[0] * key[1], runner.config.hf_config.vocab_size)
                        or entry.output.device != entry.device or entry.output.dtype != entry.dtype):
                    raise RuntimeError("captured VERIFY logits layout mismatch")
                if not self._memory_ok(key=key, phase="post_capture"):
                    raise MemoryError("VERIFY capture exceeded reserved-growth/free-memory guard: "
                                      + repr(self.memory_checks[-1]))
                entry.capture_seconds = perf_counter() - started
                self.entries[key] = entry
                self.capture_seconds += entry.capture_seconds
                self.counts["captures"] += 1
            except BaseException as error:
                # Do not silently fall back after a potentially failed capture.
                # RandomLLM closes the failed engine; no partial output escapes.
                try:
                    entry.close()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise
            finally:
                _restore_context(before_context)
        else:
            entry.check_binding(runner)
            entry.update(metadata)
        entry.graph.replay()
        self.counts["replays"] += 1
        self.counts[f"replay_B{key[0]}_Q{key[1]}"] += 1
        # Independent lifetime, matching the old run_queries logits contract.
        return list(entry.output.clone().split(key[1]))

    def close(self):
        if self.runner is None:
            return
        errors = []
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as error:
            errors.append(error)
        for entry in self.entries.values():
            try:
                entry.close()
            except BaseException as error:
                errors.append(error)
        self.entries.clear()
        self.runner = None
        reset_context()
        if errors:
            raise errors[0]
