"""Single-process, two-thread Prefill-Decode (PD) disaggregated engine."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import threading
from time import perf_counter
import torch

from nanovllm.disagg.kv_transfer import KVCacheTransfer, _NoopEvent
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

_WORKER_POLL_INTERVAL_S = 0.0005


class DisaggLLM:
    """Composes a Prefill-only `LLMEngine` (P) and a Decode-only `LLMEngine` (D) connected by `KVCacheTransfer`.

    - `prefill_engine` runs with `enforce_eager=True` on `prefill_device` and is mutated only by `prefill_step`.
      Completed prefills move from `prefill_engine.scheduler.running` to `_pending_handoffs`.
    - `decode_engine` runs with CUDA Graphs enabled on `decode_device` and is mutated only by `decode_step`.
      Handed-off sequences enter `decode_engine.scheduler.running` directly, eliminating prefill stalls during decode.
    """

    def __init__(
        self,
        model: str,
        *,
        prefill_device: int = 0,
        decode_device: int = 0,
        prefill_engine_cls=None,
        decode_engine_cls=None,
        use_cuda_streams: bool = True,
        event_factory=None,
        **kwargs,
    ) -> None:
        if prefill_device == decode_device and kwargs.get("kv_cache_memory_bytes") is None and (
            prefill_engine_cls is None or decode_engine_cls is None
        ):
            raise ValueError(
                "Single-GPU DisaggLLM (prefill_device == decode_device) requires an explicit kv_cache_memory_bytes"
            )
        if prefill_engine_cls is None or decode_engine_cls is None:
            from nanovllm.engine.llm_engine import LLMEngine
            prefill_engine_cls = prefill_engine_cls or LLMEngine
            decode_engine_cls = decode_engine_cls or LLMEngine
        p_kwargs = {**kwargs, "device_id": prefill_device, "enforce_eager": True}
        d_seed = None if kwargs.get("seed") is None else kwargs["seed"] + 1
        d_kwargs = {**kwargs, "device_id": decode_device, "seed": d_seed}
        self.prefill_engine = prefill_engine_cls(model, **p_kwargs)
        self.decode_engine = decode_engine_cls(model, **d_kwargs)
        self.tokenizer = self.decode_engine.tokenizer

        self.p_device = self.prefill_engine.model_runner.device
        self.d_device = self.decode_engine.model_runner.device
        if use_cuda_streams and self.p_device.type == "cuda":
            self.p_stream = torch.cuda.Stream(device=self.p_device)
            self.transfer_stream = torch.cuda.Stream(device=self.p_device)
            self.d_transfer_stream = (
                torch.cuda.Stream(device=self.d_device) if self.d_device != self.p_device else None
            )
            self.d_stream = torch.cuda.Stream(device=self.d_device)
            self._new_event = event_factory or (lambda: torch.cuda.Event())
        else:
            self.p_stream = self.transfer_stream = self.d_transfer_stream = self.d_stream = None
            self._new_event = event_factory or _NoopEvent

        p_draft_kv = self.prefill_engine.speculative.drafter.kv_cache if self.prefill_engine.speculative else None
        d_draft_kv = self.decode_engine.speculative.drafter.kv_cache if self.decode_engine.speculative else None
        self.transfer = KVCacheTransfer(
            self.prefill_engine.model_runner.kv_cache,
            self.decode_engine.model_runner.kv_cache,
            self.prefill_engine.scheduler.block_manager,
            self.decode_engine.scheduler.block_manager,
            p_draft_kv_cache=p_draft_kv,
            d_draft_kv_cache=d_draft_kv,
            transfer_stream=self.transfer_stream,
            d_transfer_stream=self.d_transfer_stream,
            event_factory=self._new_event,
        )

        self._lock = threading.Lock()
        self._arrival_queue: deque[tuple[str | list[int], SamplingParams]] = deque()
        self._max_in_flight: int = 0
        self._in_flight_count: int = 0
        self._total_expected: int = 0
        self._submitted_seq_ids: list[int] = []
        self._pending_handoffs: deque[tuple[Sequence, object]] = deque()
        self._ready_for_decode: deque[tuple[Sequence, object]] = deque()
        self._finished: dict[int, tuple[list[int], dict]] = {}
        self._req_meta: dict[int, dict] = {}

    def exit(self) -> None:
        import atexit
        self.transfer.reclaim_completed(wait_all=True)
        for eng in (self.decode_engine, self.prefill_engine):
            eng.exit()
            atexit.unregister(eng.exit)
            if getattr(eng, "speculative", None) is not None:
                eng.speculative = None

    def start_measurement(self) -> None:
        if hasattr(self.prefill_engine, "start_measurement"):
            self.prefill_engine.start_measurement()
        if hasattr(self.decode_engine, "start_measurement"):
            self.decode_engine.start_measurement()

    def _admit_arrivals_unlocked(self) -> None:
        """Pop queued arrivals into `prefill_engine` while under the closed-loop concurrency cap. Must run on P thread."""
        while self._arrival_queue and (self._max_in_flight <= 0 or self._in_flight_count < self._max_in_flight):
            prompt, sp = self._arrival_queue.popleft()
            seq = self.prefill_engine.add_request(prompt, sp)
            self._submitted_seq_ids.append(seq.seq_id)
            self._in_flight_count += 1
            self._req_meta[seq.seq_id] = {
                "enqueue_t": perf_counter(),
                "ttft_ms": None,
                "last_delivery_t": None,
                "itl_ms": [],
                "d_deliveries": [],
            }

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        with self._lock:
            seq = self.prefill_engine.add_request(prompt, sampling_params)
            self._submitted_seq_ids.append(seq.seq_id)
            self._in_flight_count += 1
            self._total_expected += 1
            self._req_meta[seq.seq_id] = {
                "enqueue_t": perf_counter(),
                "ttft_ms": None,
                "last_delivery_t": None,
                "itl_ms": [],
                "d_deliveries": [],
            }
            return seq

    def is_finished(self) -> bool:
        with self._lock:
            return len(self._finished) == self._total_expected and not self._arrival_queue

    def prefill_step(self) -> int:
        """Run one prefill batch on P if waiting requests exist, and move completed prefills to `_pending_handoffs`."""
        self.transfer.reclaim_completed()
        with self._lock:
            self._admit_arrivals_unlocked()
        p_sched = self.prefill_engine.scheduler
        if not p_sched.waiting:
            return 0
        first_waiting = p_sched.waiting[0]
        if not first_waiting.block_table and p_sched.block_manager.can_allocate(first_waiting) == -1:
            self.transfer.reclaim_completed(wait_all=True)
            if p_sched.block_manager.can_allocate(first_waiting) == -1:
                return 0

        ctx = torch.cuda.stream(self.p_stream) if self.p_stream is not None else nullcontext()
        with ctx:
            if self.p_device.type == "cuda":
                torch.cuda.set_device(self.p_device)
            outputs, num_tokens = self.prefill_engine.step()
            assert num_tokens > 0, "prefill_engine must only execute prefill steps"
            prefill_done = self._new_event()
            if self.p_stream is not None:
                prefill_done.record(self.p_stream)
            else:
                prefill_done.record()

        now = perf_counter()
        with self._lock:
            for seq_id, token_ids, _, _ in outputs:
                meta = self._req_meta[seq_id]
                meta["ttft_ms"] = (now - meta["enqueue_t"]) * 1e3
                self._finished[seq_id] = (token_ids, meta)
                self._in_flight_count -= 1
            while self.prefill_engine.scheduler.running:
                seq = self.prefill_engine.scheduler.running.popleft()
                meta = self._req_meta[seq.seq_id]
                meta["ttft_ms"] = (now - meta["enqueue_t"]) * 1e3
                meta["last_delivery_t"] = now
                self._pending_handoffs.append((seq, prefill_done))
            self._admit_arrivals_unlocked()
        return num_tokens

    def _flush_handoffs(self) -> int:
        """Transfer as many `_pending_handoffs` to `_ready_for_decode` as D's capacity and block budget allow."""
        transferred = 0
        d_sched = self.decode_engine.scheduler
        k = d_sched.num_speculative_tokens
        while True:
            with self._lock:
                if not self._pending_handoffs:
                    break
                seq, prefill_done = self._pending_handoffs[0]
                in_flight_d = len(d_sched.running) + len(self._ready_for_decode)
                if in_flight_d >= d_sched.max_num_seqs or not self.transfer.can_accept(seq, k):
                    break
                self._pending_handoffs.popleft()
            d_alloc_event = self._new_event()
            if self.d_stream is not None:
                d_alloc_event.record(self.d_stream)
            else:
                d_alloc_event.record()
            copy_done = self.transfer.handoff(seq, prefill_done, d_alloc_event)
            self._ready_for_decode.append((seq, copy_done))
            transferred += 1
        return transferred

    def decode_step(self) -> int:
        """Admit ready handoffs into D and run one decode / speculative round on D."""
        self._flush_handoffs()
        d_sched = self.decode_engine.scheduler
        while self._ready_for_decode and len(d_sched.running) < d_sched.max_num_seqs:
            seq, copy_done = self._ready_for_decode.popleft()
            if self.d_stream is not None:
                self.d_stream.wait_event(copy_done)
            seq.status = SequenceStatus.RUNNING
            d_sched.running.append(seq)
        if d_sched.waiting:
            raise RuntimeError(
                "Decode engine preempted a sequence due to insufficient KV cache blocks; "
                "increase kv_cache_memory_bytes or reduce max_num_seqs"
            )
        if not d_sched.running:
            return 0
        d_step = getattr(self.decode_engine, "num_steps", 0)
        before_counts = [(seq, seq.num_completion_tokens) for seq in d_sched.running]

        ctx = torch.cuda.stream(self.d_stream) if self.d_stream is not None else nullcontext()
        with ctx:
            if self.d_device.type == "cuda":
                torch.cuda.set_device(self.d_device)
            outputs, num_tokens = self.decode_engine.step()
            assert num_tokens < 0, "decode_engine must only execute decode or speculative rounds"

        now = perf_counter()
        with self._lock:
            for seq, prev_count in before_counts:
                meta = self._req_meta[seq.seq_id]
                delta = seq.num_completion_tokens - prev_count
                if delta > 0:
                    meta["itl_ms"].append((now - meta["last_delivery_t"]) * 1e3)
                    meta["last_delivery_t"] = now
                    meta["d_deliveries"].append((d_step, delta))
            for seq_id, token_ids, _, _ in outputs:
                self._finished[seq_id] = (token_ids, self._req_meta[seq_id])
                self._in_flight_count -= 1
        return -num_tokens

    def _precapture_sampling_graphs(self, prompts: list, sampling_params: list[SamplingParams]) -> None:
        """Pre-capture DSpark's sampling graphs on the main thread before worker threads start, up to MAX_SAMPLING_KEYS."""
        spec = self.decode_engine.speculative
        if spec is None or spec.drafter.model.markov_head is None or not spec.drafter.graphs or not prompts:
            return
        from nanovllm.engine.speculative import MAX_SAMPLING_KEYS, sampling_graph_key
        from nanovllm.layers.spec_sampler import sampling_tensors
        if self.d_device.type == "cuda":
            torch.cuda.set_device(self.d_device)
        max_bs = self.decode_engine.scheduler.max_num_seqs
        candidate_batches = [[Sequence([0], sp) for sp in sampling_params[: min(len(sampling_params), max_bs)]]]
        seen_params: set[tuple] = set()
        for sp in sampling_params:
            sig = (sp.temperature <= 1e-10, sp.top_k, sp.top_p < 1.0)
            if sig not in seen_params:
                seen_params.add(sig)
                candidate_batches.append([Sequence([0], sp)])
        for batch in candidate_batches:
            if len(spec.drafter.sampling_graphs) >= MAX_SAMPLING_KEYS:
                break
            key = sampling_graph_key(sampling_tensors(batch, self.d_device))
            if key not in spec.drafter.sampling_graphs:
                spec.drafter._capture_sampling_graphs(key)

    def step(self) -> tuple[int, int]:
        """Single-threaded step (runs one prefill step if available, then one decode step). Useful for deterministic tests."""
        p_tokens = self.prefill_step()
        d_tokens = self.decode_step()
        return p_tokens, d_tokens

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        *,
        threaded: bool = True,
        concurrency: int | None = None,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        with self._lock:
            self._finished.clear()
            self._req_meta.clear()
            self._arrival_queue.clear()
            self._submitted_seq_ids.clear()
            self._in_flight_count = 0
            self._total_expected = len(prompts)
            self._max_in_flight = concurrency or 0
            for p, sp in zip(prompts, sampling_params):
                self._arrival_queue.append((p, sp))
            self._admit_arrivals_unlocked()

        self._precapture_sampling_graphs(prompts, sampling_params)
        if not threaded:
            while not self.is_finished():
                self.step()
        else:
            stop_event = threading.Event()
            errors: list[BaseException] = []

            def p_worker() -> None:
                try:
                    while not stop_event.is_set() and not self.is_finished():
                        if self.prefill_step() == 0:
                            stop_event.wait(_WORKER_POLL_INTERVAL_S)
                except BaseException as exc:
                    errors.append(exc)
                    stop_event.set()

            def d_worker() -> None:
                try:
                    while not stop_event.is_set() and not self.is_finished():
                        if self.decode_step() == 0:
                            stop_event.wait(_WORKER_POLL_INTERVAL_S)
                except BaseException as exc:
                    errors.append(exc)
                    stop_event.set()

            t_p = threading.Thread(target=p_worker, name="pd-prefill", daemon=True)
            t_d = threading.Thread(target=d_worker, name="pd-decode", daemon=True)
            spec = self.decode_engine.speculative
            if spec is not None:
                spec.drafter.freeze_sampling_graphs = True
            try:
                t_p.start()
                t_d.start()
                t_p.join()
                t_d.join()
            finally:
                if spec is not None:
                    spec.drafter.freeze_sampling_graphs = False
            if errors:
                raise errors[0]

        self.transfer.reclaim_completed(wait_all=True)
        if self.p_stream is not None:
            self.p_stream.synchronize()
        if self.d_stream is not None:
            self.d_stream.synchronize()
        return [
            {
                "text": self.tokenizer.decode(token_ids) if self.tokenizer else "",
                "token_ids": token_ids,
                "ttft_ms": meta["ttft_ms"],
                "itl_ms": meta["itl_ms"],
                "d_deliveries": meta["d_deliveries"],
            }
            for token_ids, meta in (self._finished[seq_id] for seq_id in self._submitted_seq_ids)
        ]
