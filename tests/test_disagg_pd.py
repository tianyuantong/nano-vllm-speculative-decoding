"""Unit tests for lightweight PD disaggregation (`KVCacheTransfer` and `DisaggLLM`)."""
from __future__ import annotations

from types import SimpleNamespace
import threading
import pytest
import torch

from nanovllm.disagg import DisaggLLM, KVCacheTransfer
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


def test_thread_local_context_isolation():
    """Prefill thread and Decode thread setting Context concurrently must never overwrite each other."""
    reset_context()
    barrier = threading.Barrier(2)
    observed = {}

    def p_thread():
        set_context(True, max_seqlen_q=128, causal=True)
        barrier.wait()
        ctx = get_context()
        observed["p"] = (ctx.is_prefill, ctx.max_seqlen_q, ctx.causal)
        reset_context()

    def d_thread():
        set_context(False, query_len_per_seq=7, causal=False)
        barrier.wait()
        ctx = get_context()
        observed["d"] = (ctx.is_prefill, ctx.query_len_per_seq, ctx.causal)
        reset_context()

    t1 = threading.Thread(target=p_thread)
    t2 = threading.Thread(target=d_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert observed["p"] == (True, 128, True)
    assert observed["d"] == (False, 7, False)


def test_kv_cache_transfer_remaps_blocks_and_copies_target_and_draft_kv(monkeypatch):
    """`KVCacheTransfer.handoff` copies both Target KV and Draft KV from P's blocks to D's blocks and frees P's blocks on reclaim."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    p_bm = BlockManager(num_blocks=6, block_size=block_size, enable_prefix_cache=False)
    d_bm = BlockManager(num_blocks=6, block_size=block_size, enable_prefix_cache=False)

    # Occupy blocks 0 and 1 on D so D allocates non-trivial block IDs [2, 3] for the handoff sequence.
    dummy = Sequence([9, 9, 9, 9, 9], SamplingParams(max_tokens=4))
    d_bm.allocate(dummy, 0)
    assert dummy.target_kv.block_table == [0, 1]

    p_kv = torch.zeros(2, 2, 6, block_size, 2, 4)
    d_kv = torch.zeros_like(p_kv)
    p_draft_kv = torch.zeros(2, 1, 6, block_size, 2, 4)
    d_draft_kv = torch.zeros_like(p_draft_kv)

    # Sequence with 6 prompt tokens -> 2 blocks on P: [0, 1], plus 1st generated token appended after prefill.
    seq = Sequence([10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=5))
    p_bm.allocate(seq, 0)
    assert seq.target_kv.block_table == [0, 1]
    seq.target_kv.num_cached_tokens = 6
    seq.append_token(99)  # first token produced by prefill: len(seq) == 7, num_cached_tokens == 6

    # Fill P's blocks [0, 1] with distinct values.
    p_kv[:, :, 0] = 1.5
    p_kv[:, :, 1] = 2.5
    p_draft_kv[:, :, 0] = 3.5
    p_draft_kv[:, :, 1] = 4.5

    transfer = KVCacheTransfer(
        p_kv, d_kv, p_bm, d_bm, p_draft_kv_cache=p_draft_kv, d_draft_kv_cache=d_draft_kv
    )
    assert transfer.can_accept(seq, num_speculative_tokens=3)
    transfer.handoff(seq)

    # P's blocks remain held in `_in_flight` until `reclaim_completed()` is called by P.
    assert len(p_bm.free_block_ids) == 4
    assert transfer.reclaim_completed() == 2
    assert len(p_bm.free_block_ids) == 6

    assert seq.target_kv.block_table == [2, 3]
    assert seq.target_kv.num_cached_tokens == 6
    assert not seq.is_prefill
    assert torch.equal(d_kv[:, :, 2], p_kv[:, :, 0])
    assert torch.equal(d_kv[:, :, 3], p_kv[:, :, 1])
    assert torch.equal(d_draft_kv[:, :, 2], p_draft_kv[:, :, 0])
    assert torch.equal(d_draft_kv[:, :, 3], p_draft_kv[:, :, 1])


def test_kv_cache_transfer_exact_block_boundary(monkeypatch):
    """When `len(prompt)` is an exact multiple of `block_size`, D allocates one extra block for the 1st decode token."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    p_bm = BlockManager(num_blocks=6, block_size=block_size, enable_prefix_cache=False)
    d_bm = BlockManager(num_blocks=6, block_size=block_size, enable_prefix_cache=False)

    p_kv = torch.zeros(2, 1, 6, block_size, 1, 2)
    d_kv = torch.zeros_like(p_kv)

    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
    p_bm.allocate(seq, 0)
    assert seq.target_kv.block_table == [0]
    p_kv[:, :, 0] = 7.0
    seq.target_kv.num_cached_tokens = 4
    seq.append_token(50)  # len(seq) == 5 -> needs 2 blocks on D

    transfer = KVCacheTransfer(p_kv, d_kv, p_bm, d_bm)
    assert transfer.can_accept(seq)
    transfer.handoff(seq)
    transfer.reclaim_completed()

    assert seq.target_kv.block_table == [0, 1]
    assert seq.target_kv.num_cached_tokens == 4
    assert torch.equal(d_kv[:, :, 0], p_kv[:, :, 0])
    assert len(p_bm.free_block_ids) == 6


class _FakeEngine:
    """Minimal CPU fake of `LLMEngine` that writes deterministic token sums into its KV cache."""

    def __init__(self, model: str, **kwargs):
        self.block_size = 4
        Sequence.block_size = self.block_size
        cfg = SimpleNamespace(
            max_num_seqs=kwargs.get("max_num_seqs", 4),
            max_num_batched_tokens=kwargs.get("max_num_batched_tokens", 64),
            eos=999,
            kvcache_block_size=self.block_size,
            num_speculative_tokens=0,
            num_kvcache_blocks=16,
            enable_prefix_cache=False,
            round_robin_prefill=False,
        )
        self.scheduler = Scheduler(cfg)
        self.kv = torch.zeros(2, 1, 16, self.block_size, 1, 1)
        self.model_runner = SimpleNamespace(device=torch.device("cpu"), kv_cache=self.kv)
        self.speculative = None
        self.num_steps = 0
        self.tokenizer = SimpleNamespace(decode=lambda ids: ",".join(map(str, ids)))

    def exit(self):
        pass

    def add_request(self, prompt: list[int], sp: SamplingParams) -> Sequence:
        seq = Sequence(list(prompt), sp)
        self.scheduler.add(seq)
        return seq

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = []
        for seq in seqs:
            if is_prefill:
                for blk in seq.target_kv.block_table:
                    self.kv[:, :, blk] = float(sum(seq.prompt_token_ids))
                token_ids.append(100 + len(seq.prompt_token_ids))
            else:
                # Decode reads the first allocated block in D's KV cache to verify KV transfer happened.
                kv_val = int(self.kv[0, 0, seq.target_kv.block_table[0], 0, 0, 0].item())
                token_ids.append(kv_val + seq.num_completion_tokens)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.num_steps += 1
        outputs = [(s.seq_id, s.completion_token_ids, 0, 1) for s in seqs if s.is_finished]
        return outputs, num_tokens


def test_disagg_llm_end_to_end_single_and_multi_threaded(monkeypatch):
    """`DisaggLLM` produces identical token sequences in single-thread `step()` and two-thread `generate()` modes."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    prompts = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [10, 20], [7, 8, 9, 10, 11, 12]]
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    disagg = DisaggLLM("dummy", prefill_engine_cls=_FakeEngine, decode_engine_cls=_FakeEngine, use_cuda_streams=False)
    out_serial = disagg.generate(prompts, sp, threaded=False)
    out_threaded = disagg.generate(prompts, sp, threaded=True)
    disagg.exit()

    for s, t in zip(out_serial, out_threaded):
        assert s["token_ids"] == t["token_ids"]
        assert len(s["token_ids"]) == 4
        assert len(s["itl_ms"]) == 3
        assert s["ttft_ms"] > 0


class _SlowStream:
    """Stands for one CUDA stream: its events complete in recorded order, one per `POLLS_PER_EVENT` queries, and
    `synchronize()` on an event completes it and every earlier one."""

    POLLS_PER_EVENT = 3

    def __init__(self):
        self.num_recorded = 0
        self.num_done = 0
        self.num_polls = 0

    def new_event(self):
        stream = self

        class _Event:
            def record(self, _stream=None) -> None:
                stream.num_recorded += 1
                self.index = stream.num_recorded

            def query(self) -> bool:
                if self.index > stream.num_done:
                    stream.num_polls += 1
                    if stream.num_polls % stream.POLLS_PER_EVENT == 0:
                        stream.num_done += 1
                return self.index <= stream.num_done

            def synchronize(self) -> None:
                stream.num_done = max(stream.num_done, self.index)

        return _Event()


def test_disagg_llm_serial_step_waits_for_slow_copies(monkeypatch):
    """In single-thread mode the sequences of one prefill batch start decoding in the same round, however slow the copies."""
    monkeypatch.setattr(Sequence, "block_size", 4)

    class _RecordingEngine(_FakeEngine):
        def __init__(self, model: str, **kwargs):
            super().__init__(model, **kwargs)
            self.batch_sizes: list[int] = []

        def step(self):
            self.batch_sizes.append(len(self.scheduler.running))
            return super().step()

    prompts = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [10, 20], [7, 8, 9, 10, 11, 12]]
    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_RecordingEngine,
        use_cuda_streams=False,
        event_factory=_SlowStream().new_event,
    )
    out = disagg.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4), threaded=False)
    disagg.exit()

    assert disagg.decode_engine.batch_sizes[0] == len(prompts)
    assert all(len(o["token_ids"]) == 4 for o in out)


def test_disagg_llm_chunked_prefill_makes_progress(monkeypatch):
    """A chunked prefill sequence that already holds >50% of P's blocks must continue prefilling without livelocking."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_FakeEngine,
        use_cuda_streams=False,
        max_num_batched_tokens=8,
    )
    prompt = list(range(1, 42))  # 41 tokens -> 11 of P's 16 blocks, chunked at 8 tokens per prefill step
    out = disagg.generate([prompt], SamplingParams(temperature=0.0, max_tokens=3), threaded=True)
    disagg.exit()
    assert len(out[0]["token_ids"]) == 3


def test_disagg_llm_precaptures_dspark_sampling_graphs_before_threads(monkeypatch):
    """`DisaggLLM.generate(threaded=True)` pre-captures DSpark sampling graphs on the main thread before spawning workers."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    captured_on_threads: list[tuple[str, object]] = []

    disagg = DisaggLLM("dummy", prefill_engine_cls=_FakeEngine, decode_engine_cls=_FakeEngine, use_cuda_streams=False)
    fake_drafter = SimpleNamespace(
        model=SimpleNamespace(markov_head=object()),
        graphs={1: object()},
        sampling_graphs={},
        _capture_sampling_graphs=lambda key: (
            captured_on_threads.append((threading.current_thread().name, key)),
            fake_drafter.sampling_graphs.__setitem__(key, {1: object()}),
        ),
    )
    disagg.decode_engine.speculative = SimpleNamespace(drafter=fake_drafter)

    sps = [
        SamplingParams(temperature=0.8, top_k=20, top_p=0.9, max_tokens=2),
        SamplingParams(temperature=0.8, top_k=20, top_p=0.9, max_tokens=2),
    ]
    disagg.generate([[1, 2, 3], [4, 5, 6]], sps, threaded=True)
    disagg.exit()

    assert captured_on_threads == [
        (threading.main_thread().name, (20, True, True)),
    ]


def test_disagg_llm_worker_error_unblocks_peer_thread(monkeypatch):
    """An exception in `d_worker` sets `stop_event` and terminates `generate(threaded=True)` promptly without hanging."""
    monkeypatch.setattr(Sequence, "block_size", 4)

    class _FailingDecodeEngine(_FakeEngine):
        def step(self):
            raise RuntimeError("boom in decode")

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_FailingDecodeEngine,
        use_cuda_streams=False,
    )
    with pytest.raises(RuntimeError, match="boom in decode"):
        disagg.generate([[1, 2, 3], [4, 5, 6]], SamplingParams(temperature=0.0, max_tokens=3), threaded=True)
    disagg.exit()


def test_disagg_llm_exit_destroys_decode_before_prefill(monkeypatch):
    """`DisaggLLM.exit()` tears down `decode_engine` before `prefill_engine` (reverse of construction order)."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    exit_order: list[str] = []

    class _NamedEngine(_FakeEngine):
        def __init__(self, role: str, model: str, **kwargs):
            super().__init__(model, **kwargs)
            self._role = role

        def exit(self):
            exit_order.append(self._role)

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=lambda m, **kw: _NamedEngine("prefill", m, **kw),
        decode_engine_cls=lambda m, **kw: _NamedEngine("decode", m, **kw),
        use_cuda_streams=False,
    )
    disagg.exit()
    assert exit_order == ["decode", "prefill"]


def test_disagg_llm_derives_distinct_seeds_for_prefill_and_decode(monkeypatch):
    """When `seed` is provided, `prefill_engine` receives `seed` and `decode_engine` receives `seed + 1`."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    seen_seeds: dict[str, int | None] = {}

    class _SeedRecordingEngine(_FakeEngine):
        def __init__(self, role: str, model: str, **kwargs):
            super().__init__(model, **kwargs)
            seen_seeds[role] = kwargs.get("seed")

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=lambda m, **kw: _SeedRecordingEngine("p", m, **kw),
        decode_engine_cls=lambda m, **kw: _SeedRecordingEngine("d", m, **kw),
        use_cuda_streams=False,
        seed=17011,
    )
    disagg.exit()
    assert seen_seeds == {"p": 17011, "d": 17012}


def test_disagg_llm_single_gpu_requires_explicit_kv_budget():
    """Co-locating P and D on the same GPU without `kv_cache_memory_bytes` fails fast before loading weights."""
    with pytest.raises(ValueError, match="kv_cache_memory_bytes"):
        DisaggLLM("dummy", prefill_device=0, decode_device=0)


def test_disagg_llm_decode_preemption_raises_instead_of_hanging(monkeypatch):
    """If D runs out of KV blocks mid-decode and preempts into `waiting`, `decode_step` raises `RuntimeError` immediately."""
    monkeypatch.setattr(Sequence, "block_size", 4)

    class _TinyDecodeEngine(_FakeEngine):
        def __init__(self, model: str, **kwargs):
            super().__init__(model, **kwargs)
            self.scheduler.block_manager = BlockManager(num_blocks=2, block_size=4, enable_prefix_cache=False)

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_TinyDecodeEngine,
        use_cuda_streams=False,
    )
    with pytest.raises(RuntimeError, match="preempted"):
        disagg.generate([[1, 2, 3], [4, 5, 6]], SamplingParams(temperature=0.0, max_tokens=6), threaded=True)
    disagg.exit()


def test_disagg_llm_respects_max_sampling_keys_and_freezes_graphs_during_threads(monkeypatch):
    """Pre-capture caps at MAX_SAMPLING_KEYS and sets `freeze_sampling_graphs=True` while worker threads are active."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    freeze_during_decode: list[bool] = []

    class _CheckingDecodeEngine(_FakeEngine):
        def step(self):
            freeze_during_decode.append(self.speculative.drafter.freeze_sampling_graphs)
            return super().step()

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_CheckingDecodeEngine,
        use_cuda_streams=False,
    )
    fake_drafter = SimpleNamespace(
        model=SimpleNamespace(markov_head=object()),
        graphs={1: object()},
        sampling_graphs={},
        freeze_sampling_graphs=False,
        _capture_sampling_graphs=lambda key: fake_drafter.sampling_graphs.__setitem__(key, {1: object()}),
    )
    disagg.decode_engine.speculative = SimpleNamespace(drafter=fake_drafter)

    sps = [SamplingParams(temperature=0.8, top_k=k, top_p=0.9, max_tokens=2) for k in (10, 20, 30, 40, 50, 60)]
    prompts = [[1, 2, 3] for _ in sps]
    disagg.generate(prompts, sps, threaded=True)
    disagg.exit()

    assert len(fake_drafter.sampling_graphs) == 4
    assert freeze_during_decode and all(freeze_during_decode)
    assert fake_drafter.freeze_sampling_graphs is False


def test_bench_pd_disagg_q_steady_and_empty_steady_window():
    """`disagg_q_steady` counts committed `d_deliveries` + replacement first tokens and handles missing full batches."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from bench_pd import disagg_cuda_itl, disagg_q_steady

    d_steps = [
        {"step": 0, "kind": "round", "batch_size": 1, "tokens_out": 4, "start_ms": 0.0, "duration_ms": 10.0},
        {"step": 1, "kind": "round", "batch_size": 2, "tokens_out": 8, "start_ms": 10.0, "duration_ms": 20.0},
        {"step": 2, "kind": "round", "batch_size": 2, "tokens_out": 8, "start_ms": 30.0, "duration_ms": 20.0},
    ]
    # Step 1+2 span 40 ms (0.04 s). Req A entered before step 1 (d_deliveries at 0, 1: committed 3 inside window);
    # Req B entered at step 1 (not > first=1: committed 4); Req C replaced A at step 2 (first_d_step=2 > 1 -> +1 prefill tok, committed 2).
    outputs = [
        {"d_deliveries": [(0, 2), (1, 3)]},
        {"d_deliveries": [(1, 2), (2, 2)]},
        {"d_deliveries": [(2, 2)]},
    ]
    q_total, q_decode = disagg_q_steady(d_steps, outputs, concurrency=2)
    # decode_tokens in [1, 2] = 3 + 2 + 2 + 2 = 9; admitted_first_tokens in (1, 2] = 1 (Req C) -> 10 / 0.04s = 250.0
    assert q_decode == 225.0
    assert q_total == 250.0

    # Concurrency 4 never reached -> returns (None, None) and disagg_cuda_itl falls back cleanly without IndexError.
    assert disagg_q_steady(d_steps, outputs, concurrency=4) == (None, None)
    gaps = disagg_cuda_itl(d_steps, outputs, concurrency=4)
    assert gaps["round_p50"] == 20.0


def test_disagg_llm_handoff_thread_ownership(monkeypatch):
    """In two-thread mode, `begin_handoff` runs only on `pd-prefill` and `end_handoff` runs only on `pd-decode`."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    begin_threads: set[str] = set()
    end_threads: set[str] = set()

    disagg = DisaggLLM("dummy", prefill_engine_cls=_FakeEngine, decode_engine_cls=_FakeEngine, use_cuda_streams=False)
    orig_begin = disagg.transfer.begin_handoff
    orig_end = disagg.transfer.end_handoff

    def tracked_begin(*args, **kwargs):
        begin_threads.add(threading.current_thread().name)
        return orig_begin(*args, **kwargs)

    def tracked_end(*args, **kwargs):
        end_threads.add(threading.current_thread().name)
        return orig_end(*args, **kwargs)

    disagg.transfer.begin_handoff = tracked_begin
    disagg.transfer.end_handoff = tracked_end

    prompts = [[1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 20], [3, 4, 5, 6, 7, 8]]
    out = disagg.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4), threaded=True, concurrency=2)
    disagg.exit()

    assert len(out) == 4
    assert begin_threads == {"pd-prefill"}
    assert end_threads == {"pd-decode"}


def test_kv_cache_transfer_staging_ring_wraparound_and_backpressure(monkeypatch):
    """A staging ring smaller than one prefill batch (`max_stage_blocks=3`) wraps cleanly without deadlock or leaks."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    prompts = [
        [1, 2, 3, 4, 5],          # 2 blocks on P
        [6, 7, 8, 9, 10],         # 2 blocks on P (2 + 2 = 4 > max_stage_blocks=3 -> staging backpressure)
        [11, 12, 13, 14, 15],     # 2 blocks on P
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # 3 blocks on P (forces wrap to slot 0)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=3)

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_FakeEngine,
        use_cuda_streams=False,
        max_stage_blocks=3,
    )
    p_total = len(disagg.prefill_engine.scheduler.block_manager.blocks)
    d_total = len(disagg.decode_engine.scheduler.block_manager.blocks)
    serial_out = disagg.generate(prompts, sp, threaded=False)
    threaded_out = disagg.generate(prompts, sp, threaded=True)
    assert len(disagg.prefill_engine.scheduler.block_manager.free_block_ids) == p_total
    assert len(disagg.decode_engine.scheduler.block_manager.free_block_ids) == d_total
    assert disagg.transfer._occupied_stage == []
    disagg.exit()

    assert [o["token_ids"] for o in serial_out] == [o["token_ids"] for o in threaded_out]


def test_smoke_pd_replay_rounds_reproduces_target_admission_steps(monkeypatch):
    """`tools/smoke_pd.py:replay_rounds` holds staged sequences until each target decode round `[0, 1, 1, 3]`."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from smoke_pd import first_decode_rounds, replay_rounds

    monkeypatch.setattr(Sequence, "block_size", 4)
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    disagg = DisaggLLM("dummy", prefill_engine_cls=_FakeEngine, decode_engine_cls=_FakeEngine, use_cuda_streams=False)
    target_rounds = [0, 1, 1, 3]
    replayed = replay_rounds(disagg, prompts, sp, target_rounds)
    disagg.exit()

    assert first_decode_rounds(replayed) == target_rounds


def test_disagg_llm_recovers_staging_and_blocks_after_aborted_generate(monkeypatch):
    """If `d_worker` fails while staged handoffs sit in `_ready_for_decode` / `_pending_handoffs`, a subsequent run has zero leaks."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    should_fail = True

    class _FailOnceDecodeEngine(_FakeEngine):
        def step(self):
            if should_fail:
                raise RuntimeError("transient decode failure")
            return super().step()

    disagg = DisaggLLM(
        "dummy",
        prefill_engine_cls=_FakeEngine,
        decode_engine_cls=_FailOnceDecodeEngine,
        use_cuda_streams=False,
        max_stage_blocks=3,
    )
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    sp = SamplingParams(temperature=0.0, max_tokens=3)
    with pytest.raises(RuntimeError, match="transient decode failure"):
        disagg.generate(prompts, sp, threaded=True)

    # Clear D's scheduler running queue from the aborted step (mirrors a fresh engine state) and verify transfer state recovers.
    while disagg.decode_engine.scheduler.running:
        seq = disagg.decode_engine.scheduler.running.popleft()
        disagg.decode_engine.scheduler.block_manager.deallocate(seq)
    should_fail = False
    out = disagg.generate(prompts, sp, threaded=True)
    p_bm = disagg.prefill_engine.scheduler.block_manager
    d_bm = disagg.decode_engine.scheduler.block_manager
    assert len(p_bm.free_block_ids) == len(p_bm.blocks)
    assert len(d_bm.free_block_ids) == len(d_bm.blocks)
    assert disagg.transfer._occupied_stage == []
    assert len(out) == 3
    disagg.exit()
