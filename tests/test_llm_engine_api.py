from collections import Counter
from types import SimpleNamespace

import pytest

import gpu_import_stubs  # noqa: F401  (before the engine import)
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.step_timer import StepTimer
from nanovllm.sampling_params import SamplingParams


class FakeEvent:
    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.0


def make_engine(**attrs):
    """An LLMEngine without models: only the attributes add_request and start_measurement touch."""
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(max_model_len=4096, num_speculative_tokens=0)
    engine.scheduler = SimpleNamespace(waiting=[])
    engine.scheduler.add = engine.scheduler.waiting.append
    engine.step_timer = StepTimer(True, event_factory=FakeEvent, synchronize=lambda: None)
    engine.speculative = None
    engine.num_steps = 0
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine


def test_add_request_returns_the_queued_sequence():
    engine = make_engine()
    seq = engine.add_request([1, 2, 3], SamplingParams(max_tokens=4))
    assert isinstance(seq, Sequence)
    assert engine.scheduler.waiting[-1] is seq
    assert seq.token_ids == [1, 2, 3]


def test_add_request_still_checks_the_length_budget():
    engine = make_engine()
    with pytest.raises(ValueError):
        engine.add_request(list(range(4090)), SamplingParams(max_tokens=8))


def test_start_measurement_clears_timings_stats_and_steps():
    resets = []
    speculative = SimpleNamespace(stats=Counter(rounds=3), confidence_log=[{"logits": [[0.0]], "accepted": [0]}],
                                  reset_phase_timings=lambda: resets.append(1))
    engine = make_engine(speculative=speculative, num_steps=7)
    engine.scheduler.num_preemptions = 3
    engine.scheduler.initial_requests_left = 8
    for _ in range(2):
        engine.step_timer.end(engine.step_timer.begin(), "round", 1, 2, decode_rows=1, prefill_tokens=0, delivered_tokens=2)
    engine.start_measurement()
    assert engine.step_timer.timings() == []
    assert engine.num_steps == 0 and not speculative.stats and not speculative.confidence_log and len(resets) == 1
    assert engine.scheduler.num_preemptions == 0 and engine.scheduler.initial_requests_left == 0
    engine.start_measurement(initial_group=8)
    assert engine.scheduler.initial_requests_left == 8


def test_start_measurement_without_a_draft():
    engine = make_engine(num_steps=3)
    engine.start_measurement()
    assert engine.num_steps == 0


def test_step_records_what_it_delivered_not_what_the_round_returned():
    config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=64, eos=7, kvcache_block_size=256,
                             num_kvcache_blocks=16, enable_prefix_cache=False, num_speculative_tokens=3,
                             round_robin_prefill=False)
    speculative = SimpleNamespace(prefill_aux=lambda seqs: None, prefill_draft=lambda seqs, aux: None,
                                  run_round=lambda seqs: [[7, 5, 6, 8]])    # EOS first: the row delivers 1 of its 4
    engine = make_engine(scheduler=Scheduler(config), speculative=speculative,
                         model_runner=SimpleNamespace(call=lambda *args: [9]))
    engine.scheduler.add(Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=8)))
    engine.step()
    engine.step()
    prefill, round_step = engine.step_timer.timings()
    assert (prefill["decode_rows"], prefill["prefill_tokens"], prefill["delivered_tokens"]) == (0, 5, 1)
    assert (round_step["decode_rows"], round_step["prefill_tokens"], round_step["delivered_tokens"]) == (1, 0, 1)
    assert round_step["tokens_out"] == 4
