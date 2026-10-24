from collections import Counter
from types import SimpleNamespace

import pytest

import gpu_import_stubs  # noqa: F401  (before the engine import)
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
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
    speculative = SimpleNamespace(stats=Counter(rounds=3), reset_phase_timings=lambda: resets.append(1))
    engine = make_engine(speculative=speculative, num_steps=7)
    for _ in range(2):
        engine.step_timer.end(engine.step_timer.begin(), "round", 1, 2)
    engine.start_measurement()
    assert engine.step_timer.timings() == []
    assert engine.num_steps == 0 and not speculative.stats and len(resets) == 1


def test_start_measurement_without_a_draft():
    engine = make_engine(num_steps=3)
    engine.start_measurement()
    assert engine.num_steps == 0
