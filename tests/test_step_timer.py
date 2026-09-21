from nanovllm.engine.step_timer import StepTimer


class FakeEvent:
    clock = 0.0

    def record(self):
        self.at = FakeEvent.clock
        FakeEvent.clock += 1.0

    def elapsed_time(self, other):
        return other.at - self.at


def make_timer(enabled=True):
    FakeEvent.clock = 0.0
    return StepTimer(enabled, event_factory=FakeEvent, synchronize=lambda: None)


def test_disabled_timer_records_nothing():
    timer = make_timer(enabled=False)
    timer.end(timer.begin(), "decode", 2, 2)
    assert timer.timings() == []


def test_timings_are_relative_to_the_first_step():
    timer = make_timer()
    start = timer.begin()                      # t=0
    timer.end(start, "prefill", 3, 300)        # t=1
    start = timer.begin()                      # t=2
    timer.end(start, "round", 3, 7)            # t=3
    assert timer.timings() == [
        {"step": 0, "kind": "prefill", "batch_size": 3, "tokens_out": 300, "start_ms": 0.0, "duration_ms": 1.0},
        {"step": 1, "kind": "round", "batch_size": 3, "tokens_out": 7, "start_ms": 2.0, "duration_ms": 1.0},
    ]


def test_reset_clears_records():
    timer = make_timer()
    timer.end(timer.begin(), "decode", 1, 1)
    timer.reset()
    assert timer.timings() == []
