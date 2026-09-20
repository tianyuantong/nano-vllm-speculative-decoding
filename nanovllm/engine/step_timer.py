"""Per-step timing with CUDA events, read once after a run so no step pays a synchronization."""
from typing import Callable

import torch


class StepTimer:

    def __init__(self, enabled: bool, event_factory: Callable | None = None, synchronize: Callable | None = None):
        self.enabled = enabled
        self._new_event = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self._synchronize = synchronize or torch.cuda.synchronize
        self._records: list[tuple] = []

    def reset(self) -> None:
        self._records.clear()

    def begin(self):
        if not self.enabled:
            return None
        event = self._new_event()
        event.record()
        return event

    def end(self, start_event, kind: str, batch_size: int, tokens_out: int) -> None:
        if not self.enabled:
            return
        end_event = self._new_event()
        end_event.record()
        self._records.append((kind, batch_size, tokens_out, start_event, end_event))

    def timings(self) -> list[dict]:
        """One dict per step; `start_ms` is relative to the first recorded step."""
        if not self._records:
            return []
        self._synchronize()
        origin = self._records[0][3]
        return [
            {"step": index, "kind": kind, "batch_size": batch_size, "tokens_out": tokens_out,
             "start_ms": origin.elapsed_time(start), "duration_ms": start.elapsed_time(end)}
            for index, (kind, batch_size, tokens_out, start, end) in enumerate(self._records)
        ]
