"""tools/gate_prefix.py's five arms end to end on CPU, around test_dflash_round's fake models and a real Scheduler.

The fake target's logits do not read its K/V, so every delta is zero here: this checks the flow, the forcing and the
hit accounting, not the numbers, which only a GPU run measures. The fake draft does check every context read."""
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpu_import_stubs  # noqa: F401  (before the engine import)
from test_dflash_round import BLOCK_SIZE, K, make_decoder, right
from nanovllm.engine.batch_metadata import decode_metadata, padded_block_tables, prefill_metadata
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.step_timer import StepTimer
from nanovllm.utils.context import reset_context, set_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gate_prefix  # noqa: E402

GATE = {"per_category": 2, "floor_batch": 3, "rounds": 4, "floor_multiple": 2, "margin": 0.5}
SHARED = 8


class FakeEvent:
    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.0


class FakeRunner:
    """ModelRunner's prefill and decode over test_dflash_round's FakeTarget."""

    def __init__(self, target):
        self.target, self.generator = target, target.generator
        self.sampler = lambda logits, sampling, generator=None: logits.argmax(dim=-1)

    def call(self, name, seqs, is_prefill, aux=None):
        assert name == "run" and is_prefill
        metadata = prefill_metadata(seqs, BLOCK_SIZE)
        set_context(True, slot_mapping=torch.tensor(metadata.slot_mapping))
        logits = self.target.run_model(torch.tensor(metadata.input_ids), torch.tensor(metadata.positions), True,
                                       aux_out=aux)
        reset_context()
        last = torch.tensor(metadata.cu_seqlens_q[1:]) - 1
        return self.sampler(logits[last], None, self.generator).tolist()

    def prepare_decode(self, seqs):
        metadata = decode_metadata(seqs, BLOCK_SIZE)
        set_context(False, slot_mapping=torch.tensor(metadata.slot_mapping),
                    context_lens=torch.tensor(metadata.context_lens),
                    block_tables=torch.tensor(padded_block_tables(seqs)))
        return torch.tensor(metadata.input_ids), torch.tensor(metadata.positions)

    def run_model(self, input_ids, positions, is_prefill):
        return self.target.run_model(input_ids, positions, is_prefill)


def make_llm(drafting: bool):
    decoder, target, draft = make_decoder(lambda position: position % 3 != 0)
    forward = decoder.drafter._forward_eager

    def forward_eager(seqs):          # the fake draft reads the round's sequences from the decoder
        decoder.current_seqs = seqs
        return forward(seqs)
    decoder.drafter._forward_eager = forward_eager
    k = K if drafting else 0
    config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=4096, eos=-1, kvcache_block_size=BLOCK_SIZE,
                             num_kvcache_blocks=256, enable_prefix_cache=True, num_speculative_tokens=k,
                             round_robin_prefill=False, max_model_len=4096, variable_length=False)
    llm = object.__new__(LLMEngine)
    llm.config, llm.scheduler = config, Scheduler(config)
    llm.model_runner = gate_prefix.CapturingSampler.__new__(gate_prefix.CapturingSampler)    # replaced below
    llm.model_runner = FakeRunner(target)
    llm.model_runner.sampler = gate_prefix.CapturingSampler(llm.model_runner.sampler)
    llm.speculative = decoder if drafting else None
    if not drafting:
        target.run_model = lambda input_ids, positions, is_prefill, **kw: torch.nn.functional.one_hot(
            torch.tensor([right(p + 1) for p in positions.tolist()]), 16).float() * 10
    llm.step_timer = StepTimer(True, event_factory=FakeEvent, synchronize=lambda: None)
    llm.num_steps = 0
    decoder.stats = Counter()
    return llm, draft


def panel():
    """Two of each of four categories plus two more, each its own private part after the shared tokens."""
    shared = [right(p) for p in range(SHARED)]
    entries = []
    for index in range(10):
        private = [(7 * index + 2 * j + 5) % 15 for j in range(9 + index % 4)]    # never the shared rule's run
        entries.append({"id": f"q{index}", "category": f"c{index % 4}", "prompt_token_ids": shared + private})
    return entries


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


@pytest.mark.parametrize("drafting", [True, False])
def test_the_five_arms_run_and_every_hit_is_as_designed(drafting):
    llm, draft = make_llm(drafting)
    donor = [right(p) for p in range(SHARED)] + [14, 13, 12, 11, 10]
    result = gate_prefix.gate_panel(llm, panel(), donor, SHARED, K, GATE)
    assert result["hits_as_designed"] and result["hashing_off_hits"] == 0, result["hits"]
    assert result["hits"]["hit"] == [SHARED] * 10 + [0] and result["hits"]["hit_batch"] == result["hits"]["hit"]
    assert result["passed"] and result["target"]["hit"]["statistic"] == 0
    assert ("draft" in result) == drafting
    if drafting:
        assert draft.context_reads > 0 and not draft.bad_reads, draft.bad_reads[:3]
    assert not llm.scheduler.running and not llm.scheduler.waiting
