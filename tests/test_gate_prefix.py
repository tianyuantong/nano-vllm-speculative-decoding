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


NUM_BLOCKS = 256


def rendered(cache: dict) -> torch.Tensor:
    """A fake model's slot -> record dict as a [1, 1, block, slot, 1, 3] tensor, so the gate can read blocks of it."""
    out = torch.zeros(1, 1, NUM_BLOCKS + 1, BLOCK_SIZE, 1, 3)
    for slot, record in cache.items():
        if slot >= 0:
            values = [{"context": 1.0, "block": 2.0}.get(record[0], 3.0), *map(float, record[1:])] \
                if isinstance(record[0], str) else [3.0, *map(float, record)]
            out[0, 0, slot // BLOCK_SIZE, slot % BLOCK_SIZE, 0, :len(values)] = torch.tensor(values)
    return out


class FakeRunner:
    """ModelRunner's prefill and decode over test_dflash_round's FakeTarget."""

    def __init__(self, target):
        self.target, self.generator = target, target.generator
        self.sampler = lambda logits, sampling, generator=None: logits.argmax(dim=-1)

    override = None    # the shared-block bench copies blocks, so it needs a real tensor

    @property
    def kv_cache(self):
        return self.override if self.override is not None else rendered(self.target.kv)

    @kv_cache.setter
    def kv_cache(self, tensor):
        self.override = tensor

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
                             num_kvcache_blocks=NUM_BLOCKS, enable_prefix_cache=True, num_speculative_tokens=k,
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
def test_the_five_arms_run_and_every_hit_is_as_designed(drafting, monkeypatch):
    from nanovllm.engine.speculative import DFlashDrafter
    monkeypatch.setattr(DFlashDrafter, "kv_cache", property(lambda self: rendered(self.model.cache)), raising=False)
    llm, draft = make_llm(drafting)
    donor = [right(p) for p in range(SHARED)] + [14, 13, 12, 11, 10]
    result = gate_prefix.gate_panel(llm, panel(), donor, SHARED, K, GATE)
    assert result["hits_as_designed"] and result["hashing_off_hits"] == 0, result["hits"]
    assert result["hits"]["hit"] == [SHARED, 0] + [SHARED] * 9 and result["hits"]["hit_batch"] == result["hits"]["hit"]
    assert result["shared_blocks_kept"] == {"hit": True, "hit_batch": True} and result["padding_writes"] == 0
    assert result["passed"] and result["target"]["hit"]["statistic"] == 0
    assert ("draft" in result) == drafting
    if drafting:
        assert draft.context_reads > 0 and not draft.bad_reads, draft.bad_reads[:3]
    assert not llm.scheduler.running and not llm.scheduler.waiting


class FakeTimedEvent:
    clock = 0.0

    def __init__(self):
        FakeTimedEvent.clock += 1.0
        self.at = FakeTimedEvent.clock

    def elapsed_time(self, other):
        return other.at - self.at


@pytest.mark.parametrize("drafting", [True, False])
def test_the_shared_block_bench_points_one_state_at_shared_or_private_blocks(drafting, monkeypatch):
    import shared_kv_bench
    llm, _ = make_llm(drafting)
    num_blocks = llm.scheduler.block_manager.blocks.__len__() + 1
    llm.model_runner.kv_cache = torch.arange(2 * num_blocks * BLOCK_SIZE, dtype=torch.float32).view(2, 1, num_blocks,
                                                                                                    BLOCK_SIZE, 1, 1)
    if drafting:
        drafter = llm.speculative.drafter
        drafter.kv_cache = llm.model_runner.kv_cache.clone() + 0.5
        monkeypatch.setattr(drafter, "_forward_graph", drafter._forward_eager)
    monkeypatch.setattr(shared_kv_bench, "event", FakeTimedEvent)
    monkeypatch.setattr(shared_kv_bench.torch.cuda, "synchronize", lambda: None)
    prompts = [entry["prompt_token_ids"] for entry in panel()[:3]]
    donor = [right(p) for p in range(SHARED)] + [14, 13, 12, 11, 10]
    seqs, shared, private = shared_kv_bench.build_state(llm, prompts, donor, SHARED)
    assert len(shared) == SHARED // BLOCK_SIZE and all(len(set(ids) & set(shared)) == 0 for ids in private)
    for cache in shared_kv_bench.caches(llm):
        assert all(torch.equal(cache[:, :, ids], cache[:, :, shared]) for ids in private)    # the same values
    spec = {"warmup": 1, "steps": 3}
    lengths = [len(seq) for seq in seqs]
    for table in (private, [shared] * len(seqs)):
        medians = shared_kv_bench.measure(llm, seqs, table, len(shared), spec)
        assert set(medians) == ({"draft_forward", "verify"} if drafting else {"decode"})
    assert [len(seq) for seq in seqs] == lengths                                             # nothing committed
    shared_kv_bench.teardown(llm, seqs, shared, private)
    manager = llm.scheduler.block_manager
    assert not llm.scheduler.running and not manager.used_block_ids
