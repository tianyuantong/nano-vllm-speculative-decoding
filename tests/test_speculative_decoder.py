from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nanovllm.engine import speculative
from nanovllm.engine.batch_metadata import verify_varlen_metadata
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import SpeculativeDecoder
from nanovllm.layers.spec_sampler import probs_from_logits
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context

VOCAB = 16
BLOCK_SIZE = 4
K = 3
CPU = torch.device("cpu")


def target_rule(token: int) -> int:
    return (token + 1) % VOCAB


def draft_rule(token: int) -> int:
    return 0 if token == 5 else (token + 1) % VOCAB


def peaked(tokens: torch.Tensor) -> torch.Tensor:
    """Logits that put almost all mass on the given tokens."""
    return torch.full((*tokens.shape, VOCAB), -10.0).scatter_(-1, tokens.unsqueeze(-1), 10.0)


class FakeRunner:
    """Stands in for the target ModelRunner on CPU: records every forward's context, predicts by a token rule."""

    def __init__(self, rule):
        self.rule = rule
        self.block_size = BLOCK_SIZE
        self.device = CPU
        self.enforce_eager = True
        self.generator = torch.Generator().manual_seed(0)
        self.pad_block_id = 99
        self.graph_bs = [1, 2, 4, 8]
        self.graph_pool = None
        self.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=VOCAB, hidden_size=VOCAB, dtype=torch.float32))
        self.model = SimpleNamespace(compute_logits=lambda hidden, all_logits=False: hidden)
        self.calls = []

    def run_model(self, input_ids, positions, is_prefill, *, need_logits=True, aux_out=None):
        context = get_context()
        self.calls.append({
            "is_prefill": is_prefill,
            "input_ids": input_ids.tolist(),
            "positions": positions.tolist(),
            "slot_mapping": context.slot_mapping.tolist(),
            "cu_seqlens_q": None if context.cu_seqlens_q is None else context.cu_seqlens_q.tolist(),
            "cu_seqlens_k": None if context.cu_seqlens_k is None else context.cu_seqlens_k.tolist(),
            "block_tables": None if context.block_tables is None else context.block_tables.tolist(),
        })
        return peaked(torch.tensor([self.rule(token) for token in input_ids.tolist()]))


class FakeDrafter:
    """Stands in for DFlashDrafter: proposes k drafts by draft_rule from the last token, peaked like the target."""
    aux_width, aux_dtype = 2, torch.float32

    def __init__(self, config, decoder):
        self.decoder = decoder

    def propose(self, seqs, sampling, with_confidence=False):
        rows = []
        for seq in seqs:
            token, row = seq.last_token, []
            for _ in range(self.decoder.num_drafts):
                token = draft_rule(token)
                row.append(token)
            rows.append(row)
        drafts = torch.tensor(rows)
        confidence = torch.zeros(len(seqs), self.decoder.num_drafts) if with_confidence else None
        if sampling.temperatures is None:
            return drafts, None, confidence
        return drafts, probs_from_logits(peaked(drafts), sampling), confidence

    def after_verify(self, seqs, aux, verified_lens=None):
        pass


def make_decoder():
    target = FakeRunner(target_rule)
    config = SimpleNamespace(num_speculative_tokens=K, kvcache_block_size=BLOCK_SIZE, max_model_len=64,
                             hf_config=SimpleNamespace(vocab_size=VOCAB), variable_length=False, verify_schedule=None,
                             record_confidences=False)
    with mock.patch.object(speculative, "DFlashDrafter", FakeDrafter):    # the real one loads a checkpoint
        decoder = SpeculativeDecoder(target, config)
    return decoder, target, decoder.drafter


def make_sequence(prompt, first_token, target_blocks, **params):
    """A sequence right after its prefill step: the KV covers the prompt, first token appended."""
    seq = Sequence(prompt, SamplingParams(**params))
    seq.append_token(first_token)
    seq.target_kv.block_table = list(target_blocks)
    seq.target_kv.num_cached_tokens = len(seq) - 1
    return seq


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def test_full_acceptance_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[12, 13, 14, 15]]
    assert decoder.stats["accepted_3"] == 1 and decoder.stats["rounds"] == 1


def test_verification_forward_layout():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    decoder.run_round([seq])
    (verify,) = target.calls
    assert verify["is_prefill"] and verify["input_ids"] == [11, 12, 13, 14]
    assert verify["positions"] == [3, 4, 5, 6]
    assert verify["slot_mapping"] == [15, 16, 17, 18]
    assert verify["cu_seqlens_q"] == [0, 4] and verify["cu_seqlens_k"] == [0, 7]
    assert verify["block_tables"] == [[3, 4]]


def test_partial_acceptance_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]                       # drafts [5, 0, 1]; target says 6 after 5
    assert decoder.stats["accepted_1"] == 1


def test_rows_are_independent_in_a_batch():
    decoder, target, drafter = make_decoder()
    full = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    partial = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], temperature=0)
    appended = decoder.run_round([full, partial])
    assert appended == [[12, 13, 14, 15], [5, 6]]
    (verify,) = target.calls
    assert verify["cu_seqlens_q"] == [0, 4, 8] and verify["cu_seqlens_k"] == [0, 7, 14]


def test_random_mode_with_peaked_distributions_matches_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=1.0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]


@pytest.mark.parametrize("temperature", [0, 1.0], ids=["greedy", "random"])
def test_rows_verify_lengths_of_their_own(temperature):
    # Drafts 12, 13, 14 after 11 and 5, 0, 1 after 4. The first row verifies 12 only: accepted, then the bonus at
    # its second position, 13. The second verifies all three: 5 accepted, 0 rejected for the target's 6.
    decoder, target, drafter = make_decoder()
    first = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=temperature)
    second = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], temperature=temperature)
    appended = decoder.run_round([first, second], verified_lens=[2, 4])
    assert appended == [[12, 13], [5, 6]]
    (verify,) = target.calls
    assert verify["input_ids"] == [11, 12, 4, 5, 0, 1] and verify["positions"] == [3, 4, 3, 4, 5, 6]
    assert verify["slot_mapping"] == [15, 16, 35, 36, 37, 38]
    assert verify["cu_seqlens_q"] == [0, 2, 6] and verify["cu_seqlens_k"] == [0, 5, 12]
    assert (decoder.stats["verified_2"], decoder.stats["verified_4"], decoder.stats["accepted_1"]) == (1, 1, 2)


def test_graph_inputs_hold_the_metadata_and_the_drafts():
    # What _verify_graph writes before a replay, on CPU buffers: a graph of 8 tokens and 4 slots, rows of 2 and 3.
    decoder, target, drafter = make_decoder()
    decoder.graph_vars = {name: torch.zeros(size, dtype=torch.int64 if name in ("input_ids", "positions") else torch.int32)
                          for name, size in [("input_ids", 32), ("positions", 32), ("slot_mapping", 32),
                                             ("cu_seqlens_q", 9), ("cu_seqlens_k", 9), ("block_tables", (8, 2))]}
    seqs = [make_sequence([8, 9, 10], 11, target_blocks=[3, 4]), make_sequence([1, 2, 3], 4, target_blocks=[8, 9])]
    metadata = verify_varlen_metadata(seqs, [2, 3], K, BLOCK_SIZE, 4, 8, 99, 2)
    decoder._load_graph_inputs(metadata, 4, 8)
    decoder._place_drafts(decoder.graph_vars["input_ids"], torch.tensor([[12, 13, 14], [5, 0, 1]]), metadata)
    buffers = decoder.graph_vars
    assert buffers["input_ids"][:8].tolist() == [11, 12, 4, 5, 0, 0, 0, 0]
    assert buffers["cu_seqlens_q"][:5].tolist() == [0, 2, 5, 8, 8]
    assert buffers["block_tables"][:4].tolist() == [[3, 4], [8, 9], [99, 99], [99, 99]]


def test_a_variable_length_round_takes_its_lengths_from_the_schedule():
    # The same two rows as above, the lengths now picked by the schedule from the round's confidences, [B, k].
    decoder, target, drafter = make_decoder()
    shapes = []

    def pick_lengths(logits):
        shapes.append(logits.shape)
        return [2, 4]

    decoder.variable_length = True
    decoder.schedule = SimpleNamespace(verify_lengths=pick_lengths)
    decoder.enable_phase_timings(event_factory=FakeEvent, synchronize=lambda: None)
    first = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    second = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], temperature=0)
    assert decoder.run_round([first, second]) == [[12, 13], [5, 6]] and shapes == [(2, K)]
    assert [phase["phase"] for phase in decoder.phase_timings()][:2] == ["propose", "schedule"]


def test_a_recording_round_keeps_its_confidences_and_accepted_counts():
    # Fixed length: the first row accepts all three drafts, the second one of 5, 0, 1.
    decoder, target, drafter = make_decoder()
    decoder.record_confidences = True
    first = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    second = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], temperature=0)
    decoder.run_round([first, second])
    assert decoder.confidence_log == [{"logits": [[0.0] * K, [0.0] * K], "accepted": [3, 1]}]


def test_round_entry_invariants_are_asserted():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=0)
    seq.target_kv.num_cached_tokens = len(seq)
    with pytest.raises(AssertionError):
        decoder.run_round([seq])


def test_phase_timings_are_recorded_when_enabled():
    decoder, target, drafter = make_decoder()
    decoder.enable_phase_timings(event_factory=FakeEvent, synchronize=lambda: None)
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    decoder.run_round([seq])
    phases = decoder.phase_timings()
    assert [phase["phase"] for phase in phases] == ["propose", "verify", "ctx_store", "accept", "commit_copy"]
    assert all(phase["duration_ms"] >= 0 for phase in phases)


class FakeEvent:
    clock = 0.0

    def record(self):
        self.at = FakeEvent.clock
        FakeEvent.clock += 1.0

    def elapsed_time(self, other):
        return other.at - self.at
