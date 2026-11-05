"""One speculative round per decode batch on top of the target ModelRunner and a block draft (DFlash, DSpark)."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

from nanovllm.engine.batch_metadata import (
    VerifyMetadata,
    draft_block_metadata,
    prefill_metadata,
    verify_metadata,
    verify_varlen_metadata,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.verify_schedule import VerifySchedule, verify_token_counts
from nanovllm.layers.spec_sampler import (
    SamplingTensors,
    accept_greedy,
    accept_greedy_varlen,
    accept_random,
    accept_random_varlen,
    exponential_noise_,
    probs_from_logits,
    race,
    sample,
    sampling_per_token,
    sampling_tensors,
)
from nanovllm.utils.context import reset_context, set_context

if TYPE_CHECKING:
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner


# When every row truncates with top-k, a DSpark row is drawn among its top max_top_k + TIE_MARGIN logits: the
# truncation also keeps the tokens tied with its cut value, and the margin leaves room for them.
TIE_MARGIN = 64
# DSpark captures sampling graphs, and a noise buffer, for at most this many sampling_graph_key values, the first
# ones it meets; a batch with any other key samples eagerly, so varying top-k cannot grow them without bound.
MAX_SAMPLING_KEYS = 4


def num_candidates(max_top_k: int, all_top_k: bool, vocab_size: int) -> int:
    """How many of its top logits a DSpark row is drawn among; 0 means the whole vocabulary (some row does not
    truncate). Exact unless more than max_top_k + TIE_MARGIN tokens reach a row's cut value, which takes more than
    TIE_MARGIN of them tied with it."""
    return min(max_top_k + TIE_MARGIN, vocab_size) if all_top_k else 0


def sampling_graph_key(sampling: SamplingTensors) -> tuple[int, bool, bool] | None:
    """What a DSpark sampling graph depends on besides the batch size: None when greedy, else the host-side
    truncation settings (the top-k width, whether top-p runs, whether every row truncates), which fix its shapes
    and branches."""
    return None if sampling.temperatures is None else (sampling.max_top_k, sampling.has_top_p, sampling.all_top_k)


class DFlashDrafter:
    """DeepSpec's block draft: one forward over a block [anchor, mask x (k-1)] proposes k drafts. DFlash samples
    the k rows at once; DSpark, whose model has a Markov head, samples them left to right.

    Its KV holds context K/V computed from the target's aux features, written for every position the target
    computes: each prefilled chunk, and every verified position of each verification, all k+1 unless the row
    verifies fewer. A rejected or unverified position is overwritten by the next block before any block reads it,
    so at round entry the context covers len-1 positions, like the target's KV. It shares the target's block table: same block count, same pad block.
    With graphs, the block forward replays one graph per padded batch size. DFlash's logits and sampling stay
    outside; DSpark's logits and k rows replay a sampling graph per padded batch size and sampling_graph_key,
    captured for every batch size when that key first appears, for the first MAX_SAMPLING_KEYS keys only.
    DSpark's confidence logits, when asked for, are computed eagerly after the drafts, outside the graphs.
    """

    def __init__(self, config: Config, decoder: SpeculativeDecoder):
        from transformers import AutoConfig
        from nanovllm.models.dflash import DFlashDraftModel
        from nanovllm.utils.loader import load_model

        self.decoder = decoder
        hf_config = AutoConfig.from_pretrained(config.draft_model)
        default_dtype, default_device = torch.get_default_dtype(), torch.get_default_device()
        try:
            torch.set_default_dtype(config.hf_config.dtype)
            torch.set_default_device(decoder.device)
            self.model = DFlashDraftModel(hf_config)
            load_model(self.model, config.draft_model)
            head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
            self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks + 1,
                                        config.kvcache_block_size, hf_config.num_key_value_heads, head_dim)
        finally:
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)
        for layer_id, layer in enumerate(self.model.layers):
            layer.self_attn.attn.k_cache = self.kv_cache[0, layer_id]
            layer.self_attn.attn.v_cache = self.kv_cache[1, layer_id]
        decoder.target.model.model.aux_layer_ids = self.model.aux_layer_ids
        self.aux_width = len(self.model.aux_layer_ids) * hf_config.hidden_size
        self.aux_dtype = config.hf_config.dtype
        self.hidden_size = hf_config.hidden_size
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.sampling_graphs: dict[tuple[int, bool, bool] | None, dict[int, torch.cuda.CUDAGraph]] = {}
        self.sampling_noise: dict[tuple[int, bool, bool], torch.Tensor] = {}
        self.graph_vars: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence], aux: torch.Tensor) -> None:
        """Context K/V of the chunk the target just prefilled, from the aux features its forward exported."""
        decoder = self.decoder
        metadata = prefill_metadata(seqs, decoder.block_size)
        self.model.store_context_kv(self.model.context_features(aux),
                                    decoder._device_tensor(metadata.positions, torch.int64),
                                    decoder._device_tensor(metadata.slot_mapping, torch.int32))

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], sampling: SamplingTensors,
                with_confidence: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One block forward; returns drafts [B, k], probabilities [B, k, V] (None when greedy) and, with_confidence,
        DSpark's confidence logits [B, k] (None otherwise)."""
        hidden = self._forward_graph(seqs) if self.graphs else self._forward_eager(seqs)
        drafts, draft_probs = self._draw(seqs, hidden, sampling)
        confidence = self._confidence_logits(seqs, hidden, drafts) if with_confidence else None
        return drafts, draft_probs, confidence

    def _draw(self, seqs: list[Sequence], hidden: torch.Tensor,
              sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The drafts and their q from the block outputs: DSpark's rows left to right, DFlash's at once."""
        decoder, k = self.decoder, self.decoder.num_drafts
        if self.model.markov_head is not None and self.graphs and self._has_sampling_graphs(sampling):
            return self._sample_markov_graph(len(seqs), sampling)    # reads the block graph's outputs
        logits = self.model.compute_logits(hidden).view(len(seqs), k, -1)
        if self.model.markov_head is not None:
            return self._sample_markov(seqs, logits, sampling)
        if sampling.temperatures is None:
            return logits.argmax(dim=-1), None
        draft_probs = probs_from_logits(logits, sampling)
        return sample(draft_probs, decoder.target.generator), draft_probs

    def _confidence_logits(self, seqs: list[Sequence], hidden: torch.Tensor, drafts: torch.Tensor) -> torch.Tensor:
        """DSpark's confidence logits [B, k] from the block outputs and, per row, the token before its draft: the
        anchor for row 0, draft j-1 for row j. Read after the drafts are drawn, but row j never reads draft j."""
        anchors = self.decoder._device_tensor([seq.last_token for seq in seqs], torch.int64)
        prev = torch.cat([anchors.unsqueeze(1), drafts[:, :-1]], dim=1)
        return self.model.confidence_logits(hidden.view(len(seqs), self.decoder.num_drafts, -1), prev)

    def _sample_markov(self, seqs: list[Sequence], logits: torch.Tensor,
                       sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """DSpark: row j's logits get the Markov bias of the token drawn at row j-1 (the anchor for row 0), so the
        rows are drawn left to right, and each row's q is its distribution after its bias. The noise of all k rows
        is drawn before row 0 in one call, and the sampling graph consumes the generator alike."""
        decoder = self.decoder
        batch_size, k, vocab_size = logits.shape
        drafts = torch.empty(batch_size, k, dtype=torch.int64, device=decoder.device)
        candidates = num_candidates(sampling.max_top_k, sampling.all_top_k, vocab_size)
        draft_probs = noise = None
        if sampling.temperatures is not None:
            draft_probs = torch.empty(logits.shape, dtype=torch.float32, device=decoder.device)
            noise = exponential_noise_(torch.empty(batch_size, k, candidates or vocab_size, dtype=torch.float32,
                                                   device=decoder.device), decoder.target.generator)
        prev = decoder._device_tensor([seq.last_token for seq in seqs], torch.int64)
        self._markov_rows(logits, prev, sampling, noise, drafts, draft_probs, candidates)
        return drafts, draft_probs

    def _markov_rows(self, logits: torch.Tensor, prev: torch.Tensor, sampling: SamplingTensors | None,
                     noise: torch.Tensor | None, drafts: torch.Tensor, draft_probs: torch.Tensor | None,
                     candidates: int) -> None:
        """Draw the k rows left to right into drafts, and their q into draft_probs unless greedy; prev holds the
        anchors, the previous tokens of row 0. With candidates, a row is drawn among its top `candidates` logits,
        which hold every token the truncation keeps, and its q is scattered back over the vocabulary; otherwise
        over the whole vocabulary."""
        if draft_probs is not None and candidates:
            draft_probs.zero_()                    # q is zero outside each row's candidates
        for row in range(logits.shape[1]):
            row_logits = logits[:, row] + self.model.markov_head(prev)
            if draft_probs is None:
                prev = row_logits.argmax(dim=-1)
            elif candidates:
                values, ids = row_logits.topk(candidates, dim=-1)
                probs = probs_from_logits(values, sampling)
                prev = ids.gather(-1, race(probs, noise[:, row]).unsqueeze(-1)).squeeze(-1)
                draft_probs[:, row].scatter_(-1, ids, probs)
            else:
                draft_probs[:, row] = probs_from_logits(row_logits, sampling)
                prev = race(draft_probs[:, row], noise[:, row])
            drafts[:, row] = prev

    def _has_sampling_graphs(self, sampling: SamplingTensors) -> bool:
        """Whether this batch's sampling key has its sampling graphs, or may still capture them."""
        return sampling_graph_key(sampling) in self.sampling_graphs or len(self.sampling_graphs) < MAX_SAMPLING_KEYS

    def _sample_markov_graph(self, batch_size: int,
                             sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """_sample_markov through the sampling graph of this batch's key, from the block graph's outputs; the
        drafts and q are views of the graph's buffers, valid until the next propose. The noise is drawn outside the
        graph, so capturing draws no random numbers and each call consumes the generator as _sample_markov does."""
        key = sampling_graph_key(sampling)
        if key not in self.sampling_graphs:
            self._capture_sampling_graphs(key)
        self._load_sampling_inputs(batch_size, sampling, key)
        padded = next(size for size in self.decoder.graph_batch_sizes if size >= batch_size)
        self.sampling_graphs[key][padded].replay()
        buffers = self.graph_vars
        return buffers["drafts"][:batch_size], None if key is None else buffers["probs"][:batch_size]

    def _load_sampling_inputs(self, batch_size: int, sampling: SamplingTensors,
                              key: tuple[int, bool, bool] | None) -> None:
        """The batch's temperatures, top-k, top-p and noise into the sampling graphs' buffers; nothing when greedy.
        Padding rows keep earlier values, which stay finite."""
        if key is None:
            return
        buffers = self.graph_vars
        buffers["temperatures"][:batch_size] = sampling.temperatures
        buffers["top_k"][:batch_size] = sampling.top_k
        buffers["top_p"][:batch_size] = sampling.top_p
        exponential_noise_(self.sampling_noise[key][:batch_size], self.decoder.target.generator)

    def _sampling_forward(self, batch_size: int, key: tuple[int, bool, bool] | None) -> None:
        """A sampling graph's body: logits of the block graph's outputs, then the k rows into the buffers. Row 0's
        previous token is each block's first input, the anchor."""
        k, buffers = self.decoder.num_drafts, self.graph_vars
        logits = self.model.compute_logits(buffers["outputs"][:batch_size * k]).view(batch_size, k, -1)
        prev = buffers["input_ids"].view(-1, k)[:batch_size, 0]
        drafts = buffers["drafts"][:batch_size]
        if key is None:
            self._markov_rows(logits, prev, None, None, drafts, None, 0)
            return
        max_top_k, has_top_p, all_top_k = key
        sampling = SamplingTensors(temperatures=buffers["temperatures"][:batch_size],
                                   top_k=buffers["top_k"][:batch_size], top_p=buffers["top_p"][:batch_size],
                                   max_top_k=max_top_k, has_top_p=has_top_p, all_top_k=all_top_k)
        self._markov_rows(logits, prev, sampling, self.sampling_noise[key][:batch_size], drafts,
                          buffers["probs"][:batch_size], num_candidates(max_top_k, all_top_k, logits.shape[-1]))

    def _sampling_buffers(self, max_batch_size: int) -> dict[str, torch.Tensor]:
        """What DSpark's sampling graphs read and write besides the block graph's buffers and the noise. Parameters
        start finite, so padding rows stay finite."""
        decoder, shape = self.decoder, (max_batch_size, self.decoder.num_drafts)
        vocab_size, device = decoder.vocab_size, decoder.device
        return {"drafts": torch.zeros(shape, dtype=torch.int64, device=device),
                "probs": torch.zeros(*shape, vocab_size, dtype=torch.float32, device=device),
                "temperatures": torch.ones(max_batch_size, dtype=torch.float32, device=device),
                "top_k": torch.zeros(max_batch_size, dtype=torch.int64, device=device),
                "top_p": torch.ones(max_batch_size, dtype=torch.float32, device=device)}

    def _allocate_sampling_noise(self, key: tuple[int, bool, bool], max_batch_size: int) -> None:
        """A random key's noise buffer, [max batch, k, candidates or V], with finite padding rows."""
        max_top_k, _, all_top_k = key
        vocab_size = self.decoder.vocab_size
        width = num_candidates(max_top_k, all_top_k, vocab_size) or vocab_size
        self.sampling_noise[key] = torch.ones(max_batch_size, self.decoder.num_drafts, width, dtype=torch.float32,
                                              device=self.decoder.device)

    @torch.inference_mode()
    def _capture_sampling_graphs(self, key: tuple[int, bool, bool] | None) -> None:
        """One sampling graph per batch size of the block graphs, into the target's pool."""
        decoder, graphs = self.decoder, {}
        if key is not None:
            self._allocate_sampling_noise(key, decoder.graph_batch_sizes[-1])
        with torch.device(decoder.device):
            for batch_size in reversed(decoder.graph_batch_sizes):
                graph = torch.cuda.CUDAGraph()
                self._sampling_forward(batch_size, key)    # warm-up
                with torch.cuda.graph(graph, decoder.target.graph_pool):
                    self._sampling_forward(batch_size, key)
                graphs[batch_size] = graph
                torch.cuda.synchronize()
        self.sampling_graphs[key] = graphs

    def _forward_eager(self, seqs: list[Sequence]) -> torch.Tensor:
        decoder, k = self.decoder, self.decoder.num_drafts
        metadata = draft_block_metadata(seqs, decoder.block_size, k, self.model.mask_token_id, len(seqs),
                                        decoder.target.pad_block_id, decoder.max_num_blocks)    # the graphs' width
        set_context(False, slot_mapping=decoder._device_tensor(metadata.slot_mapping, torch.int32),
                    context_lens=decoder._device_tensor(metadata.context_lens, torch.int32),
                    block_tables=decoder._device_tensor(metadata.block_tables, torch.int32),
                    query_len_per_seq=k, causal=False)
        hidden = self.model(decoder._device_tensor(metadata.input_ids, torch.int64),
                            decoder._device_tensor(metadata.positions, torch.int64))
        reset_context()
        return hidden

    def _forward_graph(self, seqs: list[Sequence]) -> torch.Tensor:
        decoder, k = self.decoder, self.decoder.num_drafts
        padded = next(size for size in decoder.graph_batch_sizes if size >= len(seqs))
        metadata = draft_block_metadata(seqs, decoder.block_size, k, self.model.mask_token_id, padded,
                                        decoder.target.pad_block_id, decoder.max_num_blocks)
        rows, buffers = padded * k, self.graph_vars
        buffers["input_ids"][:rows].copy_(decoder._host_tensor(metadata.input_ids, torch.int64), non_blocking=True)
        buffers["positions"][:rows].copy_(decoder._host_tensor(metadata.positions, torch.int64), non_blocking=True)
        buffers["slot_mapping"][:rows].copy_(decoder._host_tensor(metadata.slot_mapping, torch.int32), non_blocking=True)
        buffers["context_lens"][:padded].copy_(decoder._host_tensor(metadata.context_lens, torch.int32), non_blocking=True)
        buffers["block_tables"][:padded].copy_(decoder._host_tensor(metadata.block_tables, torch.int32), non_blocking=True)
        self.graphs[padded].replay()
        return buffers["outputs"][:len(seqs) * k]

    @torch.inference_mode()
    def capture_graphs(self) -> None:
        """One block-forward graph per batch size of the decoder's verify graphs, into the target's pool."""
        decoder, k = self.decoder, self.decoder.num_drafts
        max_batch_size = decoder.graph_batch_sizes[-1]
        all_padding = draft_block_metadata([], decoder.block_size, k, self.model.mask_token_id, max_batch_size,
                                           decoder.target.pad_block_id, decoder.max_num_blocks)
        self.graph_vars = buffers = {
            "input_ids": decoder._device_tensor(all_padding.input_ids, torch.int64),
            "positions": decoder._device_tensor(all_padding.positions, torch.int64),
            "slot_mapping": decoder._device_tensor(all_padding.slot_mapping, torch.int32),
            "context_lens": decoder._device_tensor(all_padding.context_lens, torch.int32),
            "block_tables": decoder._device_tensor(all_padding.block_tables, torch.int32),
            "outputs": torch.zeros(max_batch_size * k, self.hidden_size, dtype=self.aux_dtype, device=decoder.device),
        }
        if self.model.markov_head is not None:
            buffers.update(self._sampling_buffers(max_batch_size))
        with torch.device(decoder.device):
            for batch_size in reversed(decoder.graph_batch_sizes):
                rows = batch_size * k
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=buffers["slot_mapping"][:rows],
                            context_lens=buffers["context_lens"][:batch_size],
                            block_tables=buffers["block_tables"][:batch_size], query_len_per_seq=k, causal=False)
                buffers["outputs"][:rows] = self.model(buffers["input_ids"][:rows], buffers["positions"][:rows])    # warm-up
                with torch.cuda.graph(graph, decoder.target.graph_pool):
                    buffers["outputs"][:rows] = self.model(buffers["input_ids"][:rows], buffers["positions"][:rows])
                self.graphs[batch_size] = graph
                torch.cuda.synchronize()
                reset_context()

    @torch.inference_mode()
    def after_verify(self, seqs: list[Sequence], aux: torch.Tensor, verified_lens: list[int] | None = None) -> None:
        """Context K/V of every verified position, all k+1 per row unless verified_lens, from the aux features the
        verification exported."""
        decoder = self.decoder
        lens = [decoder.num_drafts + 1] * len(seqs) if verified_lens is None else verified_lens
        width = max(len(seq.target_kv.block_table) for seq in seqs)
        metadata = verify_varlen_metadata(seqs, lens, decoder.num_drafts, decoder.block_size, len(seqs), sum(lens),
                                          decoder.target.pad_block_id, width)
        self.model.store_context_kv(self.model.context_features(aux),
                                    decoder._device_tensor(metadata.positions, torch.int64),
                                    decoder._device_tensor(metadata.slot_mapping, torch.int32))


class SpeculativeDecoder:
    """Drafts, one verification forward, batched acceptance, one D2H copy.

    Round-entry invariant (asserted): target KV covers len-1 positions, and so does the drafter's context K/V
    that mirrors it. Every active sequence proposes exactly k drafts. In fixed-length verification each row
    verifies all of them, so the verification graphs have B row slots of k+1 tokens for each padded batch size B.
    With variable_length, row r may verify only its first verified_lens[r] - 1, and the graphs have one token count
    each (verify_token_counts), with a row slot per token up to the largest batch size; the slots a round does not
    use are empty rows. The lengths come from the round's caller or from the verify schedule, which reads DSpark's
    confidences.
    """

    def __init__(self, target: ModelRunner, config: Config):
        self.target = target
        self.num_drafts = config.num_speculative_tokens
        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.vocab_size = config.hf_config.vocab_size
        self.device = target.device
        self.variable_length = config.variable_length
        self.schedule = None if config.verify_schedule is None else VerifySchedule.load(
            config.verify_schedule, self.num_drafts, config.max_num_seqs)
        self.record_confidences = config.record_confidences
        self.confidence_log: list[dict] = []    # with record_confidences, per round: each row's logits and accepted count
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}            # verification graphs by token count
        self.graph_shapes: list[tuple[int, int]] = []                # their (row slots, tokens), by tokens
        self.graph_batch_sizes: list[int] = []
        self.graph_vars: dict[str, torch.Tensor] = {}
        self.stats: Counter = Counter()
        self._phase_events: list[tuple[str, object, object]] = []
        self._new_event = None
        self._synchronize = None
        self.drafter = DFlashDrafter(config, self)

    # ------------------------------------------------------------------ prefill
    def prefill_aux(self, seqs: list[Sequence]) -> torch.Tensor:
        """The buffer the target's prefill exports the drafter's aux features into."""
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        return torch.empty(num_tokens, self.drafter.aux_width, dtype=self.drafter.aux_dtype, device=self.device)

    @torch.inference_mode()
    def prefill_draft(self, seqs: list[Sequence], aux: torch.Tensor) -> None:
        """Bring the draft's KV up to the chunk the target just prefilled."""
        self.drafter.prefill(seqs, aux)

    # ------------------------------------------------------------------ phase timings
    def enable_phase_timings(self, event_factory=None, synchronize=None) -> None:
        """Record one CUDA event after each phase of every round; read with phase_timings()."""
        self._new_event = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self._synchronize = synchronize or torch.cuda.synchronize

    def _mark(self, phase: str, previous) -> object:
        if self._new_event is None:
            return None
        event = self._new_event()
        event.record()
        if previous is not None:
            self._phase_events.append((phase, previous, event))
        return event

    def phase_timings(self) -> list[dict]:
        if not self._phase_events:
            return []
        self._synchronize()
        return [{"phase": phase, "duration_ms": start.elapsed_time(end)} for phase, start, end in self._phase_events]

    def reset_phase_timings(self) -> None:
        self._phase_events.clear()

    # ------------------------------------------------------------------ the round
    @torch.inference_mode()
    def run_round(self, seqs: list[Sequence], verified_lens: list[int] | None = None) -> list[list[int]]:
        """Return, per sequence, the accepted drafts followed by the tail token. With verified_lens, row r verifies
        only its last token and its first verified_lens[r] - 1 drafts; with variable_length and no verified_lens,
        the verify schedule picks them from DSpark's confidences."""
        for seq in seqs:
            assert seq.target_kv.num_cached_tokens == len(seq) - 1
        mark = self._mark("start", None)
        sampling = sampling_tensors(seqs, self.device)
        scheduled = self.variable_length and verified_lens is None
        drafts, draft_probs, confidence = self.drafter.propose(seqs, sampling,
                                                               with_confidence=scheduled or self.record_confidences)
        mark = self._mark("propose", mark)
        if scheduled:
            assert self.schedule is not None, "a variable-length round needs a verify schedule or its lengths"
            verified_lens = self.schedule.verify_lengths(confidence.cpu().numpy())    # the round's one extra sync
            mark = self._mark("schedule", mark)
        num_tokens = len(seqs) * (self.num_drafts + 1) if verified_lens is None else sum(verified_lens)
        aux = self._aux_buffer(num_tokens)
        target_logits = self.verify_logits(seqs, drafts, aux, verified_lens)
        mark = self._mark("verify", mark)
        self.drafter.after_verify(seqs, aux, verified_lens)
        mark = self._mark("ctx_store", mark)
        n_accept, tail = self._accept(target_logits, drafts, draft_probs, sampling, verified_lens)
        mark = self._mark("accept", mark)
        payload = torch.cat([drafts, n_accept.unsqueeze(1), tail.unsqueeze(1)], dim=1).tolist()   # the one D2H copy
        self._mark("commit_copy", mark)
        appended = []
        for row in payload:
            count, tail_token = row[self.num_drafts], row[self.num_drafts + 1]
            appended.append(row[:count] + [tail_token])
            self.stats[f"accepted_{count}"] += 1
        for length in verified_lens or ():
            self.stats[f"verified_{length}"] += 1
        self.stats["rounds"] += 1
        if self.record_confidences:
            self.confidence_log.append({"logits": confidence.tolist(), "accepted": [row[self.num_drafts] for row in payload]})
        return appended

    def _accept(self, target_logits: torch.Tensor, drafts: torch.Tensor, draft_probs: torch.Tensor | None,
                sampling: SamplingTensors, verified_lens: list[int] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Acceptance of every row's k drafts over [B, k+1, V] logits, or with verified_lens of each row's first
        verified_lens[r] - 1 over the flat [T, V] ones."""
        if verified_lens is None:
            if sampling.temperatures is None:
                return accept_greedy(target_logits, drafts)
            return accept_random(probs_from_logits(target_logits, sampling), draft_probs, drafts, self.target.generator)
        lens = self._device_tensor(verified_lens, torch.int64)
        if sampling.temperatures is None:
            return accept_greedy_varlen(target_logits, drafts, lens)
        target_probs = probs_from_logits(target_logits, sampling_per_token(sampling, lens, target_logits.shape[0]))
        return accept_random_varlen(target_probs, draft_probs, drafts, lens, self.target.generator)

    def _aux_buffer(self, num_tokens: int) -> torch.Tensor:
        """Where a verification of num_tokens tokens writes the aux features: the verify graph's buffer when the
        graphs exist, a new tensor otherwise."""
        if self.graphs:
            return self.graph_vars["aux"][:num_tokens]
        return torch.empty(num_tokens, self.drafter.aux_width, dtype=self.drafter.aux_dtype, device=self.device)

    @torch.inference_mode()
    def verify_logits(self, seqs: list[Sequence], drafts: torch.Tensor, aux_out: torch.Tensor | None = None,
                      verified_lens: list[int] | None = None) -> torch.Tensor:
        """Target logits at the verified positions. Without verified_lens every row verifies [last token, d1..dk]:
        [B, k+1, V]. With them row r verifies its last token and its first verified_lens[r] - 1 drafts, row after
        row: [T, V], T = sum(verified_lens). aux_out [T, aux width] receives the target's aux features (DFlash).
        With graphs the verification writes them into its own buffer, so aux_out must be the slice of it that
        _aux_buffer returns."""
        lens = [self.num_drafts + 1] * len(seqs) if verified_lens is None else verified_lens
        hidden = self._verify_graph(seqs, drafts, lens) if self.graphs else self._verify_eager(seqs, drafts, lens, aux_out)
        logits = self.target.model.compute_logits(hidden, all_logits=True)
        return logits if verified_lens is not None else logits.view(len(seqs), self.num_drafts + 1, -1)

    def _verify_eager(self, seqs: list[Sequence], drafts: torch.Tensor, verified_lens: list[int],
                      aux_out: torch.Tensor | None = None) -> torch.Tensor:
        width = max(len(seq.target_kv.block_table) for seq in seqs)
        metadata = verify_varlen_metadata(seqs, verified_lens, self.num_drafts, self.block_size, len(seqs),
                                          sum(verified_lens), self.target.pad_block_id, width)
        input_ids = self._device_tensor(metadata.input_ids, torch.int64)
        self._place_drafts(input_ids, drafts, metadata)
        positions = self._device_tensor(metadata.positions, torch.int64)
        set_context(True,
                    self._device_tensor(metadata.cu_seqlens_q, torch.int32),
                    self._device_tensor(metadata.cu_seqlens_k, torch.int32),
                    self.num_drafts + 1, self.max_model_len,
                    self._device_tensor(metadata.slot_mapping, torch.int32), None,
                    self._device_tensor(metadata.block_tables, torch.int32))
        hidden = self.target.run_model(input_ids, positions, True, need_logits=False, aux_out=aux_out)
        reset_context()
        return hidden

    def _verify_graph(self, seqs: list[Sequence], drafts: torch.Tensor, verified_lens: list[int]) -> torch.Tensor:
        """The smallest graph with as many tokens as the rows verify; its other tokens fill padding rows."""
        num_verified = sum(verified_lens)
        num_rows, num_tokens = next(shape for shape in self.graph_shapes if shape[1] >= num_verified)
        assert num_rows >= len(seqs)
        metadata = verify_varlen_metadata(seqs, verified_lens, self.num_drafts, self.block_size, num_rows, num_tokens,
                                          self.target.pad_block_id, self.max_num_blocks)
        self._load_graph_inputs(metadata, num_rows, num_tokens)
        self._place_drafts(self.graph_vars["input_ids"], drafts, metadata)
        self.graphs[num_tokens].replay()
        return self.graph_vars["outputs"][:num_verified]

    def _place_drafts(self, input_ids: torch.Tensor, drafts: torch.Tensor, metadata: VerifyMetadata) -> None:
        """Write the verified drafts into the flat input ids, where the metadata puts them."""
        source = self._device_tensor(metadata.draft_source, torch.int64)
        input_ids[self._device_tensor(metadata.draft_index, torch.int64)] = drafts.flatten()[source]

    def _load_graph_inputs(self, metadata: VerifyMetadata, num_rows: int, num_tokens: int) -> None:
        """The metadata into the verification graphs' buffers: num_tokens token entries, num_rows row entries."""
        buffers = self.graph_vars
        buffers["input_ids"][:num_tokens].copy_(self._host_tensor(metadata.input_ids, torch.int64), non_blocking=True)
        buffers["positions"][:num_tokens].copy_(self._host_tensor(metadata.positions, torch.int64), non_blocking=True)
        buffers["slot_mapping"][:num_tokens].copy_(self._host_tensor(metadata.slot_mapping, torch.int32), non_blocking=True)
        buffers["cu_seqlens_q"][:num_rows + 1].copy_(self._host_tensor(metadata.cu_seqlens_q, torch.int32), non_blocking=True)
        buffers["cu_seqlens_k"][:num_rows + 1].copy_(self._host_tensor(metadata.cu_seqlens_k, torch.int32), non_blocking=True)
        buffers["block_tables"][:num_rows].copy_(self._host_tensor(metadata.block_tables, torch.int32), non_blocking=True)

    # ------------------------------------------------------------------ graphs
    @torch.inference_mode()
    def capture_graphs(self) -> None:
        """One verification graph per (row slots, tokens) shape, into the target's pool. In fixed-length
        verification, B slots of k+1 tokens for each batch size B of the target's decode-graph table; with
        variable_length, each token count of verify_token_counts with a slot per token, up to the largest batch
        size. Each is captured on padding rows. The drafter then captures its block forward for the batch sizes."""
        query_len = self.num_drafts + 1
        self.graph_batch_sizes = list(self.target.graph_bs)
        max_batch_size = self.graph_batch_sizes[-1]
        if self.variable_length:
            self.graph_shapes = [(min(num_tokens, max_batch_size), num_tokens)
                                 for num_tokens in verify_token_counts(max_batch_size, query_len)]
        else:
            self.graph_shapes = [(batch_size, batch_size * query_len) for batch_size in self.graph_batch_sizes]
        hf_config = self.target.config.hf_config
        all_padding = verify_metadata([], self.num_drafts, self.block_size, max_batch_size,
                                      self.target.pad_block_id, self.max_num_blocks)
        self.graph_vars = {
            "input_ids": self._device_tensor(all_padding.input_ids, torch.int64),
            "positions": self._device_tensor(all_padding.positions, torch.int64),
            "slot_mapping": self._device_tensor(all_padding.slot_mapping, torch.int32),
            "cu_seqlens_q": self._device_tensor(all_padding.cu_seqlens_q, torch.int32),
            "cu_seqlens_k": self._device_tensor(all_padding.cu_seqlens_k, torch.int32),
            "block_tables": self._device_tensor(all_padding.block_tables, torch.int32),
            "outputs": torch.zeros(max_batch_size * query_len, hf_config.hidden_size, dtype=hf_config.dtype, device=self.device),
            # the graphs also export the target's aux features
            "aux": torch.zeros(max_batch_size * query_len, self.drafter.aux_width, dtype=self.drafter.aux_dtype,
                               device=self.device),
        }
        with torch.device(self.device):
            for num_rows, num_tokens in reversed(self.graph_shapes):
                padding = verify_varlen_metadata([], [], self.num_drafts, self.block_size, num_rows, num_tokens,
                                                 self.target.pad_block_id, self.max_num_blocks)
                self._load_graph_inputs(padding, num_rows, num_tokens)
                graph = torch.cuda.CUDAGraph()
                self._set_graph_context(num_rows, num_tokens)
                self.graph_vars["outputs"][:num_tokens] = self._graph_forward(num_tokens)    # warm-up
                with torch.cuda.graph(graph, self.target.graph_pool):
                    self.graph_vars["outputs"][:num_tokens] = self._graph_forward(num_tokens)
                self.graphs[num_tokens] = graph
                torch.cuda.synchronize()
                reset_context()
        self.drafter.capture_graphs()

    def _graph_forward(self, num_tokens: int) -> torch.Tensor:
        return self.target.model(self.graph_vars["input_ids"][:num_tokens], self.graph_vars["positions"][:num_tokens],
                                 self.graph_vars["aux"][:num_tokens])

    def _set_graph_context(self, num_rows: int, num_tokens: int) -> None:
        buffers = self.graph_vars
        set_context(True, buffers["cu_seqlens_q"][:num_rows + 1], buffers["cu_seqlens_k"][:num_rows + 1],
                    self.num_drafts + 1, self.max_model_len, buffers["slot_mapping"][:num_tokens], None,
                    buffers["block_tables"][:num_rows])

    def close(self) -> None:
        self.graphs.clear()
        self.graph_vars.clear()
        self.drafter.graphs.clear()
        self.drafter.sampling_graphs.clear()

    # ------------------------------------------------------------------ tensors
    def _host_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        # on the CPU even inside a torch.device context: graph capture loads its padding rows in one
        return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=self.device.type == "cuda")

    def _device_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return self._host_tensor(values, dtype).to(self.device, non_blocking=True)
