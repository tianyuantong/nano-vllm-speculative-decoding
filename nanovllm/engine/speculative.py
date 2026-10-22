"""One speculative round per decode batch on top of a target and a draft ModelRunner."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

from nanovllm.engine.batch_metadata import (decode_metadata, draft_block_metadata, padded_block_tables,
                                            prefill_metadata, verify_metadata)
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.spec_sampler import (
    SamplingTensors,
    accept_greedy,
    accept_random,
    probs_from_logits,
    sample,
    sampling_tensors,
)
from nanovllm.utils.context import reset_context, set_context

if TYPE_CHECKING:
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner


class ModelDrafter:
    """A draft model with its own KV cache and decode path (the 0.6B line).

    At round entry its KV covers len-2 or len-1 positions: catch_up fills the missing one, then propose
    runs k single-token decode steps.
    """

    def __init__(self, draft: ModelRunner, decoder: SpeculativeDecoder):
        self.draft = draft
        self.decoder = decoder    # block size, k, vocabulary, device, the target's generator and the round stats

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence], aux: None) -> None:
        """Run the draft over the chunk the target just prefilled; the scheduler set draft_kv.num_scheduled_tokens.
        aux is None: this draft reads tokens, not target features."""
        input_ids, positions = self.draft.prepare_prefill(seqs)
        self.draft.run_model(input_ids, positions, True, need_logits=False)
        reset_context()

    def catch_up(self, seqs: list[Sequence]) -> None:
        """Recompute the one draft KV entry (position len-2) a fully accepted round left missing.

        By the round-entry invariant the gap is exactly one token, so this is a single decode
        step of the draft through its decode graph; no logits are needed.
        """
        decoder = self.decoder
        for seq in seqs:
            assert len(seq) - 2 <= seq.draft_kv.num_cached_tokens <= len(seq) - 1
        lagging = [seq for seq in seqs if seq.draft_kv.num_cached_tokens < len(seq) - 1]
        if not lagging:
            return
        metadata = decode_metadata(lagging, "draft", decoder.block_size, num_steps=1, position_offset=-1)
        set_context(False,
                    slot_mapping=decoder._device_tensor(metadata.slot_mapping[0], torch.int32),
                    context_lens=decoder._device_tensor(metadata.context_lens, torch.int32),
                    block_tables=decoder._device_tensor(padded_block_tables(lagging, "draft"), torch.int32))
        self.draft.run_model(decoder._device_tensor(metadata.input_ids, torch.int64),
                             decoder._device_tensor(metadata.positions, torch.int64), False, need_logits=False)
        reset_context()
        for seq in lagging:
            seq.draft_kv.num_cached_tokens = len(seq) - 1
        decoder.stats["catch_up_forwards"] += 1

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """k draft steps through the draft's decode path; returns drafts [B, k] and probabilities [B, k, V] (None when greedy)."""
        decoder = self.decoder
        batch_size, k = len(seqs), decoder.num_drafts
        metadata = decode_metadata(seqs, "draft", decoder.block_size, num_steps=k)
        input_ids = decoder._device_tensor(metadata.input_ids, torch.int64)
        positions = decoder._device_tensor(metadata.positions, torch.int64)
        context_lens = decoder._device_tensor(metadata.context_lens, torch.int32)
        slot_mapping = decoder._device_tensor(metadata.slot_mapping, torch.int32)          # [k, B]
        block_tables = decoder._device_tensor(padded_block_tables(seqs, "draft"), torch.int32)
        drafts = torch.empty(batch_size, k, dtype=torch.int64, device=decoder.device)
        draft_probs = None
        if sampling.temperatures is not None:
            draft_probs = torch.empty(batch_size, k, decoder.vocab_size, dtype=torch.float32, device=decoder.device)
        for step in range(k):
            set_context(False, slot_mapping=slot_mapping[step], context_lens=context_lens + step, block_tables=block_tables)
            logits = self.draft.run_model(input_ids, positions + step, False)
            if draft_probs is None:
                tokens = logits.argmax(dim=-1)
            else:
                draft_probs[:, step] = probs_from_logits(logits, sampling)
                tokens = sample(draft_probs[:, step], decoder.target.generator)
            drafts[:, step] = tokens
            input_ids = tokens
        reset_context()
        return drafts, draft_probs


class DFlashDrafter:
    """DeepSpec's DFlash draft: one forward over a block [anchor, mask x (k-1)] proposes k drafts.

    Its KV holds context K/V computed from the target's aux features, written for every position the target
    computes: each prefilled chunk, and all k+1 positions of each verification. A rejected position is
    overwritten by the next block before any block reads it, so at round entry the context covers len-1
    positions, like the target's KV. It shares the target's block table: same block count, same pad block.
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

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence], aux: torch.Tensor) -> None:
        """Context K/V of the chunk the target just prefilled, from the aux features its forward exported."""
        decoder = self.decoder
        metadata = prefill_metadata(seqs, "target", decoder.block_size)
        self.model.store_context_kv(self.model.context_features(aux),
                                    decoder._device_tensor(metadata.positions, torch.int64),
                                    decoder._device_tensor(metadata.slot_mapping, torch.int32))

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One block forward; returns drafts [B, k] and probabilities [B, k, V] (None when greedy)."""
        decoder, k = self.decoder, self.decoder.num_drafts
        metadata = draft_block_metadata(seqs, decoder.block_size, k, self.model.mask_token_id)
        set_context(False, slot_mapping=decoder._device_tensor(metadata.slot_mapping, torch.int32),
                    context_lens=decoder._device_tensor(metadata.context_lens, torch.int32),
                    block_tables=decoder._device_tensor(padded_block_tables(seqs, "target"), torch.int32),
                    query_len_per_seq=k, causal=False)
        hidden = self.model(decoder._device_tensor(metadata.input_ids, torch.int64),
                            decoder._device_tensor(metadata.positions, torch.int64))
        reset_context()
        logits = self.model.compute_logits(hidden).view(len(seqs), k, -1)
        if sampling.temperatures is None:
            return logits.argmax(dim=-1), None
        draft_probs = probs_from_logits(logits, sampling)
        return sample(draft_probs, decoder.target.generator), draft_probs

    @torch.inference_mode()
    def after_verify(self, seqs: list[Sequence], aux: torch.Tensor) -> None:
        """Context K/V of all k+1 verified positions, from the aux features the verification exported."""
        decoder = self.decoder
        width = max(len(seq.target_kv.block_table) for seq in seqs)
        metadata = verify_metadata(seqs, decoder.num_drafts, decoder.block_size, len(seqs), decoder.target.pad_block_id, width)
        self.model.store_context_kv(self.model.context_features(aux),
                                    decoder._device_tensor(metadata.positions, torch.int64),
                                    decoder._device_tensor(metadata.slot_mapping, torch.int32))


class SpeculativeDecoder:
    """Drafts, one verification forward, batched acceptance, one D2H copy.

    The drafter is a ModelDrafter when a draft ModelRunner is given (draft_type "model"), else a DFlashDrafter.
    Round-entry invariant (asserted): target KV covers len-1 positions; each drafter asserts its own. Every
    active sequence proposes exactly k drafts, so the verification shape is (B, k+1) and a graph keyed by the
    padded batch size always matches.
    """

    def __init__(self, target: ModelRunner, draft: ModelRunner | None, config: Config):
        self.target = target
        self.num_drafts = config.num_speculative_tokens
        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.vocab_size = config.hf_config.vocab_size
        self.device = target.device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.ctx_graphs: dict[int, torch.cuda.CUDAGraph] = {}    # DFlash: the context store, per padded batch size
        self.graph_batch_sizes: list[int] = []
        self.graph_vars: dict[str, torch.Tensor] = {}
        self.stats: Counter = Counter()
        self._phase_events: list[tuple[str, object, object]] = []
        self._new_event = None
        self._synchronize = None
        self.drafter = ModelDrafter(draft, self) if draft is not None else DFlashDrafter(config, self)

    # ------------------------------------------------------------------ prefill
    def prefill_aux(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """The buffer the target's prefill exports aux features into: DFlash only."""
        if not isinstance(self.drafter, DFlashDrafter):
            return None
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        return torch.empty(num_tokens, self.drafter.aux_width, dtype=self.drafter.aux_dtype, device=self.device)

    @torch.inference_mode()
    def prefill_draft(self, seqs: list[Sequence], aux: torch.Tensor | None = None) -> None:
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
    def run_round(self, seqs: list[Sequence]) -> list[list[int]]:
        """Return, per sequence, the accepted drafts followed by the tail token."""
        for seq in seqs:
            assert seq.target_kv.num_cached_tokens == len(seq) - 1
        mark = self._mark("start", None)
        if isinstance(self.drafter, ModelDrafter):
            self.drafter.catch_up(seqs)
            mark = self._mark("catch_up", mark)
        sampling = sampling_tensors(seqs, self.device)
        drafts, draft_probs = self.drafter.propose(seqs, sampling)
        mark = self._mark("propose", mark)
        aux = self._aux_buffer(len(seqs)) if isinstance(self.drafter, DFlashDrafter) else None
        target_logits = self.verify_logits(seqs, drafts, aux)
        mark = self._mark("verify", mark)
        if aux is not None:
            if self.ctx_graphs:
                self.ctx_graphs[self._padded_batch_size(len(seqs))].replay()
            else:
                self.drafter.after_verify(seqs, aux)
            mark = self._mark("ctx_store", mark)
        if sampling.temperatures is None:
            n_accept, tail = accept_greedy(target_logits, drafts)
        else:
            target_probs = probs_from_logits(target_logits, sampling)
            n_accept, tail = accept_random(target_probs, draft_probs, drafts, self.target.generator)
        mark = self._mark("accept", mark)
        payload = torch.cat([drafts, n_accept.unsqueeze(1), tail.unsqueeze(1)], dim=1).tolist()   # the one D2H copy
        self._mark("commit_copy", mark)
        appended = []
        for row in payload:
            count, tail_token = row[self.num_drafts], row[self.num_drafts + 1]
            appended.append(row[:count] + [tail_token])
            self.stats[f"accepted_{count}"] += 1
        self.stats["rounds"] += 1
        return appended

    def _aux_buffer(self, batch_size: int) -> torch.Tensor:
        """Where the verification of batch_size rows writes the aux features: the verify graph's buffer when the
        graphs exist, a new tensor otherwise."""
        rows = batch_size * (self.num_drafts + 1)
        if self.graphs:
            return self.graph_vars["aux"][:rows]
        return torch.empty(rows, self.drafter.aux_width, dtype=self.drafter.aux_dtype, device=self.device)

    @torch.inference_mode()
    def verify_logits(self, seqs: list[Sequence], drafts: torch.Tensor, aux_out: torch.Tensor | None = None) -> torch.Tensor:
        """Target logits at the k+1 positions [last token, d1..dk] of every sequence: [B, k+1, V].
        aux_out [B*(k+1), aux width] receives the target's aux features (DFlash). With graphs the verification
        writes them into its own buffer, so aux_out must be the slice of it that _aux_buffer returns."""
        batch_size, query_len = len(seqs), self.num_drafts + 1
        hidden = self._verify_graph(seqs, drafts) if self.graphs else self._verify_eager(seqs, drafts, aux_out)
        logits = self.target.model.compute_logits(hidden, all_logits=True)
        return logits.view(batch_size, query_len, -1)

    def _verify_eager(self, seqs: list[Sequence], drafts: torch.Tensor, aux_out: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, query_len = len(seqs), self.num_drafts + 1
        width = max(len(seq.target_kv.block_table) for seq in seqs)
        metadata = verify_metadata(seqs, self.num_drafts, self.block_size, batch_size, self.target.pad_block_id, width)
        input_ids = self._device_tensor(metadata.input_ids, torch.int64)
        input_ids.view(batch_size, query_len)[:, 1:] = drafts
        positions = self._device_tensor(metadata.positions, torch.int64)
        set_context(True,
                    self._device_tensor(metadata.cu_seqlens_q, torch.int32),
                    self._device_tensor(metadata.cu_seqlens_k, torch.int32),
                    query_len, self.max_model_len,
                    self._device_tensor(metadata.slot_mapping, torch.int32), None,
                    self._device_tensor(metadata.block_tables, torch.int32))
        hidden = self.target.run_model(input_ids, positions, True, need_logits=False, aux_out=aux_out)
        reset_context()
        return hidden

    def _verify_graph(self, seqs: list[Sequence], drafts: torch.Tensor) -> torch.Tensor:
        batch_size, query_len = len(seqs), self.num_drafts + 1
        padded = self._padded_batch_size(batch_size)
        metadata = verify_metadata(seqs, self.num_drafts, self.block_size, padded, self.target.pad_block_id, self.max_num_blocks)
        num_tokens = padded * query_len
        buffers = self.graph_vars
        buffers["input_ids"][:num_tokens].copy_(self._host_tensor(metadata.input_ids, torch.int64), non_blocking=True)
        buffers["input_ids"].view(-1, query_len)[:batch_size, 1:] = drafts
        buffers["positions"][:num_tokens].copy_(self._host_tensor(metadata.positions, torch.int64), non_blocking=True)
        buffers["slot_mapping"][:num_tokens].copy_(self._host_tensor(metadata.slot_mapping, torch.int32), non_blocking=True)
        buffers["cu_seqlens_q"][:padded + 1].copy_(self._host_tensor(metadata.cu_seqlens_q, torch.int32), non_blocking=True)
        buffers["cu_seqlens_k"][:padded + 1].copy_(self._host_tensor(metadata.cu_seqlens_k, torch.int32), non_blocking=True)
        buffers["block_tables"][:padded].copy_(self._host_tensor(metadata.block_tables, torch.int32), non_blocking=True)
        self.graphs[padded].replay()
        return buffers["outputs"][:batch_size * query_len]

    # ------------------------------------------------------------------ graphs
    @torch.inference_mode()
    def capture_graphs(self) -> None:
        """One verification graph per batch size of the target's decode-graph table; query length fixed to k+1.
        DFlash also gets one context-store graph per batch size."""
        query_len = self.num_drafts + 1
        self.graph_batch_sizes = list(self.target.graph_bs)
        max_batch_size = self.graph_batch_sizes[-1]
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
        }
        if isinstance(self.drafter, DFlashDrafter):    # the graphs also export the target's aux features
            self.graph_vars["aux"] = torch.zeros(max_batch_size * query_len, self.drafter.aux_width,
                                                 dtype=self.drafter.aux_dtype, device=self.device)
        with torch.device(self.device):
            for batch_size in reversed(self.graph_batch_sizes):
                rows = batch_size * query_len
                graph = torch.cuda.CUDAGraph()
                self._set_graph_context(batch_size)
                self.graph_vars["outputs"][:rows] = self._graph_forward(rows)    # warm-up
                with torch.cuda.graph(graph, self.target.graph_pool):
                    self.graph_vars["outputs"][:rows] = self._graph_forward(rows)
                self.graphs[batch_size] = graph
                torch.cuda.synchronize()
                reset_context()
        if isinstance(self.drafter, DFlashDrafter):
            self._capture_ctx_graphs()

    def _padded_batch_size(self, batch_size: int) -> int:
        return next(size for size in self.graph_batch_sizes if size >= batch_size)

    def _capture_ctx_graphs(self) -> None:
        """DFlash: per padded batch size, the context store of a verification, reading the verify graph's aux,
        positions and slot mapping. Padding rows carry slot -1, so their K/V are never stored."""
        query_len = self.num_drafts + 1
        for batch_size in reversed(self.graph_batch_sizes):
            rows = batch_size * query_len
            graph = torch.cuda.CUDAGraph()
            self._ctx_store_forward(rows)    # warm-up
            with torch.cuda.graph(graph, self.target.graph_pool):
                self._ctx_store_forward(rows)
            self.ctx_graphs[batch_size] = graph
            torch.cuda.synchronize()

    def _ctx_store_forward(self, rows: int) -> None:
        model, buffers = self.drafter.model, self.graph_vars
        model.store_context_kv(model.context_features(buffers["aux"][:rows]), buffers["positions"][:rows],
                               buffers["slot_mapping"][:rows])

    def _graph_forward(self, rows: int) -> torch.Tensor:
        aux = self.graph_vars["aux"][:rows] if "aux" in self.graph_vars else None
        return self.target.model(self.graph_vars["input_ids"][:rows], self.graph_vars["positions"][:rows], aux)

    def _set_graph_context(self, batch_size: int) -> None:
        rows = batch_size * (self.num_drafts + 1)
        buffers = self.graph_vars
        set_context(True, buffers["cu_seqlens_q"][:batch_size + 1], buffers["cu_seqlens_k"][:batch_size + 1],
                    self.num_drafts + 1, self.max_model_len, buffers["slot_mapping"][:rows], None,
                    buffers["block_tables"][:batch_size])

    def close(self) -> None:
        self.graphs.clear()
        self.ctx_graphs.clear()
        self.graph_vars.clear()

    # ------------------------------------------------------------------ tensors
    def _host_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, pin_memory=self.device.type == "cuda")

    def _device_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return self._host_tensor(values, dtype).to(self.device, non_blocking=True)
