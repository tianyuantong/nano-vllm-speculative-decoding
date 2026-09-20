"""One speculative round per decode batch on top of a target and a draft ModelRunner."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

from nanovllm.engine.batch_metadata import decode_metadata, padded_block_tables, verify_metadata
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


class SpeculativeDecoder:
    """Draft catch-up, k draft steps, one verification forward, batched acceptance, one D2H copy.

    Round-entry invariants (asserted): target KV covers len-1 positions; draft KV covers
    len-2 or len-1. Every active sequence proposes exactly k drafts, so the verification
    shape is (B, k+1) and a graph keyed by the padded batch size always matches.
    """

    def __init__(self, target: ModelRunner, draft: ModelRunner, config: Config):
        self.target = target
        self.draft = draft
        self.num_drafts = config.num_speculative_tokens
        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.vocab_size = config.hf_config.vocab_size
        self.device = target.device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_batch_sizes: list[int] = []
        self.graph_vars: dict[str, torch.Tensor] = {}
        self.stats: Counter = Counter()
        self._phase_events: list[tuple[str, object, object]] = []
        self._new_event = None
        self._synchronize = None

    # ------------------------------------------------------------------ draft KV only
    @torch.inference_mode()
    def prefill_draft(self, seqs: list[Sequence]) -> None:
        """Run the draft over the chunk the target just prefilled; the scheduler set draft_kv.num_scheduled_tokens."""
        self._draft_kv_forward(seqs)

    def _draft_kv_forward(self, seqs: list[Sequence]) -> None:
        input_ids, positions = self.draft.prepare_prefill(seqs)
        self.draft.run_model(input_ids, positions, True, need_logits=False)
        reset_context()

    def _catch_up(self, seqs: list[Sequence]) -> None:
        """Recompute the one draft KV entry (position len-2) a fully accepted round left missing.

        By the round-entry invariant the gap is exactly one token, so this is a single decode
        step of the draft through its decode graph; no logits are needed.
        """
        lagging = [seq for seq in seqs if seq.draft_kv.num_cached_tokens < len(seq) - 1]
        if not lagging:
            return
        metadata = decode_metadata(lagging, "draft", self.block_size, num_steps=1, position_offset=-1)
        set_context(False,
                    slot_mapping=self._device_tensor(metadata.slot_mapping[0], torch.int32),
                    context_lens=self._device_tensor(metadata.context_lens, torch.int32),
                    block_tables=self._device_tensor(padded_block_tables(lagging, "draft"), torch.int32))
        self.draft.run_model(self._device_tensor(metadata.input_ids, torch.int64),
                             self._device_tensor(metadata.positions, torch.int64), False, need_logits=False)
        reset_context()
        for seq in lagging:
            seq.draft_kv.num_cached_tokens = len(seq) - 1
        self.stats["catch_up_forwards"] += 1

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
            assert len(seq) - 2 <= seq.draft_kv.num_cached_tokens <= len(seq) - 1
        mark = self._mark("start", None)
        self._catch_up(seqs)
        mark = self._mark("catch_up", mark)
        sampling = sampling_tensors(seqs, self.device)
        drafts, draft_probs = self._propose(seqs, sampling)
        mark = self._mark("propose", mark)
        target_logits = self.verify_logits(seqs, drafts)
        mark = self._mark("verify", mark)
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

    @torch.inference_mode()
    def _propose(self, seqs: list[Sequence], sampling: SamplingTensors) -> tuple[torch.Tensor, torch.Tensor | None]:
        """k draft steps through the draft's decode path; returns drafts [B, k] and probabilities [B, k, V] (None when greedy)."""
        batch_size, k = len(seqs), self.num_drafts
        metadata = decode_metadata(seqs, "draft", self.block_size, num_steps=k)
        input_ids = self._device_tensor(metadata.input_ids, torch.int64)
        positions = self._device_tensor(metadata.positions, torch.int64)
        context_lens = self._device_tensor(metadata.context_lens, torch.int32)
        slot_mapping = self._device_tensor(metadata.slot_mapping, torch.int32)          # [k, B]
        block_tables = self._device_tensor(padded_block_tables(seqs, "draft"), torch.int32)
        drafts = torch.empty(batch_size, k, dtype=torch.int64, device=self.device)
        draft_probs = None
        if sampling.temperatures is not None:
            draft_probs = torch.empty(batch_size, k, self.vocab_size, dtype=torch.float32, device=self.device)
        for step in range(k):
            set_context(False, slot_mapping=slot_mapping[step], context_lens=context_lens + step, block_tables=block_tables)
            logits = self.draft.run_model(input_ids, positions + step, False)
            if draft_probs is None:
                tokens = logits.argmax(dim=-1)
            else:
                draft_probs[:, step] = probs_from_logits(logits, sampling)
                tokens = sample(draft_probs[:, step], self.target.generator)
            drafts[:, step] = tokens
            input_ids = tokens
        reset_context()
        return drafts, draft_probs

    @torch.inference_mode()
    def verify_logits(self, seqs: list[Sequence], drafts: torch.Tensor) -> torch.Tensor:
        """Target logits at the k+1 positions [last token, d1..dk] of every sequence: [B, k+1, V]."""
        batch_size, query_len = len(seqs), self.num_drafts + 1
        hidden = self._verify_graph(seqs, drafts) if self.graphs else self._verify_eager(seqs, drafts)
        logits = self.target.model.compute_logits(hidden, all_logits=True)
        return logits.view(batch_size, query_len, -1)

    def _verify_eager(self, seqs: list[Sequence], drafts: torch.Tensor) -> torch.Tensor:
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
        hidden = self.target.run_model(input_ids, positions, True, need_logits=False)
        reset_context()
        return hidden

    def _verify_graph(self, seqs: list[Sequence], drafts: torch.Tensor) -> torch.Tensor:
        batch_size, query_len = len(seqs), self.num_drafts + 1
        padded = next(size for size in self.graph_batch_sizes if size >= batch_size)
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
        """One verification graph per batch size of the target's decode-graph table; query length fixed to k+1."""
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

    def _graph_forward(self, rows: int) -> torch.Tensor:
        return self.target.model(self.graph_vars["input_ids"][:rows], self.graph_vars["positions"][:rows])

    def _set_graph_context(self, batch_size: int) -> None:
        rows = batch_size * (self.num_drafts + 1)
        buffers = self.graph_vars
        set_context(True, buffers["cu_seqlens_q"][:batch_size + 1], buffers["cu_seqlens_k"][:batch_size + 1],
                    self.num_drafts + 1, self.max_model_len, buffers["slot_mapping"][:rows], None,
                    buffers["block_tables"][:batch_size])

    def close(self) -> None:
        self.graphs.clear()
        self.graph_vars.clear()

    # ------------------------------------------------------------------ tensors
    def _host_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, pin_memory=self.device.type == "cuda")

    def _device_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return self._host_tensor(values, dtype).to(self.device, non_blocking=True)
