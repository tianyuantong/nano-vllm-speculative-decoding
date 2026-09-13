"""Offline drain-batch B/S0 control flow. No CUDA or probability implementation.

Backend supplies batched forward, sampling, and a synchronized commit boundary.
Request history is authoritative; Query histories are temporary proposal views.
Failures abort the entire generate call, with no partial output or RNG retry.
"""

from dataclasses import dataclass, field
import math

from nanovllm.engine.kv_state import KVState
from nanovllm.engine.ngram import history_proposal


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_tokens: int
    rng: object
    temperature: float = 1.0
    ignore_eos: bool = False
    tokens: list[int] = field(init=False)
    target: KVState = field(default_factory=KVState, init=False)
    draft: KVState = field(default_factory=KVState, init=False)
    stop: str | None = field(default=None, init=False)

    def __post_init__(self):
        self.prompt = list(self.prompt)
        self.tokens = list(self.prompt)


@dataclass
class Query:
    request: Request
    tokens: list[int]
    kv: KVState
    block_size: int

    @property
    def num_cached_tokens(self):
        return self.kv.num_cached_tokens

    @property
    def num_scheduled_tokens(self):
        return len(self.tokens) - self.num_cached_tokens

    @property
    def block_table(self):
        return self.kv.block_table

    @property
    def last_token(self):
        return self.tokens[-1]

    @property
    def last_block_num_tokens(self):
        return (len(self.tokens) - 1) % self.block_size + 1

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        return self.tokens[key]


class PrivateKVPool:
    """No sharing/prefix cache/preemption. Owns only this offline engine's blocks."""

    def __init__(self, num_blocks, block_size):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("positive KV capacity required")
        self.block_size = block_size
        self.free = list(reversed(range(num_blocks)))
        self.used = set()

    def reserve(self, kv, length):
        needed = (length + self.block_size - 1) // self.block_size
        extra = max(0, needed - len(kv.block_table))
        if extra > len(self.free):
            raise MemoryError("offline batch exceeds explicit KV budget")
        for _ in range(extra):
            block = self.free.pop()
            self.used.add(block)
            kv.block_table.append(block)

    def truncate(self, kv, cached):
        # Caller may invalidate already computed suffixes, never invent KV.
        if not 0 <= cached <= kv.num_cached_tokens:
            raise ValueError("truncation cannot compute KV")
        needed = (cached + self.block_size - 1) // self.block_size
        while len(kv.block_table) > needed:
            block = kv.block_table.pop()
            self.used.remove(block)
            self.free.append(block)
        kv.num_cached_tokens = cached


class RandomDecode:
    def __init__(self, backend, *, target_blocks, draft_blocks, block_size,
                 max_model_len, max_num_seqs, vocab_size, eos, k=4, gpu_draft_tokens=False,
                 ngram=False):
        if type(k) is not int or not 0 <= k <= 4:
            raise ValueError("this controller supports B(k=0) or S0(k=1..4)")
        self.backend = backend
        self.target_pool = PrivateKVPool(target_blocks, block_size)
        if ngram and (not k or draft_blocks or gpu_draft_tokens):
            raise ValueError("N requires k>0, no draft pool and no device draft continuation")
        self.ngram = ngram
        self.draft_pool = PrivateKVPool(draft_blocks, block_size) if k and not ngram else None
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.vocab_size = vocab_size
        self.eos = eos
        if gpu_draft_tokens and not k:
            raise ValueError("GPU draft continuation requires a draft model")
        self.gpu_draft_tokens = gpu_draft_tokens
        self.k = k
        self.poisoned = False
        self.busy = False

    def _remaining(self, request):
        return min(request.max_tokens - (len(request.tokens) - len(request.prompt)),
                   self.max_model_len - len(request.tokens))

    def _forward(self, role, items, *, all_logits=False, device_tokens=None):
        if not items:
            return []
        pool = self.target_pool if role == "target" else self.draft_pool
        queries = []
        for request, tokens in items:
            kv = getattr(request, role)
            if not kv.num_cached_tokens < len(tokens) <= self.max_model_len:
                raise ValueError("forward must compute a nonempty uncached suffix")
            if any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
                raise ValueError("out-of-vocabulary token before embedding access")
            pool.reserve(kv, len(tokens))
            queries.append(Query(request, tokens, kv, self.block_size))
        if device_tokens is None:
            result = self.backend.forward(role, queries, all_logits=all_logits)
        else:
            if role != "draft" or all_logits or any(q.num_scheduled_tokens != 1 for q in queries):
                raise ValueError("device continuation is draft single-step only")
            result = self.backend.forward_draft_device(queries, device_tokens)
        if len(result) != len(queries):
            raise RuntimeError("backend batch mismatch")
        for query in queries:
            query.kv.num_cached_tokens = len(query.tokens)
        return result

    def _commit(self, request, tokens):
        if not tokens or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise ValueError("invalid commit tokens")
        for token in tokens:
            if self._remaining(request) <= 0:
                raise RuntimeError("commit beyond budget")
            request.tokens.append(token)
            if not request.ignore_eos and token == self.eos:
                request.stop = "eos"
            elif len(request.tokens) >= self.max_model_len:
                request.stop = "context"
            elif len(request.tokens) - len(request.prompt) >= request.max_tokens:
                request.stop = "length"
            if request.stop:
                break
        if request.stop:
            self._release(request)
        else:
            for role, pool in (("target", self.target_pool), ("draft", self.draft_pool)):
                if pool is not None:
                    kv = getattr(request, role)
                    pool.truncate(kv, min(kv.num_cached_tokens, len(request.tokens) - 1))

    def _release(self, request):
        self.target_pool.truncate(request.target, 0)
        if self.draft_pool is not None:
            self.draft_pool.truncate(request.draft, 0)

    def generate(self, requests):
        if self.poisoned or self.busy:
            raise RuntimeError("engine is aborted or already generating")
        if not requests or len(requests) > self.max_num_seqs:
            raise ValueError("one nonempty drain batch within max_num_seqs required")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("request IDs must be unique within a batch")
        for r in requests:
            if (not r.prompt or type(r.max_tokens) is not int or r.max_tokens <= 0
                    or not math.isfinite(r.temperature) or r.temperature <= 0
                    or len(r.prompt) >= self.max_model_len):
                raise ValueError("invalid prompt, output budget or temperature")
            if r.tokens != r.prompt or r.stop or r.target.block_table or r.draft.block_table:
                raise ValueError("fresh requests and fresh RNG streams required")
        self.busy = True
        try:
            logits = self._forward("target", [(r, r.prompt) for r in requests])
            first = self.backend.sample_batch(requests, logits, "ordinary_target")
            # All first-token checks complete before any request state commits.
            for r, token in zip(requests, first, strict=True):
                self._commit(r, [token])
            active = [r for r in requests if not r.stop]
            if self.draft_pool is not None:
                self._forward("draft", [(r, r.prompt) for r in active])
            while active:
                if not self.k:
                    logits = self._forward("target", [(r, r.tokens) for r in active])
                    values = self.backend.sample_batch(active, logits, "ordinary_target")
                    for r, token in zip(active, values, strict=True):
                        self._commit(r, [token])
                elif self.ngram:
                    self._ngram_round(active)
                else:
                    self._round(active)
                active = [r for r in active if not r.stop]
            self.backend.synchronize()
            return [{"request_id": r.request_id, "token_ids": r.tokens[len(r.prompt):],
                     "stop_reason": r.stop} for r in requests]
        except BaseException as error:
            # No partial result escapes generate. Failed CUDA engines are not reused.
            self.poisoned = True
            try:
                self.backend.synchronize()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            finally:
                for r in requests:
                    self._release(r)
            raise
        finally:
            self.busy = False

    def _round(self, active):
        budgets = {r.request_id: min(self.k, self._remaining(r) - 1) for r in active}
        # Full acceptance leaves exactly one missing draft KV (the final draft).
        # A request with only one output left finishes on target; no draft will
        # consume this missing KV, even if other batch rows still need proposals.
        catchup = [(r, r.tokens[:-1]) for r in active
                   if budgets[r.request_id] > 0 and r.draft.num_cached_tokens < len(r.tokens) - 1]
        for r in active:
            if r.target.num_cached_tokens != len(r.tokens) - 1:
                raise RuntimeError("target round-entry KV invariant violated")
            if not len(r.tokens) - 2 <= r.draft.num_cached_tokens <= len(r.tokens) - 1:
                raise RuntimeError("draft round-entry KV invariant violated")
        self._forward("draft", catchup)  # actual forward, consumes no RNG
        proposals = {r.request_id: [] for r in active}
        probabilities = {r.request_id: [] for r in active}
        for step in range(max(budgets.values(), default=0)):
            rows = [r for r in active if step < budgets[r.request_id]]
            if self.gpu_draft_tokens:
                if any(len(proposals[r.request_id]) != step for r in rows):
                    raise RuntimeError("device proposal step/request mapping mismatch")
                # CPU temporary views carry lengths only for the device suffix.
                # Their placeholders never enter embedding or authoritative history.
                items = [(r, r.tokens + [0] * step) for r in rows]
                previous = [proposals[r.request_id][-1] for r in rows] if step else None
                logits = self._forward("draft", items, device_tokens=previous)
                tokens, saved_q = self.backend.propose_device(rows, logits)
            else:
                logits = self._forward("draft", [(r, r.tokens + proposals[r.request_id]) for r in rows])
                tokens, saved_q = self.backend.propose(rows, logits)
            for r, token, q in zip(rows, tokens, saved_q, strict=True):
                proposals[r.request_id].append(token)
                probabilities[r.request_id].append(q)
        if self.gpu_draft_tokens:
            proposals = self.backend.materialize_proposals(proposals)
        self._verify_proposals(active, proposals, probabilities)

    def _ngram_round(self, active):
        if any(r.target.num_cached_tokens != len(r.tokens) - 1 for r in active):
            raise RuntimeError("target round-entry KV invariant violated")
        proposals = {r.request_id: history_proposal(
            r.tokens, min(self.k, self._remaining(r) - 1)) for r in active}
        self._verify_proposals(active, proposals, None)

    def _verify_proposals(self, active, proposals, probabilities):
        # S0/S1 crop after the fixed draft budget; N has no proposal RNG.
        for r in active:
            ds = proposals[r.request_id]
            if not r.ignore_eos and self.eos in ds:
                length = ds.index(self.eos) + 1
                del ds[length:]
                if probabilities is not None:
                    del probabilities[r.request_id][length:]
        logits = self._forward("target", [(r, r.tokens + proposals[r.request_id]) for r in active], all_logits=True)
        if probabilities is None:
            probabilities = self.backend.point_mass_proposals(active, logits, proposals)
        pending = self.backend.verify(active, logits, proposals, probabilities)
        # Backend checks the entire batch before returning; no post-reject suffix.
        for r, tokens in zip(active, pending, strict=True):
            self._commit(r, tokens)
