# Design: batched speculative decoding on the upstream engine

Status: approved design, not yet implemented. Supersedes the `RandomLLM`
offline controller (PRs #3–#7) and the plans under `docs/plans/`.

## 1. Goal and success criterion

Add speculative decoding to the upstream nano-vLLM engine so that it runs
under the upstream scheduler (continuous batching) with one batched sampler,
one CUDA graph family for verification, and one host synchronization per
round. The `k = 0` configuration must be the upstream engine itself, so the
baseline of every measurement is `LLM.generate` without a draft model.

Success: `R = Q_spec / Q_k0 >= 1.15`, where `Q` is completion tokens per
second of a whole `generate()` call, at the same `max_num_seqs`, on the same
GPU model, with the sampling parameters of §7. Throughput and latency are
both reported (§9).

The measured reason the previous implementation broke even is recorded in
`docs/PERFORMANCE.md`: a speculative round cost ≈20 ms fixed + 2.8 ms per
request, of which ≈11 ms at batch 4 was per-request Python sampling and
≈4 ms host synchronization; mixed query lengths sent whole batches to an
eager forward. This design removes each of those by construction.

## 2. Scope

In scope (v1): draft-model speculation with random sampling (temperature,
top-k, top-p) and greedy; TP = 1; prefix cache disabled when a draft model is
configured; chunked prefill as upstream implements it (prefill batches and
speculative rounds never mix).

Out of scope (v1): n-gram proposals, TP > 1, prefix cache with a draft model,
dynamic `k`, per-request RNG streams, `min_p`, mixing greedy and random
requests in one batch (upstream already rejects this).

## 3. Round anatomy

Upstream `LLMEngine.step()` is `schedule() -> run() -> postprocess()`. A
step is either a prefill batch or a decode batch. Speculation changes only
decode batches:

```
decode batch of B sequences, each with k drafts
  1. draft catch-up   sequences whose draft KV lags len-1 get that one token
                      computed with one draft decode-graph step (only sequences
                      fully accepted last round)
  2. k draft steps    upstream decode CUDA graph on the draft runner;
                      logits [B,V] -> sample -> d[:, j], q[:, j] stay on GPU
  3. verification     one target forward, per sequence [last_token, d1..dk],
                      through the paged varlen-attention path (the path
                      upstream uses for prefill with a cached prefix);
                      CUDA graph keyed by padded batch size; logits [B,k+1,V]
  4. accept/sample    batched tensor ops -> n_accept[B], tail[B]
  5. one D2H          [B, k+2] int64: drafts, n_accept, tail
  6. commit           host bookkeeping in Scheduler.postprocess_speculative
```

Every active sequence proposes exactly `k` tokens every round. No EOS
cropping and no budget shortening before verification: surplus tokens are
dropped at commit. This makes the verification shape `(B, k+1)` with `B` the
only variable, so a graph always matches.

## 4. Components and file changes

### 4.1 `nanovllm/config.py`

New fields:

| Field | Default | Meaning |
|---|---|---|
| `draft_model` | `None` | path of the draft model; enables speculation |
| `num_speculative_tokens` | `0` | `k`; must be `>= 1` iff `draft_model` is set |
| `draft_kv_cache_memory_bytes` | `None` | explicit KV budget of the draft |
| `seed` | `None` | seed of the engine's sampling generator |
| `record_step_timings` | `False` | record CUDA events per step (§9) |

Validation in `__post_init__` (user input, so `ValueError`): with a draft
model, `tensor_parallel_size == 1`, `enable_prefix_cache is False`, both
`kv_cache_memory_bytes` and `draft_kv_cache_memory_bytes` set. Without a
draft model, `num_speculative_tokens == 0`. The explicit budgets are required
because `gpu_memory_utilization` would hand the remaining memory to whichever
model allocates first.

### 4.2 `nanovllm/engine/sequence.py`

```python
@dataclass
class KVState:
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
```

`Sequence` owns `target_kv` and `draft_kv`. The upstream attributes
`num_cached_tokens`, `num_scheduled_tokens` and `block_table` become
properties forwarding to `target_kv`, so upstream call sites are unchanged.
`seq.kv(role)` returns the state for `"target"` or `"draft"`. `kv_state.py`
is deleted; `blocks_for_budget` moves next to its only caller in
`model_runner.py`.

### 4.3 `nanovllm/engine/block_manager.py`

`BlockManager(num_blocks, block_size, enable_prefix_cache, role="target")`
operates on `seq.kv(role)`. `can_append`/`may_append` are replaced by

```python
def can_reserve(self, seq: Sequence, num_tokens: int) -> bool
def reserve(self, seq: Sequence, num_tokens: int) -> None   # grow block_table to cover num_tokens
```

`reserve(seq, len(seq))` allocates exactly when upstream `may_append` did
(`len % block_size == 1`), so `k = 0` keeps upstream's allocation sequence.
Prefix-cache hashing stays target-only and is untouched.

### 4.4 `nanovllm/engine/scheduler.py`

`Scheduler(config, draft_config=None)` holds `block_manager` (target) and,
with a draft, `draft_block_manager` (role `"draft"`, prefix cache off).

- Prefill: a sequence is admitted only if both managers can allocate; both
  allocate; with a draft, `draft_kv.num_scheduled_tokens` is set to the same
  chunk as the target's; `preempt` and finish deallocate both.
- Decode: instead of `may_append`, `reserve(target, len + k)` and
  `reserve(draft, len + k - 1)`; if either cannot, preempt as upstream does.
  With `k = 0` this is `reserve(target, len)`.
- `postprocess(seqs, token_ids, is_prefill)`: upstream, plus advancing
  `draft_kv.num_cached_tokens` by `draft_kv.num_scheduled_tokens` on prefill.
- New `postprocess_speculative(seqs, appended: list[list[int]])`: append the
  tokens one by one, stopping at EOS (unless `ignore_eos`) or `max_tokens`;
  finished sequences deallocate both managers and leave `running`; for the
  rest set `target_kv.num_cached_tokens = len(seq) - 1` and
  `draft_kv.num_cached_tokens = min(len_before - 1 + k, len(seq) - 1)`.

### 4.5 `nanovllm/engine/llm_engine.py`

With a draft model: build `draft_config` from the same kwargs (model path and
KV budget swapped, speculation fields cleared), assert both configs agree on
`max_model_len`, block size, `max_num_seqs`, `max_num_batched_tokens`,
`enforce_eager`, vocabulary size and EOS; construct the two runners through
the existing `DualModelRunner` (shared `DeviceRuntime`, weights of both
loaded before either KV cache is allocated); create `SpeculativeDecoder`
and, unless `enforce_eager`, call its `capture_graphs()` (the draft runner's
own decode graphs are captured by upstream `initialize_cache`).
`add_request` rejects `len(prompt) + max_tokens + k > max_model_len`.

`step()`:

```python
seqs, is_prefill = self.scheduler.schedule()
if is_prefill:
    token_ids = self.model_runner.call("run", seqs, True)
    if self.speculative is not None:
        self.speculative.prefill_draft(seqs)      # same chunk, KV only
    self.scheduler.postprocess(seqs, token_ids, True)
elif self.speculative is not None:
    appended = self.speculative.run_round(seqs)
    self.scheduler.postprocess_speculative(seqs, appended)
else:
    token_ids = self.model_runner.call("run", seqs, False)
    self.scheduler.postprocess(seqs, token_ids, False)
```

After `postprocess`, `step()` records on each sequence the step index at
which it produced its first token and at which it finished; `generate()`
output dicts gain `first_token_step` and `finish_step`. With
`record_step_timings`, a pair of CUDA events is recorded per step and
`engine.step_timings()` returns, after one synchronization, a list of
`(step_index, kind, batch_size, tokens_out, start_ms, duration_ms)` relative
to the first step of the last `generate()` call.

### 4.6 `nanovllm/engine/model_runner.py`

Keep: `DeviceRuntime`, `defer_cache`/`initialize_cache`, explicit KV budget,
`run_model(..., need_logits)`. Add `kv_role` so `prepare_prefill`,
`prepare_decode` and `prepare_block_tables` read `seq.kv(self.kv_role)`.
With `kv_role == "target"` and `k > 0`, allocate one extra KV block beyond
`num_kvcache_blocks`, zero it, and expose it as `pad_block_id` (§5).
Hold the sampling `torch.Generator` (seeded from `Config.seed`).
Delete `run_queries`, `r2_pack_decode`, `enable_verify_graphs`,
`verify_graph_cache`.

### 4.7 New `nanovllm/engine/speculative.py`

```python
class SpeculativeDecoder:
    def __init__(self, target: ModelRunner, draft: ModelRunner, config: Config): ...
    def capture_graphs(self) -> None                       # verification graphs, one per padded B
    def prefill_draft(self, seqs: list[Sequence]) -> None  # draft prefill of the scheduled chunk
    def run_round(self, seqs: list[Sequence]) -> list[list[int]]
    stats: dict   # rounds, histogram of n_accept
```

`run_round` performs §3 steps 1–5 and returns, per sequence,
`drafts[:n_accept] + [tail]`. Per-round host-to-device traffic: last tokens,
positions, context lengths, the draft-step slot mappings `[k, B]`, the
verification slot mapping `[B*(k+1)]`, `cu_seqlens_k`, and both block tables.
Device-to-host traffic: the single `[B, k+2]` tensor.

Draft steps use the upstream decode graph of the draft runner: step `j`
feeds `d[:, j-1]` (or the last committed token for `j = 0`) at position
`len - 1 + j`, context length `len + j`, slot of that position in the draft
block table. Draft prefill goes through the draft runner's prefill path with
`need_logits=False`. Catch-up is, by the round-entry invariant, always exactly
one token (position `len-2`, input `seq[len-2]`), so it runs as one decode-graph
step of the draft for the lagging sequences (`decode_metadata(...,
position_offset=-1)`), after which `draft_kv.num_cached_tokens = len(seq) - 1`.
An earlier version ran it through the eager prefill path; at `B >= 2` almost
every round has a lagging sequence and the eager forward cost ≈ 10 ms per
round, which is why the decode graph is used.

### 4.8 New `nanovllm/layers/spec_sampler.py`

Pure tensor functions, device-agnostic, unit-tested on CPU:

```python
def probs_from_logits(logits, temperatures, top_k, top_p) -> Tensor          # [..., V] fp32
def sample(probs, generator) -> Tensor                                        # exponential race
def accept_random(p, q, drafts, generator) -> tuple[Tensor, Tensor]           # n_accept[B], tail[B]
def accept_greedy(target_logits, drafts) -> tuple[Tensor, Tensor]
```

`layers/sampler.py` (the `k = 0` path) is rewritten on top of the same
`probs_from_logits` and `sample`, so both paths share one truncation and one
sampling implementation. `SamplingParams` gains `top_k: int = 0` (0 = off)
and `top_p: float = 1.0` (1.0 = off); `top_p < 1` requires `top_k > 0`
(§6). The `@torch.compile` decorator is dropped from the sampler: the path
is a handful of ops and one implementation is worth more than the fusion.

### 4.9 Deleted

`engine/random_decode.py`, `random_backend.py`, `random_llm.py`,
`greedy_backend.py`, `verify_graph.py`, `verify_sampling.py`, `ngram.py`,
`kv_state.py`; `layers/random_sampler.py`; `RMSNorm.compile_rms`;
`tests/perf_repair/` and every `test_random_*`, `test_s0_*`, `test_s1_*`,
`test_r2_*`, `test_ngram*`, `test_qk_norm*` file; `tools/*_gpu_gate.py`,
`tools/run_cpu_tests.py`; `docs/plans/`. `docs/GLOSSARY.md` stays because
`docs/PERFORMANCE.md` (kept as the analysis of the previous implementation)
uses its codes. `tests/random_sampling_cases.py` and the exact-fraction
reference in `tests/test_random_reference.py` are kept as the oracle for the
new sampler (§8). CI (`.github/workflows/cpu-tests.yml`) runs `pytest tests/`.

## 5. KV accounting and verification graphs

Invariants at round entry, asserted in `run_round`:

- `target_kv.num_cached_tokens == len(seq) - 1`
- `len(seq) - 2 <= draft_kv.num_cached_tokens <= len(seq) - 1`

Capacity: verification writes positions `len-1 .. len+k-1`, so the target
needs blocks for `len + k` tokens; the draft writes up to `len+k-2`, so it
needs `len + k - 1`. Positions occupied by rejected drafts are not reclaimed;
the next round overwrites them (slots are a function of position).

Verification graphs: one per batch size in the upstream table
`[1, 2, 4, 8] + range(16, max_num_seqs + 1, 16)`, query length fixed to
`k + 1`, sharing the target runner's graph pool. Static buffers: `input_ids`,
`positions`, `slot_mapping`, `cu_seqlens_q`, `cu_seqlens_k`, `block_tables`,
`outputs` (hidden states; `lm_head` runs outside the graph as upstream does).
`max_seqlen_k` is fixed to `max_model_len`; FlashAttention bounds each row by
its own `cu_seqlens_k` entry. Padding rows: `input_ids = 0`, positions
`0..k`, `slot_mapping = -1` (the upstream `store_kvcache_kernel` skips −1, so
nothing is written), `cu_seqlens_k` advancing by `k + 1`, block table row
`[pad_block_id]` whose contents are zero, outputs discarded. There is no
eager fallback for a supported batch size; `enforce_eager` runs the same
code without capture.

## 6. Sampling

Random rows (temperature > 0): `scaled = logits / T`; if `top_k > 0` keep the
top-k values of `scaled` per row (ties at the k-th value are kept); if
`top_p < 1`, take the softmax over those kept values, keep the smallest
prefix of the descending cumulative sum that reaches `top_p`; set every other
logit to −∞; `p = softmax` over the result. This is the HuggingFace order
(temperature, top-k, top-p). `top_p` without `top_k` would need a full
vocabulary sort per row; it is rejected at `SamplingParams` construction.

Draft and target use the request's own parameters. Draft tokens are drawn
from `q` with the exponential race `argmax(q / E)`, `E ~ Exp(1)`, the same
draw upstream uses. Verification (rejection sampling): accept draft `j` iff
`u_j < min(1, p_j(d_j) / q_j(d_j))` with `u ~ U[0,1)`; `n_accept` is the
length of the accepted prefix; the tail is drawn from `normalize(max(p_n -
q_n, 0))` when `n_accept < k` and from `p_k` (the bonus position) otherwise.
`p_j(d_j) = 0` (draft token outside the target's truncated set) rejects.
Greedy rows: `argmax` everywhere; accept iff equal; tail is the target's
`argmax` at the first mismatch or at the bonus position.

RNG contract: one `torch.Generator` per engine (owned by the target runner
and used by both sampling paths), seeded from `Config.seed`.
A run is reproducible given the seed and the same batch composition; unlike
the previous implementation, changing which requests share a batch changes
the draws. Documented in the README.

## 7. Error handling

Public entry points (`Config`, `LLMEngine.__init__`, `add_request`,
`SamplingParams`) validate user input with `ValueError`. Internal functions
use `assert` for invariants only: the two KV invariants, the D2H payload
shape, graph capture preconditions. No `try/except` around the round, no
validity masks, no finite/normalization checks on probability tensors: a
non-finite logit surfaces as a wrong token and is caught by the gates of §8.

## 8. Testing

CPU (`pytest`, torch CPU, no model weights):

- `spec_sampler`: frequency test of `accept_random` against the exact
  rational reference (`tests/random_sampling_cases.py`), seeded, 100k draws
  per case; `accept_greedy` for every first-mismatch position and `k = 0`;
  `probs_from_logits` truncation cases including ties and rows with
  truncation off.
- `BlockManager.reserve`: allocation count and block-table growth; `k = 0`
  reproduces upstream `may_append` block by block.
- `Scheduler.postprocess_speculative`: both invariants after every
  `n_accept`, EOS truncation, `max_tokens`, deallocation of both managers,
  step-index bookkeeping.
- `SpeculativeDecoder.run_round` with fake runners (canned logits, contexts
  recorded through `get_context()`): catch-up decisions, draft-step
  positions and slots, verification `cu_seqlens`, slot mapping and padding
  rows, returned tokens for greedy and seeded random.

GPU gates, in order, before any performance number:

- G1 `tools/gate_verify_logits.py`: for a prompt, decode `k+1` tokens
  step by step and record logits; run verification on the same prefix with
  those tokens as drafts; report max |Δlogit| per position and top-1
  agreement. Establishes the kernel-level noise floor.
- G2 `tools/gate_sampler_distribution.py`: the §8 frequency test on GPU
  with real target/draft logits.
- G3 `tools/gate_greedy_equivalence.py`: greedy `k = 0` vs `k = 4` on the
  A1 panel; match rate, and for each divergence the top-2 logit margin at
  that position, interpreted against G1.

## 9. Benchmark protocol

`bench_spec.py` runs `LLM.generate` on the A1 + A2 panels (48 prompts, from
the `expanded-k3-results` release), `max_tokens = 512`, natural EOS.

Matrix: sampling lines × `max_num_seqs ∈ {1, 2, 4, 8, 16}` × `{k = 0, k}`:

| Line | Parameters | Why |
|---|---|---|
| Qwen3 recommended | T = 0.7, top_p = 0.8, top_k = 20, k = 3 | the deployment setting (Qwen3-8B model card, non-thinking mode) |
| Flat | T = 1.0, no truncation, k = 3 | comparable with the RTX 5090 results |
| Greedy | T = 0, k = 4 | reference only; the model card advises against greedy (repetition), and repetition inflates acceptance |

Timing: `record_step_timings=True` records a pair of CUDA events per step,
read once after `generate()`. Each configuration runs one warm-up
`generate()` (captures graphs, excluded) followed by the measured call; the
measured call contains ≥ 100 rounds by construction (48 prompts × up to 512
tokens), which satisfies the ≥ 25 warm-up / ≥ 100 measured rule.

Reported per configuration:

| Metric | Definition | Form |
|---|---|---|
| `Q`, `R` | completion tokens ÷ `generate()` time; `R = Q_spec / Q_k0` | one number |
| TTFT | first-token time from `generate()` start (queueing included; stated) | median, p95 over requests |
| TPOT | (finish − first token) ÷ (tokens − 1) | median, p95 over requests |
| E2E | finish time | median, p95 over requests |
| step / round duration | by (kind, batch size); the upper bound on the gap between token deliveries | median, p95 |
| acceptance | per-position acceptance rate, tokens per round | one row |
| greedy validity | share of requests hitting the cap; share whose last 64 tokens contain a repeated 8-gram | one row |

Every result file carries a manifest: GPU model, driver, torch, flash-attn,
triton, commit, config, seed. An A/B pair (`k = 0` vs `k`) always runs on the
same GPU model; different pairs may use different models when the queue
requires it.

## 10. Risks

- FlashAttention varlen with a block table under graph capture on sm_120:
  exercised by the previous implementation on the RTX 5090; low risk.
- Padding rows attending to a zero block: finite by construction; asserted
  once in G1 by comparing padded and unpadded verification of the same batch.
- Catch-up runs through the draft's decode graph (≈ 1.8–2.3 ms per round when
  needed); the eager variant was measured at ≈ 10 ms and rejected.
- Memory for `q` and `p` at `B = 16, k = 4, V = 151936`: ≈ 90 MB fp32; fine.
