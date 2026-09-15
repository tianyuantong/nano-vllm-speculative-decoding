# Performance analysis: why speculative decoding broke even

This document consolidates the measurements from PRs #3–#7 and adds the
decomposition that explains them. Every number below is either quoted from a
PR description or computed from the archived result files with
[`tools/decompose_timings.py`](../tools/decompose_timings.py); nothing was
re-measured for this write-up.

**Short version.** Randomized speculative decoding (Qwen3-0.6B draft,
Qwen3-8B target, k = 3) produces 2.1–2.4 tokens per round, but a round costs
2.1–2.4 ordinary decode steps. The two numbers sit on top of each other, so the
sign of the result flips with the workload (+4.5 % on 8 requests, −2.7 % on 48).
The round is expensive for a structural reason: the sampler runs
**per request** in Python, at ≈225 small tensor ops per request per round, and
two host synchronizations per round keep that CPU work from overlapping the
GPU. At batch size 4 the per-request cost (≈11 ms) equals one 8B forward
(≈11 ms). PR #7 removes part of it and confirms the diagnosis; the remaining
fixed cost of a round (≈20 ms) is untouched by any PR.

---

## 1. Setup

| | |
|---|---|
| Hardware | 1× NVIDIA GeForce RTX 5090, BF16, TP = 1 |
| Models | target Qwen3-8B, draft Qwen3-0.6B (same tokenizer) |
| Software | PyTorch 2.9.1+cu130, FlashAttention 2.8.3, Triton 3.5.1 |
| Sampling | temperature 1.0 (random) or 0 (greedy); no top-k/top-p; natural EOS; ≤512 new tokens |
| Engine | `RandomLLM` — an offline drain-batch controller (`max_num_seqs = 4`, no refill), prefix cache off, explicit KV budgets (target 4 GiB, draft 1.5 GiB) |
| Timing | wall time of a complete `generate()` call after warm-up, two repetitions per configuration; loading and warm-up excluded |
| Workloads | development set (8 requests in 2 groups); panels A1 and A2 (24 requests each, 6 groups of 4); seeds 17011 / 110017 / 130031 / 170041 |

The baseline in every comparison is `RandomLLM(k=0)` — ordinary decoding
through the *same* controller and sampler, not upstream nano-vLLM's
`LLM.generate`. This keeps A/B comparisons fair but means absolute numbers
should not be compared with upstream, which has continuous batching and a
batched sampler (see §7).

## 2. Headline results as reported in the PRs

| PR | Change | Workload | Result vs. ordinary (`k=0`) |
|---|---|---|---|
| #3 | offline randomized speculative decoding (dual model, n-gram) | — | implementation; correctness checks only |
| #4 | VERIFY CUDA graphs, KV-only draft forwards, fewer temporaries | 8 requests | dual-model 21.30 s → **11.88 s** (−44 %); ordinary 11.20 s — *still faster* |
| #5 | opt-in sampling fast paths; k selection | 8 requests | k=3: 10.73 s vs 11.23 s (**−4.5 % time, +10.1 % tokens/s**); k=2 and k=4 slower than ordinary |
| #6 | expanded retest of the k=3 candidate | 48 requests × 3 seeds | A1 **+2.65 %** time, A2 **+2.74 %** time — the 8-request advantage did not generalize |
| #7 | greedy speculation; CUDA graph for the post-p/q verification block | A1 + one stress group | greedy k=4: **+12.7 %** tokens/s on A1, −0.1 % on the stress group; random k=3 + graph: **+5.4 %** on A1, **−8.6 %** on the stress group |

Two things are true at once: the implementation got 2.4× faster between #3
and #7, and it never beat ordinary decoding by more than noise except in the
greedy configuration. Section 4 explains why with one table.

## 3. Method: decomposing a `generate()` call

Every archived result carries, for each timed call, the VERIFY-graph replay
counters **bucketed by batch size** (`replay_B4_Q4`, `replay_B3_Q4`, …) and the
number of rounds that fell back to the eager forward. Because these counts are
exact, a least-squares fit of the call's wall time against them yields the cost
of one round at each batch size — a decomposition of measured time, not a
model of it. For ordinary decoding the number of steps at each batch occupancy
follows from the per-request output lengths (a drain batch shrinks as requests
finish).

```
python tools/decompose_timings.py <release>/evidence/clean   # expanded-k3-results
python tools/decompose_timings.py <release>/execution/clean  # greedy-verify-sampling-results
```

Fit error is 0.2–1.6 % of wall time across 230 timed calls.

## 4. Results of the decomposition

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/round-cost-dark.svg">
  <img alt="Per-round cost versus requests in the batch: a speculative round rises from 21.9 to 30.3 ms between batch 1 and 4 (+2.80 ms per request), the PR #7 variant from 21.0 to 26.6 ms (+1.93), an ordinary decode step from 11.5 to 12.8 ms (+0.48)." src="assets/round-cost-light.svg">
</picture>

| Configuration | Fixed cost per round | Per extra request | Round at batch 4 | Eager fallback round | Tokens / round | Timed calls |
|---|---:|---:|---:|---:|---:|---:|
| Random S1, k=3 — PR #5 engine | 19.5 ms | **2.80 ms** | 30.3 ms | 82 ms | 2.27 (1.89–3.85) | 72 |
| Random S1, k=3 — same engine, PR #7 rerun ("old") | 19.8 ms | **2.91 ms** | 30.7 ms | 74 ms | 2.11 | 14 |
| Random S1, k=3 + post-p/q graph — PR #7 | 19.7 ms | **1.93 ms** | 26.6 ms | 69 ms | 2.11 | 14 |
| Greedy S1, k=4 — PR #7 | 21.0 ms | **1.34 ms** | 24.6 ms | 68 ms | 2.36 | 14 |
| **Ordinary decode step (any mode)** | **11.0 ms** | **0.33–0.48 ms** | **12.5–12.8 ms** | — | — | 100 |

Three observations carry the whole analysis:

1. **The per-request slope is the variable that moved.** 2.91 → 1.93 when PR #7
   captured the post-p/q block; → 1.34 in greedy mode, where the probability
   machinery disappears. An ordinary step costs 0.33–0.48 ms per extra request.
   Speculative decoding paid **6× more per request** than ordinary decoding.
2. **The fixed cost of a round (≈20 ms) is identical in all three
   configurations.** Nothing in PRs #4–#7 touched it. It decomposes as the
   target VERIFY forward (≈11 ms, equal to an ordinary step's intercept),
   three draft steps plus catch-up (≈5 ms; greedy with k=4 costs exactly one
   draft step, 1.3 ms, more), and ≈4 ms of host synchronization, pinned-memory
   allocation and Python control flow.
3. **Tokens per round is 2.1–2.4** with k = 3–4, i.e. only 1.1–1.4 of the
   drafts are accepted per round (per-token acceptance ≈ 0.55–0.6). Two of three
   draft forwards and two thirds of the VERIFY positions are wasted.

### Where the per-request cost comes from

The sampler is written per request because each request owns five
`torch.Generator` streams (`RequestRNG`, `random_backend.py`) so that its
output is reproducible regardless of what else is in the batch. A
`torch.exponential_` with a per-request generator cannot be batched across
requests, so `propose_device`, `sample_batch` and `verify` all loop
`for r in requests:` and issue every op on a `[1, V]` slice. Counting the ops on
that path gives ≈120 per request in `verify` (validation, FP64 masses, invalid
masks, placeholder rows, index selects) and ≈35 per draft step, ≈225 per
request per round. At PyTorch's eager dispatch cost of ~10–20 µs per op that is
2.3–4.5 ms — the fitted 2.8 ms sits inside that range, and the ordinary path's
≈35 ops × 12.6 µs ≈ 0.44 ms matches its fitted 0.48 ms. GPU time for these
kernels is under 1 ms per request; the cost is CPU dispatch, and it is not
hidden behind GPU execution because the round synchronizes with the host twice
(`materialize_proposals`, `_materialize_checked`).

## 5. The break-even inequality

With t = one target forward, d = one draft forward, F = fixed per-round
overhead, s / s′ = per-request cost in the speculative / ordinary path and
B = requests in the batch:

```
speculative round = t + (k+1)·d + F + B·s        → 1 + accepted tokens per request
ordinary step     = t + B·s′                     → 1 token per request

speculative wins  ⇔  tokens/round  >  (t + (k+1)·d + F + B·s) / (t + B·s′)
```

At batch 4 with the PR #5 engine the right-hand side is 30.3 / 12.8 = **2.37**;
the measured left-hand side is **2.27**. That 4 % gap is the −2.7 % of the
expanded retest. The 8-request "+4.5 %" of PR #5 and the group-level swings in
PR #6 (per-group speed-up between 0.79× and 1.91×) are what a result balanced
on the break-even line looks like when the acceptance rate and batch occupancy
of individual groups vary.

The PR #7 stress-group result (−8.6 %) is the same inequality, not noise: that
group had the lowest tokens/round of any run (1.92) and spent 209 of 266
rounds at full batch, where a round is most expensive (27.1 ms ⇒ ratio 2.11 >
1.92). Greedy on the same group had 2.20 tokens/round against a ratio of 2.07
and came out +6 %.

## 6. Where a round's 30 ms goes (batch 4, PR #5 engine)

| Component | ms | Share | Reducible? |
|---|---:|---:|---|
| Target VERIFY forward (CUDA graph) | ≈11 | 36 % | No — 8B BF16 weights are 16.4 GB; at 1.79 TB/s the floor is 9.2 ms, this runs at 83 % of it |
| 3 draft steps + catch-up (0.6B, CUDA graph) | ≈5 | 17 % | Partly — `compute_logits` and the per-step sampling run outside the graph; weights alone need 0.67 ms per step |
| Host syncs, pinned allocations, Python control flow | ≈4 | 13 % | Yes — one D2H per round, preallocated buffers |
| Per-request sampling, 4 × 2.80 | ≈11 | 37 % | Yes — batch the sampler across requests |
| **Round** | **≈30** | | |
| Eager fallback rounds (5–8 per call × 68–84 ms) | | ≈9 % of call time | Yes — pad mixed query lengths to a captured shape |

Eager fallbacks happen because `exact_verify_key` (`verify_graph.py`) only
matches batches whose requests have identical query lengths, while
`budgets = min(k, remaining − 1)` shortens the draft budget of any request
within k tokens of its cap. One such request sends the whole batch to the
36-layer eager forward.

## 7. Findings that only the source shows

1. **Two host synchronizations per round, not one.** The design note says the
   host reads once at the commit boundary (`_materialize_checked`), but with
   `gpu_draft_tokens=True` the draft ids must also come back before VERIFY
   (`materialize_proposals`) because EOS cropping and the target's `input_ids`
   are built on the host. S1's measured advantage over S0 (1.7 %) is the
   difference between k small copies and one batched copy, not the removal of a
   copy.
2. **Per-request RNG ⇒ per-request Python loop** (§4). This is a deliberate
   reproducibility contract, and it is the single largest reducible cost.
3. **The baseline is the project's own k=0 path**, with the same per-request
   sampler and a drain-batch scheduler (mean occupancy 2.9 of 4). Upstream
   nano-vLLM with continuous batching would raise occupancy to ≈4 of 4 and use a
   single batched sampler; on this workload that alone is worth an estimated
   1.5–1.8× for *both* modes.
4. **The correction draw is computed even when no draft was rejected** and then
   discarded by `torch.where`; avoiding it needs a host read of `count`, which
   would be a third synchronization.
5. **Greedy speculation matched greedy ordinary decoding on only 6 of 28
   requests.** The first divergence sits at output index 1 / 100 / 258
   (min / median / max), never at index 0 (both paths sample the first token the
   same way), and only 2 of 22 divergences fall inside the first VERIFY round.
   That distribution is consistent with bf16 rounding: `compute_logits` emits
   bf16, whose resolution is 0.125 for logits between 16 and 32, so top-2 ties
   are common, and the decode kernel (`flash_attn_with_kvcache`) and the VERIFY
   kernel (`flash_attn_varlen_func`) accumulate in different orders. It is not
   consistent with a KV or position bug, which would diverge most requests in
   the first round. The decisive check — top-2 logit margins at the 22
   divergence points, or an fp32 `lm_head` rerun — was not run and remains open.

## 8. What would close the gap (not implemented)

Estimated from the decomposition; ordered by change size.

| Change | Mechanism | Estimated effect |
|---|---|---|
| Pad mixed query lengths to a captured shape | no eager fallback rounds | −9 % call time |
| Batch the random sampler across requests (one global generator, `[B, k+1, V]` ops, as `greedy_backend._argmax` already does for argmax) | per-request slope 1.93 → ≈0.5 ms | −6 ms per round at batch 4 |
| One D2H per round, preallocated pinned buffers, no per-round `Query` construction | F ≈4 → ≈1 ms | −3 ms per round |
| Move draft `compute_logits` into the draft graph | (k+1)·d ≈5 → ≈3.5 ms | −1.5 ms per round |
| Replace the drain-batch controller with the upstream scheduler | continuous batching for both modes | ≈1.5–1.8× for both, ratio unchanged |

With all of the first four, a batch-4 round drops to ≈16–17 ms (ratio ≈1.3),
and the measured 2.1–2.4 tokens/round would translate into **≈1.6–1.8×** over
ordinary decoding. Batching the sampler alone is worth ≈1.1–1.25×. Beyond that
the limit is the draft model: per-token acceptance of ≈0.55–0.6 is what a 0.6B
draft achieves against an 8B target at temperature 1.

## 9. Reproduce

1. Download the evidence archives from the releases
   [`expanded-k3-results`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/expanded-k3-results)
   (144 timed calls) and
   [`greedy-verify-sampling-results`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/greedy-verify-sampling-results)
   (86 timed calls) and unzip them.
2. `python tools/decompose_timings.py <expanded>/evidence/clean <greedy>/execution/clean`
   (needs only numpy) prints the tables in §4.
3. GPU measurements: `tools/perf_repair_gpu_gate.py`, `tools/greedy_gpu_gate.py`
   and `tools/verify_sampling_gpu_gate.py` reproduce the timed calls on a CUDA
   machine with the two models; see [DECODING.md](DECODING.md) for the exact
   configuration.

## 10. Limitations

- One GPU model, one model pair, one temperature; no serving-latency (TTFT /
  per-token) measurements — all timings are offline drain-batch throughput.
- Random-mode A/B pairs generate different text, so per-group speed-ups carry
  trajectory noise; the decomposition avoids this by normalizing to rounds and
  steps, the headline speed-ups do not.
- The break-down of the fixed 20 ms into t, d and F uses the ordinary step's
  intercept and the greedy/random intercept difference; it was not profiled per
  kernel.
- Output quality of speculative vs. ordinary decoding was not evaluated;
  strict finite-precision equivalence to ordinary decoding was never claimed.
