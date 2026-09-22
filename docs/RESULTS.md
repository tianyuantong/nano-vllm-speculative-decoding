# Results: batched speculative decoding vs ordinary decoding

Measured 2026-09-21 on one NVIDIA RTX PRO 6000 Blackwell **Server Edition** (GB202, 96 GB,
the same silicon and memory bandwidth as the RTX 5090 of the earlier experiments), driver
580.173.02, torch 2.9.1+cu130, FlashAttention 2.8.3, Triton 3.5.1, Python 3.12.14,
source `9a6afc3+src.ed54ec4112d3` (base commit + sha256 of the synced sources). Target
Qwen3-8B, draft Qwen3-0.6B, both BF16, TP = 1. Every A/B pair (`k = 0` vs `k`) of one
line and batch size ran in the same Slurm job on the same GPU; all 40 runs are one job
(`spec-matrix-164484`). Raw result files: `benchmarks/results/pro6000-server-9a6afc3/`
(not in git). The complete evidence package — the 40 result files, the gate, control and probe
reports, the input panel, and a snapshot of the measured sources with SHA-256 sums — is the
release asset `results-batched-v1`:
<https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-batched-v1>.
`tools/report_matrix.py` regenerates every table below from the result files.

## Setup

- Workload: the 48-prompt panel of `docs/PERFORMANCE.md` (A1 groups 2, 6, 7, 8, 10, 11 and
  A2 groups 0–5; Qwen3 chat template, non-thinking mode; prompts 980–2262 tokens),
  `max_tokens = 512`, natural EOS. Fetched with `tools/fetch_panels.py`.
- Engine: upstream nano-vLLM scheduler (continuous batching), `enable_prefix_cache=False`
  in every run, `kv_cache_memory_bytes = 10 GiB`, `draft_kv_cache_memory_bytes = 6 GiB`,
  `max_model_len = 4096`, `seed = 17011`.
- Baseline: the same engine's ordinary decoding path (`k = 0`): the upstream scheduler and decode
  CUDA graph, and the rewritten sampler (which adds top-k / top-p) shared with the speculative path.
  It is not the untouched upstream release.
- Lines: **rec** = Qwen3's recommended non-thinking sampling (T 0.7, top-p 0.8, top-k 20);
  **flat** = T 1.0 without truncation (comparable with the RTX 5090 numbers of
  `docs/PERFORMANCE.md`); **greedy** = T 0 (the model card advises against it; see
  validity below).
- Timing: one pair of CUDA events per engine step and one event per phase of every round,
  read once after `generate()`; one warm-up `generate()` on 4 prompts excluded. Throughput
  `Q` = completion tokens ÷ GPU timeline of the measured `generate()`; `R = Q(k) / Q(k=0)`.
  TTFT/TPOT/E2E per request are reconstructed from the step a request got its first token
  and the step it finished; p50/p95 are nearest-rank percentiles over the 48 requests.

## Headline

| max_num_seqs B | rec k=3 | flat k=3 | greedy k=4 |
|---:|---:|---:|---:|
| 1 | **1.50** | 1.45 | 1.53 |
| 2 | **1.37** | 1.34 | 1.37 |
| 4 | **1.24** | 1.25 | 1.26 |
| 8 | **1.17** | 1.20 | 1.14 |
| 16 | **1.04** | 1.03 | 1.06 |

`R ≥ 1.15` holds for `B ≤ 8` on the recommended and the T = 1.0 lines, and for `B ≤ 4` on the
greedy line (greedy `B = 8` is 1.135); at `B = 16` speculation is within ±6 % of ordinary
decoding. Each configuration is one seed and one timed call. The ratio of raw `generate()` wall
times (the earlier project's primary metric) for the recommended line is 1.51 / 1.38 / 1.30 /
1.19 / 1.07 at `B = 1 … 16`. Speculative calls emit fewer completion tokens in all five pairs:
14949 / 14716 / 14443 / 14708 / 14841, versus 15023 / 14817 / 15165 / 14995 / 15166
for ordinary decoding. Throughput ratio equals output-count ratio × GPU-time ratio;
the wall timer has a slightly different boundary. Across `k ∈ {2, 3, 4}` on the rec line, `k = 3` is best for `B ≤ 4`,
`k = 2` and `k = 3` tie at `B = 8`, and `k = 2` is best at `B = 16`; `k = 4` is never best.
The previous implementation (`docs/PERFORMANCE.md`) measured −2.7 % … +5.4 % at `B ≤ 4`
on the same models and panel.

TPOT over the 48-request offline batch (time per output token, p50 over requests; it includes
the scheduling waits a request sees after its first token, so it is not a pure inter-token
interval) falls from 362 → 243 ms at `B = 1` and from 52 → 48 ms at `B = 8` (rec, k = 3); p95
from 900 → 584 ms and 119 → 97 ms. Recorded active speculative rounds take 19–31 ms,
against 12–14 ms for one ordinary decode step.

## Throughput, latency and the break-even ratio

`tokens/row/round` is the mean number of tokens one request commits per round
(1 + accepted drafts). `round p50 / step p50` are the median durations at the full batch
size; their ratio is the cost side of the break-even inequality.

| line | B | k | tok/s k=0 | tok/s k | R | tokens/row/round | round p50 / step p50 | ratio | TPOT p50 k=0 → k (ms) | TPOT p95 k=0 → k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 1 | 3 | 82.7 | 120.3 | **1.454** | 2.32 | 18.3 / 11.9 | 1.54 | 349.1 → 231.4 | 818.6 → 484.8 |
| flat | 2 | 3 | 159.1 | 213.6 | **1.342** | 2.31 | 20.1 / 12.2 | 1.64 | 179.9 → 136.4 | 437.4 → 312.2 |
| flat | 4 | 3 | 305.4 | 382.9 | **1.254** | 2.38 | 22.2 / 12.1 | 1.83 | 90.9 → 72.7 | 213.8 → 157.6 |
| flat | 8 | 3 | 503.2 | 605.3 | **1.203** | 2.28 | 25.0 / 12.6 | 1.98 | 48.9 → 44.5 | 103.9 → 93.5 |
| flat | 16 | 3 | 805.0 | 831.1 | **1.032** | 2.27 | 30.0 / 13.9 | 2.15 | 29.8 → 28.5 | 66.5 → 59.9 |
| greedy | 1 | 4 | 83.3 | 127.8 | **1.534** | 2.67 | 19.9 / 11.8 | 1.68 | 345.8 → 220.7 | 806.4 → 526.5 |
| greedy | 2 | 4 | 159.6 | 218.1 | **1.366** | 2.59 | 21.6 / 12.1 | 1.78 | 178.8 → 146.0 | 442.7 → 301.9 |
| greedy | 4 | 4 | 305.8 | 386.2 | **1.263** | 2.64 | 24.3 / 12.0 | 2.02 | 97.1 → 72.3 | 217.9 → 151.3 |
| greedy | 8 | 4 | 549.1 | 623.3 | **1.135** | 2.68 | 27.0 / 12.5 | 2.16 | 51.7 → 45.0 | 107.3 → 96.2 |
| greedy | 16 | 4 | 803.7 | 850.9 | **1.059** | 2.62 | 33.2 / 13.8 | 2.41 | 31.2 → 28.6 | 58.0 → 52.2 |
| rec | 1 | 2 | 81.4 | 113.2 | **1.390** | 2.07 | 17.3 / 12.1 | 1.43 | 362.2 → 261.3 | 899.6 → 578.2 |
| rec | 1 | 3 | 81.4 | 122.0 | **1.499** | 2.45 | 19.0 / 12.1 | 1.57 | 362.2 → 242.6 | 899.6 → 584.4 |
| rec | 1 | 4 | 81.4 | 114.6 | **1.408** | 2.53 | 21.2 / 12.1 | 1.75 | 362.2 → 260.1 | 899.6 → 592.7 |
| rec | 2 | 2 | 156.3 | 205.1 | **1.313** | 2.11 | 19.7 / 12.4 | 1.58 | 190.5 → 152.2 | 432.0 → 314.3 |
| rec | 2 | 3 | 156.3 | 214.1 | **1.370** | 2.44 | 21.9 / 12.4 | 1.76 | 190.5 → 138.3 | 432.0 → 337.9 |
| rec | 2 | 4 | 156.3 | 205.3 | **1.314** | 2.52 | 22.9 / 12.4 | 1.85 | 190.5 → 136.1 | 432.0 → 315.8 |
| rec | 4 | 2 | 301.5 | 364.0 | **1.207** | 2.10 | 20.6 / 12.3 | 1.67 | 93.5 → 84.5 | 205.2 → 165.4 |
| rec | 4 | 3 | 301.5 | 374.2 | **1.241** | 2.43 | 23.0 / 12.3 | 1.87 | 93.5 → 83.7 | 205.2 → 184.3 |
| rec | 4 | 4 | 301.5 | 346.0 | **1.148** | 2.49 | 25.8 / 12.3 | 2.09 | 93.5 → 93.1 | 205.2 → 169.4 |
| rec | 8 | 2 | 507.7 | 594.7 | **1.172** | 2.08 | 23.1 / 12.8 | 1.80 | 52.2 → 45.0 | 119.1 → 106.0 |
| rec | 8 | 3 | 507.7 | 592.8 | **1.168** | 2.36 | 25.9 / 12.8 | 2.02 | 52.2 → 47.6 | 119.1 → 96.7 |
| rec | 8 | 4 | 507.7 | 581.1 | **1.145** | 2.55 | 28.7 / 12.8 | 2.23 | 52.2 → 47.3 | 119.1 → 91.4 |
| rec | 16 | 2 | 793.1 | 835.2 | **1.053** | 2.12 | 27.0 / 14.1 | 1.91 | 31.7 → 28.8 | 61.1 → 54.5 |
| rec | 16 | 3 | 793.1 | 827.0 | **1.043** | 2.38 | 31.1 / 14.1 | 2.20 | 31.7 → 29.2 | 61.1 → 53.7 |
| rec | 16 | 4 | 793.1 | 784.5 | **0.989** | 2.53 | 34.9 / 14.1 | 2.47 | 31.7 → 31.8 | 61.1 → 50.8 |

At a fixed full batch, `tokens/row/round > round/step` is a rough break-even check.
The estimate `tokens/row/round ÷ ratio` differs from measured `R` by −0.053 to +0.105
(estimate minus measurement) across the 25 pairs. It uses full-batch step medians and
omits prefill and the drain at the end of a call, where rounds run below the full batch size.
The sign of `R − 1` agrees with the estimate in 24 of 25 pairs; the exception is
rec `B = 16, k = 4` (measured 0.989, estimated 1.026).

## Where a round's time goes

Median milliseconds per phase (CUDA events inside `SpeculativeDecoder.run_round`). Medians
do not add: the sum of the phase medians differs from the round median above by up to 1.1 ms
(rec, B = 16: 30.0 vs 31.1). `catch_up` is one draft decode-graph step for the sequences a
fully accepted round left one KV entry short; its p50 is ≈ 0 when fewer than half the rounds
need it. "accept rate by position" is the share of rows whose first *n* drafts were all
accepted (draft-prefix survival), not a per-position conditional rate.

| line | B | k | catch_up | propose | verify | accept | copy | sum | draft-prefix survival by position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| flat | 1 | 3 | 0.01 | 5.94 | 12.06 | 0.13 | 0.02 | 18.2 | 0.63, 0.41, 0.28 |
| flat | 2 | 3 | 0.01 | 6.34 | 13.18 | 0.14 | 0.02 | 19.7 | 0.63, 0.41, 0.27 |
| flat | 4 | 3 | 1.79 | 6.81 | 13.38 | 0.15 | 0.02 | 22.2 | 0.64, 0.43, 0.31 |
| flat | 8 | 3 | 1.93 | 7.88 | 14.99 | 0.17 | 0.02 | 25.0 | 0.61, 0.40, 0.27 |
| flat | 16 | 3 | 2.03 | 10.17 | 16.54 | 0.22 | 0.02 | 29.0 | 0.61, 0.40, 0.26 |
| greedy | 1 | 4 | 0.01 | 7.61 | 12.07 | 0.03 | 0.02 | 19.7 | 0.66, 0.45, 0.32, 0.24 |
| greedy | 2 | 4 | 0.01 | 8.11 | 13.16 | 0.03 | 0.02 | 21.3 | 0.64, 0.43, 0.30, 0.22 |
| greedy | 4 | 4 | 1.74 | 8.79 | 13.88 | 0.03 | 0.02 | 24.5 | 0.66, 0.44, 0.31, 0.23 |
| greedy | 8 | 4 | 1.86 | 10.05 | 14.93 | 0.04 | 0.02 | 26.9 | 0.67, 0.45, 0.32, 0.24 |
| greedy | 16 | 4 | 1.93 | 13.23 | 16.90 | 0.05 | 0.02 | 32.1 | 0.65, 0.44, 0.31, 0.22 |
| rec | 1 | 2 | 0.01 | 4.33 | 12.35 | 0.31 | 0.02 | 17.0 | 0.64, 0.43 |
| rec | 1 | 3 | 0.01 | 6.42 | 12.06 | 0.31 | 0.02 | 18.8 | 0.67, 0.45, 0.33 |
| rec | 1 | 4 | 0.01 | 8.58 | 12.13 | 0.33 | 0.02 | 21.1 | 0.63, 0.41, 0.29, 0.20 |
| rec | 2 | 2 | 1.78 | 4.52 | 13.15 | 0.34 | 0.02 | 19.8 | 0.66, 0.45 |
| rec | 2 | 3 | 1.78 | 6.84 | 13.19 | 0.34 | 0.03 | 22.2 | 0.67, 0.45, 0.32 |
| rec | 2 | 4 | 0.01 | 9.15 | 13.14 | 0.34 | 0.02 | 22.7 | 0.63, 0.41, 0.28, 0.20 |
| rec | 4 | 2 | 1.93 | 4.91 | 13.35 | 0.34 | 0.02 | 20.6 | 0.65, 0.44 |
| rec | 4 | 3 | 1.81 | 7.39 | 13.38 | 0.34 | 0.03 | 23.0 | 0.66, 0.45, 0.32 |
| rec | 4 | 4 | 1.78 | 9.89 | 13.94 | 0.35 | 0.02 | 26.0 | 0.62, 0.40, 0.27, 0.19 |
| rec | 8 | 2 | 2.05 | 5.64 | 14.91 | 0.36 | 0.03 | 23.0 | 0.64, 0.44 |
| rec | 8 | 3 | 1.97 | 8.49 | 14.99 | 0.38 | 0.03 | 25.9 | 0.64, 0.43, 0.29 |
| rec | 8 | 4 | 1.83 | 11.33 | 15.01 | 0.39 | 0.03 | 28.6 | 0.64, 0.42, 0.28, 0.20 |
| rec | 16 | 2 | 2.31 | 7.20 | 16.21 | 0.42 | 0.03 | 26.2 | 0.66, 0.46 |
| rec | 16 | 3 | 2.11 | 10.85 | 16.56 | 0.47 | 0.03 | 30.0 | 0.65, 0.44, 0.30 |
| rec | 16 | 4 | 1.96 | 14.46 | 16.93 | 0.49 | 0.03 | 33.9 | 0.63, 0.42, 0.28, 0.20 |

Three measurements carry the analysis; the causes named for the first two are hypotheses
consistent with the numbers, not separately profiled:

1. **One draft step costs 2.1–3.6 ms** (propose ÷ k), against 0.7 ms for reading its 1.2 GB
   of weights at 1.79 TB/s. A 28-layer model issues ≈ 300 kernels per step, so per-kernel
   launch overhead inside the graph is the likely explanation; a kernel-level profile would
   confirm it. With k = 3 the draft side (catch-up + propose) is 6–13 ms of a 19–31 ms round —
   the term `(k+1)·d` of the inequality, and the reason larger k stops paying.
2. **Verification costs 0–2.8 ms more than a decode step** (12.1 vs 11.9 ms at B = 1,
   16.6 vs 14.1 ms at B = 16) for the same weight read. The two paths differ in the attention
   kernel (`flash_attn_varlen_func` over k+1 queries without split-KV, against
   `flash_attn_with_kvcache`) and in the number of query rows; which part of the gap each
   accounts for was not measured.
3. **Draft-prefix survival is ≈ 0.64 / 0.43 / 0.29 on all three sampling lines**, with greedy
   slightly higher (0.66 / 0.45 / 0.32 / 0.23). Temperature and truncation change p and q, so
   this is an observation about these three configurations, not a general independence.

Everything the previous implementation paid for is gone: sampling is 0.03–0.5 ms per
round for the whole batch (was ≈ 2.8 ms per request), the host copy is 0.02–0.03 ms
(was two synchronizations), and there are no eager fallbacks (the round p95 tracks the
p50 within 1–2 ms).

## Correctness gates (job `spec-gates-164454`, control `spec-control-164467`)

- **Smoke**: the speculative engine and the `k = 0` engine both generate coherent text on
  three prompts; the speculative run finished 48 tokens × 3 prompts in 16 engine steps
  (1 prefill + 15 rounds) against 48 steps.
- **G1 — verification logits vs step-by-step decode logits** on the same prefix, k = 3:
  max |Δ| per position 0.25 / 0.25 / 0.34 / 0.28 (bf16 resolution at logit magnitude
  16–32 is 0.125), argmax identical at every position; a batch padded from 3 to 4 rows
  changed the real rows by 0.0.
- **G2 — sampler distribution**: eight exact reference (p, q) pairs, 200 000 draws each on
  GPU: total variation between emitted and target distributions ≤ 0.0031 (max 0.00302). On a real
  (p, q) pair from the target/draft at T 0.7 / top-p 0.8 / top-k 20 the analytic
  acceptance rate Σ min(p, q) and the observed rate agreed (both 1.0: the distributions
  coincided after truncation at that position).
- **G3 — greedy equivalence** (a measurement, not a pass/fail gate): greedy k = 4 vs greedy
  k = 0 on the 48 prompts: 7 exact matches, 41 divergences; 35 of the 41 divergences sit at a
  top-2 logit margin below the G1 noise floor (median margin 0.000, i.e. exact bf16 ties).
  **Control**: greedy k = 0 at B = 1 and at B = 8 against greedy k = 0 at B = 4 — 7 / 41 / 35
  and 7 / 41 / 34. The baseline itself is batch-size sensitive to the same degree; the
  matching counts do not prove that every individual divergence has the same cause.
- **Probe at the six divergences with margins above the noise floor** (max 1.75) plus four
  near-tie controls: with identical KV, decode-path and verification-path logits differ by
  ≤ 0.375 at all ten probed positions (including positions 245, 255, 228, 229 within a
  256-token block); their argmaxes agree at seven of ten, the three disagreements being at
  decode-path margins 0.25, 0.0 and 0.0. For the 1.75 case both paths pick the ordinary
  token on identical KV: the probe did not reproduce that flip. Its cause in the full
  executions remains unresolved.

## Output validity (greedy line)

Share of requests hitting the 512-token cap and share whose last 64 tokens contain a
repeated 8-gram. The two engines agree, so greedy acceptance is not inflated by
repetition relative to the baseline; the model card's warning against greedy decoding
stands.

| B | k | capped share | repeated-tail share |
|---:|---:|---:|---:|
| 1 | 0 | 0.35 | 0.12 |
| 1 | 4 | 0.33 | 0.08 |
| 2 | 0 | 0.33 | 0.15 |
| 2 | 4 | 0.33 | 0.15 |
| 4 | 0 | 0.38 | 0.10 |
| 4 | 4 | 0.35 | 0.12 |
| 8 | 0 | 0.33 | 0.12 |
| 8 | 4 | 0.33 | 0.12 |
| 16 | 0 | 0.33 | 0.08 |
| 16 | 4 | 0.35 | 0.12 |

## What would move R further (not implemented)

| Change | Layer | Effect on the round (estimate) |
|---|---|---|
| Fewer kernels per draft step (fuse the draft's per-layer ops, or a draft with fewer layers) | kernel / model | propose 6.4 → ≈ 3 ms at B = 1 for k = 3; R at B = 1 ≈ 1.5 → ≈ 1.8 |
| Fold `compute_logits` and sampling into the draft step graph | framework | ≈ 0.3 ms per draft step |
| Split-KV in the verification attention (FlashAttention 3 / FlashInfer paged prefill) | kernel | verify − 1…3 ms at B ≥ 8; moves the B = 16 break-even |
| A stronger draft (per-position acceptance 0.64 → 0.8, e.g. an EAGLE-style head) | model | tokens/row/round 2.4 → ≈ 3.2 at the same k |

## Limitations

- One GPU model (the Server Edition of the RTX PRO 6000; a Max-Q unit in the same cluster
  runs decode steps ≈ 4 % slower), one model pair, one seed per configuration.
- Offline throughput; TTFT includes queueing in the 48-request batch and is not a serving
  number.
- No output-quality evaluation beyond the greedy repetition shares; strict finite-precision
  equivalence to ordinary decoding is not claimed (G3 and its control show why it cannot
  be, for greedy).
- Timings are GPU-timeline durations between CUDA events; host time between steps is
  included only where the GPU waits for it.

## Reproduce

Use Python 3.10–3.12 on a CUDA machine, install the project and GPU dependencies as in
[Usage](../README.md#usage), and run from the reviewed, committed repository root.
In Bash, enter the local model directories when prompted. The new run records the clean
commit being executed; the published run's source snapshot remains in the evidence package.

```bash
(
  set -euo pipefail
  test -z "$(git status --porcelain)"
  MEASURED_COMMIT=$(git rev-parse HEAD)
  read -r -p 'Local Qwen3-8B model directory: ' TARGET_MODEL
  read -r -p 'Local Qwen3-0.6B model directory: ' DRAFT_MODEL
  test -d "$TARGET_MODEL"
  test -d "$DRAFT_MODEL"
  python tools/fetch_panels.py
  test -s benchmarks/inputs/panel-48.json
  BENCH_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/nano-vllm-reproduce.XXXXXX")
  python tools/gate_smoke.py --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --k 3 --output "$BENCH_OUTPUT/smoke.json"
  python tools/gate_verify_logits.py --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --k 3 --output "$BENCH_OUTPUT/g1.json"
  python tools/gate_sampler_distribution.py --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --k 3 --output "$BENCH_OUTPUT/g2.json"
  python tools/gate_greedy_equivalence.py run --target "$TARGET_MODEL" \
    --k 0 --output "$BENCH_OUTPUT/g3-ordinary.json"
  python tools/gate_greedy_equivalence.py run --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --k 4 --output "$BENCH_OUTPUT/g3-speculative.json"
  python tools/gate_greedy_equivalence.py compare --target "$TARGET_MODEL" \
    --ordinary "$BENCH_OUTPUT/g3-ordinary.json" --speculative "$BENCH_OUTPUT/g3-speculative.json" \
    --noise-floor 0.34375 --output "$BENCH_OUTPUT/g3.json"
  python bench_spec.py --commit "$MEASURED_COMMIT" --target "$TARGET_MODEL" \
    --max-num-seqs 4 --temperature 0.7 --top-k 20 --top-p 0.8 --output "$BENCH_OUTPUT/rec-B4-k0.json"
  python bench_spec.py --commit "$MEASURED_COMMIT" --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --k 3 --max-num-seqs 4 --temperature 0.7 --top-k 20 --top-p 0.8 --output "$BENCH_OUTPUT/rec-B4-k3.json"
  python tools/report_matrix.py "$BENCH_OUTPUT/rec-B4-k0.json" "$BENCH_OUTPUT/rec-B4-k3.json"
  printf 'Results: %s\n' "$BENCH_OUTPUT"
)
```
