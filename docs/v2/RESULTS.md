# Results: DFlash speculative decoding on nano-vLLM

> Superseded by [the DSpark results](../RESULTS.md); this page keeps the DFlash results on the K128 panel.

On an RTX PRO 6000 with Qwen3-8B in BF16, the DFlash draft (DeepSpec's `dflash_qwen3_8b_block7`) proposes 7 tokens per round and a round yields 2.8 on average. On the unseen K128 confirmation panel, its steady-state throughput is **2.11–2.24×** plain decoding at C = 1, **1.75–1.85×** at C = 8 and **1.32–1.40×** at C = 32. At C = 64 it is 1.08× (1.06–1.09), which is within timing noise.

The earlier 0.6B draft (k = 3) in the same engine gives 1.54 / 1.21 / 0.96 / 0.82×, so DFlash is 1.29–1.56× the 0.6B line at every concurrency. Against vLLM (same GPU, same draft weights, development panel, wall-clock throughput of the whole run), DFlash is level at C = 1 (155 vs 156 tok/s), 9% behind at C = 8 and 7% behind at C = 32.

**How it is measured**
- **Concurrency C**: the number of requests in flight; a new request enters as soon as one finishes.
- **Speedup R**: steady-state throughput (the stretch where all C requests are running) divided by plain decoding's (k0) at the same C and seed, in the same job. Against vLLM the comparison is wall-clock throughput of the whole run (§3).
- **K128**: 128 prompts whose outputs nobody looked at before this run. It is run once, with 3 seeds per C.
- **Rules**: every threshold and verdict is fixed in [`benchmarks/protocol-v2.json`](../../benchmarks/protocol-v2.json), frozen before the confirmation runs.

## 1. Confirmation panel (K128)

K128 is 128 prompts from SPEED-Bench `throughput_1k`, ranked within each category by sha256(salt + question id): NIAH 43, code completion 40, open-book QA 37, creative writing 8. None of them overlap D48.

Which prompts each concurrency uses:
- C = 1 and 8 take the first 48 prompts. Creative writing is 17% of those 48 against 6% of the whole panel, which likely lifts the speedup at these two concurrencies by about 0.04 (estimate).
- C = 32 takes all 128.
- C = 64 cycles through the panel twice.

One DFlash run (C = 1, seed 17011) had a round of 193 ms against a median of 15.7 ms, so the pre-registered rule voids it. Almost all of that time went to the context-store step, 177 ms against a median of 0.26 ms; propose and verify in that round were normal, and the cause is not known. The protocol also voids lines of one (C, seed) split across jobs, so all three lines of that group were rerun back to back in one job.
- All three lines of that group take their values from the rerun: DFlash 2.19, and the 0.6B line 1.567 against 1.566 in the first run.
- All 48 requests of all three lines produced the same tokens as the first run.

| C | 0.6B draft (k = 3) | DFlash (k = 7) | DFlash ÷ 0.6B |
|---:|---|---|---|
| 1 | **1.54 (1.54–1.57)** | **2.11–2.24** | 1.37–1.45 |
| 8 | **1.21 (1.19–1.21)** | **1.75–1.85** | 1.44–1.56 |
| 32 | 0.96 (0.95–0.96) | **1.32–1.40** | 1.38–1.48 |
| 64 | 0.82 (0.81–0.82) | 1.08 (1.06–1.09) | 1.29–1.33 |

Values are the median (min–max) over seeds, to two decimals, or only the range when it spans more than 0.05. **Bold** marks a line that is faster than plain decoding by the protocol's rule.

- **What "faster" means.** Every seed's R must be above 1 + 2 × timing noise. The confirmation panel has no timing repeats by design, so the timing noise is borrowed from the development matrix: 3.75%, from the same machine and the same engine.
- **C = 64 is close to the line.** The three seeds give 1.057, 1.086 and 1.080. A timing noise of 2.5% would call this faster; the table takes the conservative side.
- **DFlash against the 0.6B draft.** At every C, the per-seed paired gains all have the same sign and their median exceeds 2 × timing noise, so DFlash is faster than the 0.6B draft at every concurrency.
- The 0.6B line drops below plain decoding from C = 32; DFlash stays above 1 up to C = 64.
- Tokens per round, averaged over rounds: DFlash 2.75–2.85, 0.6B 2.49–2.53.

## 2. Development panel (D48) and confirmation panel (K128)

| C | DFlash, D48 | DFlash, K128 | 0.6B, D48 | 0.6B, K128 |
|---:|---|---|---|---|
| 1 | 1.91–1.99 | 2.11–2.24 | 1.45–1.52 | 1.54 |
| 8 | 1.66–1.72 | 1.75–1.85 | 1.17 | 1.21 |
| 32 | 1.21–1.30 | 1.32–1.40 | 0.94 | 0.96 |
| 64 | 1.01 | 1.08 | 0.79 | 0.82 |
| tokens per round | 2.47–2.55 | 2.75–2.85 | 2.37–2.40 | 2.49–2.53 |

D48 is the 48-prompt panel of the September release, with 5 seeds per C. It also covers C = 2, 4 and 16:
- DFlash: 1.91–1.97, 1.69–1.85 and 1.55;
- 0.6B: 1.42, 1.32 and 0.99–1.04.

The two panels agree on the direction and on every verdict.

The engine is the same on both panels; what differs is acceptance. K128 yields about 0.3 more tokens per round. By category (C = 1, DFlash, tokens per round):

| category | D48 | K128 |
|---|---:|---:|
| code completion | 3.42 | 3.49 |
| NIAH | 2.80 | 2.92 |
| creative writing | 2.59 | 2.95 |
| open-book QA | 2.13 | 2.42 |

Prompt length is not the cause. Both panels have a median length of about 1020 tokens, and D48 has only three longer prompts (1229, 1264 and 2262 tokens).

## 3. Against vLLM (D48, wall-clock tok/s)

| C | plain decoding: nano / vLLM v1 / vLLM v2 | 0.6B: nano / vLLM | DFlash: nano / vLLM | vLLM DSpark |
|---:|---|---|---|---:|
| 1 | 80 / 77 / 73 | 119 / 108 | 155 / 156 | 173 |
| 8 | 515 / 506 / 492 | 608 / 596 | 804 / 881 | 919 |
| 32 | 1276 / 1257 / 1224 | 1231 / 1279 | 1567 / 1681 | 1794 |

nano's numbers are the median of 5 seeds, without the timing repeats. vLLM's are the median of 2 runs.
- **vLLM version.** `0.1.1.dev52+g372477d2d`, a development build from 2026-09-12. It runs the 0.6B draft on its v1 model runner and DFlash and DSpark on v2, so the table lists its plain decoding on both runners.
- **Run-to-run variation.** vLLM's two runs differ by 7.2% at C = 8, so the gap at that concurrency is not stable; at C = 32 they differ by 0.8%.
- vLLM was not run on K128.

What the table shows:
- **Plain decoding.** nano is 1.5–4% faster than vLLM's v1 runner and 4–10% faster than its v2 runner. nano's plain decoding runs at about 69–71% memory-bandwidth utilisation: bytes of weights and KV read ÷ step time ÷ the 1.79 TB/s peak.
- **DFlash.** It is level at C = 1 and 9% and 7% behind at C = 8 and 32. The most likely cause is the prefill stall of §5.2; its size matches, but it has not been verified directly.
- **DSpark.** On vLLM, DSpark is another 4–11% faster than DFlash.

## 4. Correctness

### 4.1 Output quality (GSM8K)

| line | accuracy | difference from plain decoding, points (95% interval) |
|---|---:|---|
| plain decoding | 85.2% | — |
| 0.6B | 85.6% | +0.4 (−0.8 … +1.6) |
| DFlash | 84.9% | −0.3 (−1.5 … +0.9) |

All 1319 questions × 2 seeds at C = 32, paired by question and seed. The gate requires the interval's lower end to be above −3 points; both lines pass. Fewer than 0.2% of answers could not be parsed.

### 4.2 Gates

| gate | what it checks | result |
|---|---|---|
| G1-draft | DFlash block forward against DeepSpec's reference, 8 fixtures | relative error 0.60–1.05% (bound 2%); argmax equal at every decided position |
| Alignment | tokens per round, nano against DeepSpec, C = 1, 3 seeds per side | gap 0.009 at T 0.7 (bound 0.138), 0.032 at T 1.0 (bound 0.275); see below |
| CUDA graph vs eager | the verify graph and the draft block graph | logits, exported hidden states and KV bit-identical at captured sizes |
| G2-multi | the production sampler's multistep output distribution, k = 2, 3, 7, on the GPU | 29 cases pass |
| G1-wide | verification logits against step-by-step decoding, 8 prompts, B = 1/4/16 | worst gap 0.75 (0.6B) and 0.84 (DFlash), bound 3.0; argmax equal at every decided position |
| G2-real | acceptance rate and output distribution on real distributions | see below |

- **Decided position**: one where the reference's top logit leads the second by more than 0.5.
- **Alignment, paired.** Over 24 per-prompt pairs, nano minus DeepSpec is +0.056 ± 0.19 at T 0.7 and +0.064 ± 0.26 at T 1.0 (2 standard errors); both are consistent with 0.
- **G2-real.**
  - DFlash: acceptance rate 0.776 against an analytic 0.777, worst frequency gap 0.0011 (bound 0.0066).
  - 0.6B: acceptance rate 0.453 against an analytic 0.456, worst frequency gap 0.0008.

### 4.3 Three rules revised after seeing the data

**Alignment**
- *First run:* the rule treated the rounds of one prompt as independent samples, which made the standard error too small. T 1.0 failed on the first seed, and across three seeds the difference changed sign (+0.18 … −0.20).
- *Revision:* 3 seeds per side, with the standard error taken from per-run means.
- *After:* passes.

**G1-wide**
- *First run:* worst gaps of 1.25 (0.6B) and 1.75 (DFlash), above the fixed bound of 0.5. But decode alone, changing only the batch size, already differs by up to 1.5 (at B = 16), so even the reference path would fail a 0.5 bound.
- *Revision:* the bound is twice the noise floor, and the floor is measured first in the same job.
- *After:* passes. The argmax rule passed both times.

**G2-real** (revised twice)
- *First run:* both lines picked a position where the target split 0.5 / 0.5 and the draft put everything on one token, which tests only the simplest rejection.
- *Second run:* the positions it picked had a target with a single token, so the output-distribution check tested nothing.
- *Revisions:* first the draft's q, then also the target's p, must put probability on at least 2 tokens.
- *After:* passes, now testing the residual distribution over several tokens.

The protocol's `revised` notes record each revision together with the first run's numbers.

## 5. Where the time goes

### 5.1 One round

DFlash on D48, median GPU time per phase (ms):

| C | propose | verify | context store | accept | round |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.85 | 12.11 | 0.26 | 0.32 | 15.7 |
| 8 | 3.33 | 14.48 | 0.29 | 0.45 | 18.7 |
| 32 | 6.50 | 22.23 | 0.34 | 2.20 | 31.8 |
| 64 | 11.80 | 38.23 | 0.54 | 4.31 | 55.9 |

**Verification is 68–77% of a round**, and it gets more expensive per row as concurrency rises:
- From C = 32, each extra verified row costs about 0.023 ms; at C = 64, about 0.03–0.04 ms. Verifying 8 positions for every request costs more the more requests there are.
- Yet only 8% of rounds accept five or more drafts. Verifying only the first few drafts is therefore a possible follow-up, estimated at about +3% at C = 32 and +8% at C = 64 (estimate).
- vLLM's adaptive verification, compared on the same attention backend, gives 0.995× at C = 8 and 32.

**The two CUDA graphs**
- Capturing the draft block forward in a graph made a round 16% faster at C = 1 (18.8 → 15.7 ms) and 12% faster at C = 8. Profiling showed that before it, the GPU was waiting for the CPU to launch kernels.
- A graph for the context store gained nothing: round time did not change, and the phase's GPU time fell by only 0.04 ms, so it was reverted. The expected explanation was that its launch cost was already hidden behind verification's GPU time; the measurement confirms only that there was no gain.

### 5.2 The prefill stall

nano's prefill steps do no decoding. Each new request pauses all the others for one step of about 52 ms, because a prefill step handles one prompt of about 1020 tokens.

| C | prefill share of the steady window (DFlash / plain) | DFlash throughput lost to it (lower–upper bound) | speedup: measured → without prefill time |
|---:|---|---|---|
| 8 | 15.7% / 8.7% | 5–16% | 1.70 → 1.84 |
| 32 | 29.5% / 21.6% | 15–29% | 1.27 → 1.40 |
| 64 | 30.7% / 29.5% | 26–31% | 1.01 → 1.03 |

The lower bound assumes each stall displaces one round; the upper bound counts all prefill time as lost.

- **Why DFlash loses more.** DFlash decodes faster, so prefill takes a larger share of its time, and it loses 8–9 points more than plain decoding. vLLM mixes prefill into decode steps instead: the step gets longer, but the decoding requests still get their tokens in it, so it never stops a whole step.
- **The gap to vLLM.** Without speculation, nano is 2–4% faster than vLLM; with DFlash it is 7% slower. This asymmetry of about 10 points matches the extra stall loss. It is not verified directly yet; verifying it needs prefill merged into decode steps.
- **Inter-token latency.** ITL is the gap between two consecutive deliveries to one request. For DFlash at C = 32, its p99 is 131 ms, and 32.9 ms without the gaps that contain a prefill; those gaps are 21% of all gaps.

## 6. Estimates made beforehand

Before the first measurement, the plan estimated:

| item | estimate | measured |
|---|---|---|
| round time at C = 1 | 15–16.5 ms | 15.7 ms |
| tokens per round | 3.5–5, from the T = 1.0 numbers of the [DSpark paper](https://arxiv.org/abs/2607.05147) | 2.5 (D48), 2.8 (K128); DeepSpec's own implementation on 8 D48 prompts with temperature-only sampling (T 0.7) also gives just 2.55 (three runs, 2.49–2.59) |
| speedup at C = 1 | 2.6–4.0 | 1.91–1.99 (D48), 2.11–2.24 (K128) |

The cost was estimated well and the gain too high: acceptance on these prompts is lower than on the paper's benchmarks.

## 7. Limitations

- Offline closed loop: fixed concurrency, with a new request as soon as one finishes. This is not an online serving benchmark, and no service-level objectives were measured.
- One GPU model (RTX PRO 6000 Blackwell) and one target model (Qwen3-8B).
- Random sampling: T 0.7, top-k 20, top-p 0.8. Speculative decoding keeps the output distribution, not the exact tokens; §4 checks the distribution.
- The comparison with vLLM is on D48 only, against a development build of vLLM.
- The prefill stall is located but not fixed. Verifying fewer drafts per round and measuring the verify cost curve are not done yet.

## Appendix: environment and data

| item | value |
|---|---|
| engine | development matrix at `1f78eca`, confirmation set at `10b38b4` (same engine code) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 600 W power limit, ECC on, driver 580.173.02 |
| software | torch 2.9.1+cu130 (CUDA 13.0), flash-attn 2.8.3, triton 3.5.1, transformers 5.16.1, Python 3.12.3 |
| models | target Qwen3-8B; drafts Qwen3-0.6B and DeepSpec's `dflash_qwen3_8b_block7` |
| protocol | [`benchmarks/protocol-v2.json`](../../benchmarks/protocol-v2.json), SHA256 `d012e92933cda36f…` |
| panels | D48 `071befa6898b3187…`; K128 `44f6fb6b4922c3d8…`; GSM8K test set (openai/grade-school-math `3101c7d`) `3730d312f6e34405…` |
| tools | [`tools/report_v2.py`](../../tools/report_v2.py) applies the protocol's pre-registered rules; [`tools/stall_stats.py`](../../tools/stall_stats.py) and [`tools/itl_stats.py`](../../tools/itl_stats.py) break down prefill stalls and ITL; [`tools/plot_speedup.py`](../../tools/plot_speedup.py) draws the README figure |

Each run writes one JSON file (schema `bench-v2`) with the protocol's SHA256, a manifest (GPU, driver, library versions, commit, node, and SHA256 of the installed packages and of the panel) and the GPU state before and after the run. Every run's file and log, the gates' outputs, the job scripts and the panels are the release asset [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark), whose README gives the commands that regenerate these tables.

The September results for the 0.6B draft are in [v1/RESULTS.md](../v1/RESULTS.md).
