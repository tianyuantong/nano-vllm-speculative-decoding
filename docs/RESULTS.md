# Results: DSpark speculative decoding on nano-vLLM

On an RTX PRO 6000 with Qwen3-8B in BF16, the DSpark draft (DeepSpec's `dspark_qwen3_8b_block7`) proposes 7 tokens per round like DFlash, then corrects them left to right with its Markov head; a round yields 2.81–2.95 tokens on average. On the unseen K128b confirmation panel, its steady-state throughput is **2.08–2.20×** plain decoding at C = 1, **1.73–1.80×** at C = 8, **1.43× (1.42–1.45)** at C = 32 and **1.15× (1.13–1.15)** at C = 64, faster at all four.

DFlash in the same engine and jobs gives 1.94–2.06 / 1.68 / 1.31 / 1.01×, and DSpark is faster than it at every concurrency: by 4.4–8.1% at C = 1, 1.2–7.7% at C = 8, 8.9–9.9% at C = 32 and 11.0–14.7% at C = 64. At C = 32 and 64, an estimated 2–6 points of that lead come from the two lines sampling differently (§3). Against vLLM's DSpark (same GPU model, same weights, development panel, wall-clock throughput of the whole run), nano is 7.6–7.9% behind at C = 1, 8 and 32.

**Measurement definitions**
- **Concurrency C**: number of active requests in flight; a new request enters the batch as soon as one finishes.
- **Speedup R**: steady-state throughput (the window where all `C` requests are active) divided by plain decoding (`k0`) at the same concurrency and seed within the same job. The vLLM comparison uses end-to-end wall-clock throughput (§4).
- **K128b**: a held-out 128-prompt evaluation set run once with 3 seeds per concurrency.
- **Protocol**: all decision thresholds and invalidation rules are fixed in [`benchmarks/protocol-v2.1.json`](../benchmarks/protocol-v2.1.json), frozen prior to the GPU runs.

## 1. Confirmation panel (K128b)

K128b has K128's quotas (NIAH 43, code completion 40, open-book QA 37, creative writing 8), drawn from SPEED-Bench `throughput_1k` with a new salt; prompts used in D48 or K128 are excluded by question id and by prompt text. Categories are interleaved in proportion to their quotas, so creative writing is 6% of the first 48 prompts (used at C = 1 and 8), as in the whole panel; in K128 it was 17%.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/speedup-k128b-dark.svg">
    <img alt="Speedup over plain decoding on the K128b panel, median and range over 3 seeds. DSpark draft: 2.08–2.20× at 1 concurrent request, 1.73–1.80× at 8, 1.43× at 32 and 1.15× at 64. DFlash draft: 1.94–2.06×, 1.68×, 1.31× and 1.01×." src="assets/speedup-k128b-light.svg" width="760">
  </picture>
</p>

| C | DFlash (k = 7) | DSpark (k = 7) | DSpark over DFlash, per seed |
|---:|---|---|---|
| 1 | **1.94–2.06** | **2.08–2.20** | +4.4 … +8.1% |
| 8 | **1.68 (1.67–1.71)** | **1.73–1.80** | +1.2 … +7.7% |
| 32 | **1.31 (1.31–1.33)** | **1.43 (1.42–1.45)** | +8.9 … +9.9% |
| 64 | 1.01 (1.00–1.01) | **1.15 (1.13–1.15)** | +11.0 … +14.7% |

Values are the median (min–max) over seeds, to two decimals, or only the range when it spans more than 0.05. **Bold** marks a line that is faster than plain decoding by the protocol's rule.

- **Speedup threshold**: A line is marked faster when every seed's speedup `R` exceeds `1 + 2 × N_t`, where the timing noise `N_t = 1.38%` is measured from 9 repeat pairs on the development matrix on the same machine and engine.
- **DSpark vs DFlash**: At every concurrency, all per-seed paired gains have the same sign and their median exceeds `2 × N_t`.
- **Tokens per round** (round-weighted mean): DSpark `2.81–2.95`, DFlash `2.53–2.70`.

### Latency

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/k128b-gaps-dark.svg">
  <img alt="Gaps between a request's deliveries on K128b, plain decoding against DSpark. ITL p50: 12 and 16 ms at C = 1, 13 and 20 at C = 8, 17 and 31 at C = 32, 21 and 52 at C = 64. ITL p99: 13 and 17, 65 and 76, 69 and 125, 109 and 187 ms." src="assets/k128b-gaps-light.svg" width="760">
</picture>

| C | line | TTFT p50 / p99 | ITL p50 / p99 | ITL p99 without prefill gaps | gaps with a prefill step | round p50 / p99 |
|---:|---|---|---|---:|---:|---|
| 1 | plain decoding | 52 / 61 | 12.3 / 12.5 | 12.5 | 0% | 12.3 / 12.5 |
| 1 | DSpark | 54 / 63 | 16.2 / 16.6 | 16.6 | 0% | 16.2 / 16.6 |
| 8 | plain decoding | 53 / 341 | 13.1 / 65.5 | 13.3 | 2% | 13.1 / 13.3 |
| 8 | DSpark | 55 / 349 | 19.6 / 75.7 | 19.9 | 5% | 19.6 / 19.8 |
| 32 | plain decoding | 52 / 1369 | 17.0 / 69.4 | 17.4 | 7% | 16.9 / 17.4 |
| 32 | DSpark | 54 / 1395 | 30.6 / 125.1 | 30.9 | 22% | 30.4 / 30.8 |
| 64 | plain decoding | 54 / 2748 | 21.4 / 108.5 | 22.2 | 13% | 21.2 / 22.0 |
| 64 | DSpark | 90 / 2797 | 52.0 / 186.8 | 52.4 | 39% | 51.7 / 52.3 |

Milliseconds, medians over the 3 seeds, from [`tools/request_stats.py`](../tools/request_stats.py).

- **ITL** is the gap between two consecutive deliveries to one request, so it shows how tokens arrive, not how fast: in the steady window a DSpark delivery carries 2.80–2.99 tokens, plain decoding's one. On average a request still gets its tokens R times as fast, since in a closed loop its rate is Q_steady ÷ C: from 2.08–2.20× at C = 1 down to 1.15× at C = 64.
- **The longest gaps hold a prefill step.** At C = 8–64, DSpark's ITL p99 is 1.2–1.8 times plain decoding's. Without the gaps that hold a prefill step (the notches), the p99 is about one round. Prefill steps do no decoding in this engine ([DFlash results](v2/RESULTS.md) §5.2), and DSpark, finishing requests sooner, takes in new ones more often: 22% of its gaps hold a prefill step at C = 32, against 7% for plain decoding.
- **TTFT** is the same with and without speculation up to C = 32. A request prefilled alone gets its first token after about 52 ms, one that shares its prefill step with another after about 90. At C = 64, 39–40% of DSpark's requests that arrive after the start share it, against 29–32% for plain decoding, and that moves DSpark's median into the second group. The p99 comes from the start of each run, when all C requests arrive at once, and is the same for both.
- DFlash's TTFT and gaps are within 6% of DSpark's.

## 2. Development panel (D48) and confirmation panel (K128b)

| C | DFlash, D48 | DFlash, K128b | DSpark, D48 | DSpark, K128b |
|---:|---|---|---|---|
| 1 | 1.91–1.99 | 1.94–2.06 | 2.00 (1.98–2.03) | 2.08–2.20 |
| 8 | 1.69 (1.66–1.71) | 1.68 (1.67–1.71) | 1.72–1.82 | 1.73–1.80 |
| 32 | 1.26 (1.26–1.30) | 1.31 (1.31–1.33) | 1.40 (1.38–1.40) | 1.43 (1.42–1.45) |
| 64 | 1.01 (0.99–1.03) | 1.01 (1.00–1.01) | 1.12 (1.11–1.13) | 1.15 (1.13–1.15) |
| tokens per round | 2.47–2.55 | 2.53–2.70 | 2.70–2.79 | 2.81–2.95 |

D48 has 5 seeds per C and also covers C = 2, 4 and 16: DSpark 1.99–2.06, 1.78–1.88 and 1.56–1.65; DFlash 1.91–1.96, 1.71–1.84 and 1.54 (1.54–1.58). DSpark is faster than DFlash at all seven. The two panels agree on the direction and on every verdict.

- **Before candidate sampling.** A first matrix, on the engine before candidate-only sampling (`ed4a35c`), could not call DSpark faster than DFlash at C = 2–16. Profiling traced it to softmax and top-k over the whole vocabulary, row by row. Drawing each row among its top 84 logits instead (top-k 20 plus a margin of 64, under the same truncation rule) made DSpark faster at all seven.

## 3. Where DSpark's lead comes from

DSpark's paired gain over DFlash, split per seed into four factors that multiply to it: tokens per round, round time (DFlash's round ÷ DSpark's), the share of the steady window outside prefill steps, and the rest. A positive percentage favours DSpark; the last column is propose time, DSpark minus DFlash. Medians over seeds, 5 on D48 and 3 on K128b, from [`tools/gain_factors.py`](../tools/gain_factors.py).

| panel | C | gain | tokens per round | round | outside prefill | rest | propose difference (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| D48 | 1 | +3.4% | +6.8% | −3.7% | −0.1% | −0.2% | +0.68 |
| D48 | 2 | +4.2% | +8.5% | −3.7% | −0.3% | −0.1% | +0.65 |
| D48 | 4 | +4.5% | +9.0% | −3.9% | −0.3% | −0.1% | +0.72 |
| D48 | 8 | +4.6% | +9.6% | −4.2% | −0.4% | −0.5% | +0.79 |
| D48 | 16 | +4.0% | +9.1% | −2.8% | −2.1% | −0.5% | +0.58 |
| D48 | 32 | +9.2% | +9.4% | +2.1% | −2.7% | +0.6% | −0.97 |
| D48 | 64 | +11.4% | +8.1% | +5.7% | −3.6% | +0.4% | −2.84 |
| K128b | 1 | +6.6% | +10.3% | −2.8% | −0.1% | −0.3% | +0.64 |
| K128b | 8 | +6.1% | +11.7% | −4.3% | −1.5% | −0.4% | +0.81 |
| K128b | 32 | +9.7% | +9.5% | +3.1% | −3.5% | +0.1% | −0.97 |
| K128b | 64 | +14.1% | +10.2% | +5.9% | −4.0% | +0.1% | −3.01 |

- **Acceptance gain (+7–12% tokens per round)**: Conditioning each draft position on the token sampled at the previous position via the Markov head raises acceptance across all concurrencies.
- **Round duration**: Up to `C = 16`, DSpark's round is 3–4% longer than DFlash's; at `C = 32` and `C = 64` it is 2–6% shorter. Nearly all of the difference lies in `propose` (`verify` differs by under 0.5 ms). The negative `outside prefill` column reflects that DSpark finishes requests faster and admits new requests more frequently, so prefill stalls ([DFlash results](v2/RESULTS.md) §5.2) occupy a larger fraction of its steady-state window (31% at `C = 32`).
- **Top-k candidate sampling at high concurrency**: Candidate-only sampling (`top_k + 64`) is enabled on the DSpark path, whereas DFlash runs softmax and top-k over the full vocabulary. As concurrency grows from `C = 16` to `C = 64`, DFlash's `propose` time increases by 7.9 ms while DSpark's increases by 4.4 ms.
- **Estimated contribution of candidate sampling**:
  - At `C = 32` and `C = 64`, DSpark's `propose` does all the work of DFlash plus sequential Markov correction, so its shorter round on D48 (2.1 and 5.7 percentage points, where `verify` differs by at most 0.18 ms) is attributable to candidate sampling; without that difference, DSpark's net lead on D48 would be roughly +7% and +5% (on K128b, `verify` is also 0.29 and 0.42 ms shorter, accounting for part of the 3.1 and 5.9 points).
  - At `C = 1` and `C = 16`, DFlash spends only 0.23 ms and 0.60 ms in logits sampling outside `lm_head`, bounding any candidate-sampling savings to at most 1.5 and 2.8 percentage points.

### By prompt category

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/k128b-categories-dark.svg">
  <img alt="Tokens per round by prompt category on K128b, DFlash against DSpark: code completion 3.41 and 3.74, NIAH 3.00 and 3.48, creative writing 2.75 and 3.11, open-book QA 2.18 and 2.37." src="assets/k128b-categories-light.svg" width="760">
</picture>

Tokens per round over every C and seed of K128b, weighted by rounds as in §1, from [`tools/request_stats.py`](../tools/request_stats.py).

- **DSpark's extra tokens come from every category**: +8% on open-book QA up to +16% on NIAH. D48 gives +7% to +14%, in the same order.
- **The kind of text matters more than the draft.** With either draft, code completion yields about 1.6 times the tokens per round of open-book QA.
- **Per request**, DSpark's tokens per round have a p10 of 2.39–2.46 and a p90 of 4.00–4.15 at the four concurrencies, DFlash's 2.12–2.24 and 3.65–3.81. Per-request values run higher than the round-weighted ones, because requests with low acceptance take more rounds.

## 4. Against vLLM (D48, wall-clock tok/s)

| C | plain decoding: nano / vLLM v1 | DFlash: nano / vLLM | DSpark: nano / vLLM | DSpark's extra tokens per round over DFlash: nano / vLLM |
|---:|---|---|---|---|
| 1 | 80 / 77 | 154 / 156 | 160 / 173 | +8.3% / +13.7% |
| 8 | 517 / 506 | 800 / 881 | 849 / 919 | +9.1% / +14.1% |
| 32 | 1273 / 1257 | 1567 / 1681 | 1652 / 1794 | +10.4% / +12.2% |

nano's numbers are the median of 5 seeds on the new machine, without the timing repeats. vLLM's are the median of 2 runs on the previous machine (the same GPU model) with `0.1.1.dev52+g372477d2d`, which runs DFlash and DSpark on its v2 model runner; vLLM was not rerun.

- nano's DSpark is 7.6–7.9% behind vLLM's at all three concurrencies. Its DFlash is 1.1% behind at C = 1, and 9.2% and 6.8% behind at C = 8 and 32.
- There are four known differences; which of them matters most is not known.
  - **Tokens per round.** The two DSparks are close (nano 2.71–2.79, vLLM 2.74–2.87), but nano's DFlash already gets more (2.50–2.55 against 2.44–2.52), so DSpark adds less on top of it.
  - **Draft truncation.** nano also applies top-k and top-p to the draft; vLLM's draft uses temperature only.
  - **CUDA graphs.** At C = 1 nano's DSpark propose is 0.68 ms longer than DFlash's, about 4% of a round. vLLM captures the backbone and the 7 sampling steps in one graph; nano uses two.
  - **The prefill stall.** nano's prefill steps do no decoding ([DFlash results](v2/RESULTS.md) §5.2), which costs throughput at C = 8 and 32.

## 5. Correctness

### 5.1 Output quality (GSM8K)

| line | accuracy | difference from plain decoding, points (95% interval) |
|---|---:|---|
| plain decoding | 85.2% | — |
| DSpark | 84.7% | −0.5 (−1.7 … +0.8) |

All 1319 questions × 2 seeds, paired by question and seed. The gate requires the interval's lower end to be above −3 points; DSpark passes. Fewer than 0.2% of answers could not be parsed.

### 5.2 Gates

| gate | what it checks | result |
|---|---|---|
| G1-draft | DSpark's block logits and Markov-corrected logits against DeepSpec's reference, 8 fixtures, fed DeepSpec's greedy chain row by row | relative error 0.59–1.08% (block) and 0.60–0.92% (corrected), bound 2%; argmax equal at every decided position |
| Alignment | tokens per round, nano against DeepSpec, C = 1, temperature only, 3 seeds per side | gap 0.074 at T 0.7 (bound 0.179), 0.027 at T 1.0 (bound 0.329) |
| G1-wide | verification logits against step-by-step decoding | worst gap 0.84, bound 3.0 |
| CUDA graph vs eager | the sampling graph | bit-identical; candidate q differs from the whole-vocabulary rule by at most 6e-8, and none of the gate's 49 rows has kept tokens past the candidate window |
| Sampling graph timing | propose with the 7 sampling steps captured, against eager | C = 1: 4.07 → 3.56–3.65 ms; C = 8: 4.41 → 4.09–4.10 ms |
| G2-multi | the production sampler's multistep output distribution, on the GPU | 29 cases pass |
| G2-real | acceptance rate and output distribution on a real distribution | acceptance error 0.003 (bound 0.02), worst frequency gap 0.0023 (bound 0.0064) |

- **Decided position**: one where the reference's top logit leads the second by more than 0.5.
- G1-draft and G1-wide ran on the engine before candidate sampling (`ed4a35c`); candidate sampling changed neither the model nor verification. Alignment was rerun after it and gave the same numbers: temperature-only sampling does not take the candidate path, and its random stream did not change.

## 6. Estimates made beforehand

| item | estimate | measured |
|---|---|---|
| DSpark against DFlash | +4–11%, from vLLM | medians +3.4–11.4% (D48), +6.1–14.1% (K128b) |
| DSpark's extra tokens per round | about +12–18%, from vLLM | +7–10% (D48), +10–12% (K128b) |
| sampling steps left out of the graph | a round about 6% slower at C = 1 | 3.7% slower (17.04 against 16.43 ms) |
| candidate sampling | saves 0.5 ms (C = 1) to 0.7 ms (C = 16) a round; DSpark over DFlash up to +7% at C = 1 and +4–6% at C = 8–16, C = 8 likely still undecided | the sampling step's GPU time falls by 0.54 ms (C = 1) and 0.94 ms (C = 16); DSpark over DFlash +3.4% at C = 1, +4.6% at C = 8 and +4.0% at C = 16 (D48), and C = 8 is decided |

The extra tokens per round were estimated too high: nano's DFlash already gets more tokens per round than vLLM's, which leaves DSpark less to add (§4).

## 7. Open questions

- **DSpark's tokens per round moved after candidate sampling.** Candidate sampling changes how the random numbers are drawn, so the same seed produces different text. Across 35 (C, seed) pairs on D48, 23 went down and 12 up, a mean of 2.785 → 2.752 (−1.2%); DFlash, unchanged, reproduces all 35 exactly. At C = 64 all five seeds went down, by 0.03–0.06, while four of five went up at C = 32, which is consistent with random variation. The sampled distribution can change in only one case, a row with more than 84 tokens at or above its cut value; otherwise the candidate path and the whole-vocabulary rule agree row by row (`num_candidates` in `nanovllm/engine/speculative.py`, and the tie test in `tests/test_dspark_propose.py`). How often such rows occur in real logits cannot be told from the gate's 49 rows; it needs a count on the GPU, about 5 minutes.
- What costs nano the 8% against vLLM's DSpark (§4).

Measurement scope as in the DFlash results: offline closed loop, one GPU model, one target model, random sampling (T 0.7, top-k 20, top-p 0.8); see [v2/RESULTS.md](v2/RESULTS.md) §7.

## 8. Variable-length verification

Fixed-length DSpark verifies all 8 positions in every row of every round (yielding 2.81–2.95 tokens per round). Target verification accounts for roughly three quarters of a round's duration, and its marginal cost grows with batch size: in the measured cost table, verifying 8 positions per row instead of 1 increases round time by 14% at batch size 16, 25% at 32, and 49% at 64.

With `variable_length=True`, the engine implements DSpark's Algorithm 1: it ranks all candidate positions across the batch by predicted survival probability, admits positions while estimated throughput (computed from the measured round-time table) continues to rise, and rounds the total verified token count up to a captured CUDA Graph size ([usage](DECODING.md#variable-length-verification)).

These runs were executed on a Colab G4 RTX PRO 6000 (600 W) instance under [`protocol-v2.3.json`](../benchmarks/protocol-v2.3.json) on the K128b confirmation panel across 3 seeds:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/speedup-variable-dark.svg">
    <img alt="Speedup over plain decoding on K128b, medians over 3 seeds. Fixed-length DSpark: 2.08×, 1.76×, 1.44×, 1.14× at 1, 8, 32, 64 concurrent requests. Variable-length DSpark: 2.02×, 1.79×, 1.48×, 1.25×. Qwen3-0.6B draft: 1.50×, 1.21×, 1.00×, 0.85×." src="assets/speedup-variable-light.svg" width="760">
  </picture>
</p>

| C | 0.6B draft (k = 3) | DSpark, fixed length | DSpark, variable length | Variable over fixed, per seed | Positions verified per row | Tokens per round, fixed / variable |
|---:|---|---|---|---|---:|---|
| 1 | **1.50 (1.49–1.52)** | **2.05–2.18** | **2.00–2.12** | −2.83 … −2.48% | 8.00 | 2.82 / 2.82 |
| 8 | **1.21 (1.20–1.22)** | **1.71–1.80** | **1.73–1.80** | −3.96 … +4.51% | 6.68 | 2.81 / 2.87 |
| 32 | 1.00 (1.00–1.01) | **1.44 (1.44–1.46)** | **1.48 (1.47–1.49)** | **+2.17 … +2.77%** | 4.62 | 2.95 / 2.79 |
| 64 | 0.85 (0.84–0.85) | **1.14 (1.12–1.15)** | **1.25 (1.23–1.26)** | **+8.19 … +10.06%** | 4.43 | 2.93 / 2.77 |

Values follow §1 (`median (min–max)`). **Bold** marks lines faster than plain decoding across all seeds, and paired gains where all seeds share the same sign with a median exceeding `2 × N_t` (`N_t = 0.14%` from 15 repeat pairs; 75 runs, 0 void). Plain decoding throughput between the `main` tree and `baseline/qwen3-0.6b-draft` agrees within `2 × N_t` at every concurrency, allowing direct comparison in a single table.

- **High-concurrency speedup (`C = 32` and `C = 64`)**: Variable-length verification improves throughput by medians of +2.21% at `C = 32` and +8.91% at `C = 64` (matching the 5-seed D48 development matrix at +2.41% and +8.71%, where `C = 32` was the crossover concurrency).
- **Verification phase savings**: At `C = 64`, verification time drops from 36.51 ms to 27.96 ms per round and acceptance from 4.31 ms to 2.45 ms. Each round is 18.6% shorter while delivering 5.6% fewer tokens, resulting in a +16.1% speedup on decode rounds alone and +8.9% overall once prefill stalls are included (32.5% of the steady window on fixed-length vs 35.9% on variable-length). Full-run wall-clock throughput shows matching gains of +2.63% (`+2.39 … +4.22%`) at `C = 32` and +8.87% (`+6.88 … +9.61%`) at `C = 64`.
- **Single-request overhead (`C = 1`)**: Every round verifies all 8 positions and produces token-identical outputs to fixed-length DSpark across all 3 seeds; the 2.66% slowdown reflects the scheduling and confidence-scoring overhead. At `C = 8`, per-seed gains span `−3.96%` to `+4.51%`.
- **Latency impact**: At `C = 64`, ITL p50 drops from 51.5 ms to 42.3 ms while TTFT p50 remains at 88–89 ms (ITL p99 is 138–173 ms on variable-length vs 182–183 ms on fixed-length).
- **Verification gates**: Variable-length verification passes `G1-wide` (max difference 1.14 against the 3.0 bound), 71 multi-step sampler distribution cases (`G2-multi`), `G2-real`, and paired GSM8K accuracy (84.2% vs 85.1% plain decoding, −0.87 pt difference, 95% CI −2.14 to +0.40 pt, threshold −3 pt).

Engine `70e0d4d`, baseline branch `4d8426f`, verify schedule [`benchmarks/inputs/verify-schedule-g4.json`](../benchmarks/inputs/verify-schedule-g4.json), driver 580.82.07; raw run JSONs and gates are archived in release [`results-variable-length`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-variable-length).

## 9. Round-robin prefill

A prefill step does no decoding in this engine, and the default scheduler prefills every waiting request before it decodes again (§1, Latency). With `round_robin_prefill=True`, a step admits one request and a decode step follows each prefill step ([usage](DECODING.md#round-robin-prefill)). On the D48 development panel, closed loop, 3 seeds, each line run with and without the option back to back:

| Concurrent requests | line | ITL p99, ms | ITL p99 ratio, per seed | Throughput ratio | TTFT p50, ms | Gaps with a prefill step |
|---:|---|---|---|---|---|---|
| 8 | plain decoding | 61.0 → 61.0 | 1.00 (0.98–1.00) | 1.00 (0.99–1.00) | 50 → 49 | 2% → 2% |
| 8 | DSpark, fixed length | 71.5 → 71.5 | 1.00 (1.00–1.00) | 0.99 (0.99–1.00) | 52 → 50 | 6% → 6% |
| 8 | DSpark, variable length | 71.7 → 71.6 | 1.00 (1.00–1.00) | 1.00 (0.96–1.01) | 52 → 50 | 6% → 6% |
| 32 | plain decoding | 73.6 → 68.0 | 0.92 (0.91–1.00) | 1.00 (0.99–1.00) | 51 → 51 | 8% → 9% |
| 32 | DSpark, fixed length | 124.6 → 89.2 | **0.72 (0.70–0.72)** | 1.00 (1.00–1.00) | 52 → 52 | 23% → 25% |
| 32 | DSpark, variable length | 128.4 → 87.2 | **0.68 (0.67–0.71)** | 1.00 (0.99–1.01) | 59 → 52 | 21% → 25% |
| 64 | plain decoding | 109.5 → 76.4 | **0.70 (0.68–0.70)** | 0.98 (0.98–0.99) | 57 → 51 | 14% → 18% |
| 64 | DSpark, fixed length | 184.6 → 111.8 | **0.61 (0.60–0.61)** | 0.98 (0.98–1.00) | 89 → 151 | 40% → 52% |
| 64 | DSpark, variable length | 176.2 → 102.8 | **0.58 (0.56–0.58)** | 0.98 (0.97–1.00) | 89 → 142 | 38% → 49% |

Milliseconds are medians over the 3 seeds, default → round-robin; ratios are round-robin ÷ default per seed, `median (min–max)`. ITL is taken over the steady window; the throughput ratio is of `Q_steady`.

- **Tail latency.** At 32 and 64 concurrent requests DSpark's ITL p99 falls to 0.58–0.72 of the default scheduler's, and plain decoding's to 0.92 and 0.70. Nothing changes at 8, where a gap rarely holds more than one prompt's prefill.
- **Throughput.** Steady-state throughput stays at 0.98–1.00 of the default's: the same prefills run, one at a time.
- **Cost at 64 concurrent requests.** An arriving request waits one decode step for each request ahead of it, and DSpark's TTFT p50 rises from 89 ms to 142–151 ms. With fixed-length DSpark, 52% of the gaps in the steady window then hold one prefill step (40% under the default), so the median gap moves from 53 ms to 101 ms; with variable length the share stays under half and the median stays at 44 ms.

Engine `9a97b6f`, where the option was a switch of `bench_spec.py`; lines, sampling and panel from [`protocol-v2.3.json`](../benchmarks/protocol-v2.3.json). Not run on K128b.

## Appendix: environment and data

| item | value |
|---|---|
| engine | `2d64fa4` (`ed4a35c` for G1-draft and G1-wide) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 600 W power limit, ECC on, driver 580.173.02 |
| software | torch 2.9.1+cu130 (CUDA 13.0), flash-attn 2.8.3, triton 3.5.1, transformers 5.16.1, Python 3.12.3 |
| models | target Qwen3-8B; drafts DeepSpec's `dflash_qwen3_8b_block7` and `dspark_qwen3_8b_block7` (`deepseek-ai/dspark_qwen3_8b_block7` at `03326e5`) |
| protocol | [`benchmarks/protocol-v2.1.json`](../benchmarks/protocol-v2.1.json), SHA256 `f8adddb463ffb268…` |
| panels | D48 `071befa6898b3187…`; K128b `cfd53af26e0cdcc2…`; GSM8K test set (openai/grade-school-math `3101c7d`) `3730d312f6e34405…` |
| tools | [`tools/report_v2.py`](../tools/report_v2.py) applies the protocol's pre-registered rules; [`tools/gain_factors.py`](../tools/gain_factors.py) splits DSpark's gain (§3); [`tools/plot_speedup.py`](../tools/plot_speedup.py) draws the README figure; [`tools/request_stats.py`](../tools/request_stats.py) gives the latency and per-category tables and their figures (§1, §3) |

Each run writes one JSON file (schema `bench-v2`) with the protocol's SHA256, a manifest (GPU, driver, library versions, commit, node, and SHA256 of the installed packages and of the panel) and the GPU state before and after the run. Every run's file and log, the gates' outputs, the job scripts and the panels are the release asset [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark), whose README gives the commands that regenerate these tables.

The DFlash results on K128 are in [v2/RESULTS.md](v2/RESULTS.md), and the September results of the 0.6B draft in [v1/RESULTS.md](v1/RESULTS.md).
