# Results: DSpark speculative decoding on nano-vLLM

On an RTX PRO 6000 with Qwen3-8B in BF16, the DSpark draft (DeepSpec's `dspark_qwen3_8b_block7`) proposes 7 tokens per round like DFlash, then corrects them left to right with its Markov head; a round yields 2.81–2.95 tokens on average. On the unseen K128b confirmation panel, its steady-state throughput is **2.08–2.20×** plain decoding at C = 1, **1.73–1.80×** at C = 8, **1.43× (1.42–1.45)** at C = 32 and **1.15× (1.13–1.15)** at C = 64, faster at all four.

DFlash in the same engine and jobs gives 1.94–2.06 / 1.68 / 1.31 / 1.01×, and DSpark is faster than it at every concurrency: by 4.4–8.1% at C = 1, 1.2–7.8% at C = 8, 8.9–9.9% at C = 32 and 11.0–14.7% at C = 64. At C = 32 and 64, at least 2–6 points of that lead come from the two lines sampling differently (§3). Against vLLM's DSpark (same GPU model, same weights, development panel, wall-clock throughput of the whole run), nano is 7.6–7.9% behind at C = 1, 8 and 32.

**How it is measured**
- **Concurrency C**: the number of requests in flight; a new request enters as soon as one finishes.
- **Speedup R**: steady-state throughput (the stretch where all C requests are running) divided by plain decoding's (k0) at the same C and seed, in the same job. Against vLLM the comparison is wall-clock throughput of the whole run (§4).
- **K128b**: a new 128-prompt confirmation panel whose outputs nobody looked at before this run. It is run once, with 3 seeds per C.
- **Rules**: every threshold and verdict is fixed in [`benchmarks/protocol-v2.1.json`](../benchmarks/protocol-v2.1.json), frozen before the GPU runs.

## 1. Confirmation panel (K128b)

K128b has K128's quotas (NIAH 43, code completion 40, open-book QA 37, creative writing 8), drawn from SPEED-Bench `throughput_1k` with a new salt; prompts used in D48 or K128 are excluded by question id and by prompt text. Categories are interleaved in proportion to their quotas, so creative writing is 6% of the first 48 prompts (used at C = 1 and 8), as in the whole panel; in K128 it was 17%.

| C | DFlash (k = 7) | DSpark (k = 7) | DSpark over DFlash, per seed |
|---:|---|---|---|
| 1 | **1.94–2.06** | **2.08–2.20** | +4.4 … +8.1% |
| 8 | **1.68 (1.67–1.71)** | **1.73–1.80** | +1.2 … +7.8% |
| 32 | **1.31 (1.31–1.33)** | **1.43 (1.42–1.45)** | +8.9 … +9.9% |
| 64 | 1.01 (1.00–1.01) | **1.15 (1.13–1.15)** | +11.0 … +14.7% |

Values are the median (min–max) over seeds, to two decimals, or only the range when it spans more than 0.05. **Bold** marks a line that is faster than plain decoding by the protocol's rule.

- **What "faster" means.** Every seed's R must be above 1 + 2 × timing noise. The confirmation panel has no timing repeats by design, so the timing noise is borrowed from the development matrix: 1.38%, from 9 repeat pairs on the same machine and engine.
- **DSpark against DFlash.** At every C, the per-seed paired gains all have the same sign and their median exceeds 2 × timing noise, so DSpark is faster than DFlash at every concurrency.
- Tokens per round, averaged over rounds: DSpark 2.81–2.95, DFlash 2.53–2.70.
- **One void and its rerun.** One DFlash run (C = 1, seed 17011) had a round of 199 ms against a median of 15.7 ms, so the pre-registered rule voids it; the record is kept. Of those 199 ms, 182 went to the context-store step (median 0.26 ms); propose and verify were normal. The DFlash results' only void was the same cell and step (177 ms) at almost the same point: round 3549, 57.4 s into the run, against round 3544 and 57.3 s here, both the first DFlash run of a confirmation job. Of 78 runs at C = 1, these two are the only ones with a round over twice the median. The group's three lines were rerun back to back in one job: valid, with the same tokens as the first run on all three lines, and the group takes its values from the rerun (DFlash 2.06). The same seed and the same outputs did not reproduce the stall, so it does not follow from the data; the cause is not known.

## 2. Development panel (D48) and confirmation panel (K128b)

| C | DFlash, D48 | DFlash, K128b | DSpark, D48 | DSpark, K128b |
|---:|---|---|---|---|
| 1 | 1.91–1.99 | 1.94–2.06 | 2.00 (1.98–2.03) | 2.08–2.20 |
| 8 | 1.69 (1.66–1.71) | 1.68 (1.67–1.71) | 1.72–1.82 | 1.73–1.80 |
| 32 | 1.26 (1.26–1.30) | 1.31 (1.31–1.33) | 1.40 (1.38–1.40) | 1.43 (1.42–1.45) |
| 64 | 1.01 (0.99–1.03) | 1.01 (1.00–1.01) | 1.12 (1.11–1.13) | 1.15 (1.13–1.15) |
| tokens per round | 2.47–2.55 | 2.53–2.70 | 2.70–2.79 | 2.81–2.95 |

D48 has 5 seeds per C and also covers C = 2, 4 and 16: DSpark 1.99–2.06, 1.78–1.88 and 1.56–1.65; DFlash 1.91–1.96, 1.71–1.84 and 1.54 (1.54–1.58). DSpark is faster than DFlash at all seven. The two panels agree on the direction and on every verdict.

- **Before candidate sampling.** A first matrix, on the engine before candidate-only sampling (`3dc0dba`), could not call DSpark faster than DFlash at C = 2–16. Profiling traced it to softmax and top-k over the whole vocabulary, row by row. Drawing each row among its top 84 logits instead (top-k 20 plus a margin of 64, under the same truncation rule) made DSpark faster at all seven.
- **Releasing the confirmation set.** The design's stop rule, revised after the first matrix and before the second, released the confirmation set when, at every C, DSpark's tokens per round were higher and the median paired gain was positive; both held at all seven.
- **A new machine.** The cloud instance changed between the DFlash and DSpark runs. Plain decoding on the new one gives 80.1 / 557.4 / 1459.0 / 2031.9 tok/s at C = 1 / 8 / 32 / 64, against 80.4 / 554.8 / 1466.0 / 2040.5 before: within ±0.5%, below the timing noise, so §4 compares the new nano numbers with vLLM's from the old machine.

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

- **7–12% more tokens per round** is the Markov head's work: later positions in the block know what the earlier ones drew, so more of them are accepted. This is where DSpark's lead comes from.
- **Round time.** Up to C = 16 DSpark's round is 3–4% longer; at C = 32 and 64 it is 2–6% shorter. Almost all of the difference is in propose; verification differs by under 0.5 ms. The negative "outside prefill" column means DSpark, decoding faster, takes in new requests more often, so prefill stalls ([DFlash results](v2/RESULTS.md) §5.2) take more of its time.
- **Why the round gets shorter at high concurrency.** The two lines sample differently: candidate-only sampling applies to DSpark only, while DFlash still runs softmax and top-k over the whole vocabulary, which costs more as concurrency rises. From C = 16 to 64, DFlash's propose grows by 7.9 ms, DSpark's by 4.4 ms.
- **How much of the lead that asymmetry accounts for.**
  - C = 32 and 64: with the same sampling, DSpark's round cannot be shorter than DFlash's, since it does everything DFlash does plus the row-by-row correction. So the whole shorter-round share (2.1 and 5.7 points on D48, 3.1 and 5.9 on K128b) counts as asymmetry, a lower bound. Without it, DSpark's lead at these two concurrencies would be about +5–8% (estimate).
  - C = 1 and 16: profiled, DFlash's logits-and-sampling step spends only 0.23 ms and 0.60 ms outside the logits matrix multiply (lm_head). Candidate sampling could save at most that, 1.5% and 2.8% of a round, so the asymmetry accounts for at most 1.5 and 2.8 points there.
- Measuring the same-sampling number exactly needs DFlash on candidate sampling too; that has not been done.

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
- G1-draft and G1-wide ran on the engine before candidate sampling (`3dc0dba`); candidate sampling changed neither the model nor verification. Alignment was rerun after it and gave the same numbers: temperature-only sampling does not take the candidate path, and its random stream did not change.

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
- The context-store stall of the void run (§1).
- What costs nano the 8% against vLLM's DSpark (§4).

Measurement scope as in the DFlash results: offline closed loop, one GPU model, one target model, random sampling (T 0.7, top-k 20, top-p 0.8); see [v2/RESULTS.md](v2/RESULTS.md) §7.

## Appendix: environment and data

| item | value |
|---|---|
| engine | `f30ed0a` for the gates after candidate sampling, the development matrix, the confirmation set and the rerun; `3dc0dba` for G1-draft, G1-wide and the first matrix |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 600 W power limit, ECC on, driver 580.173.02 |
| software | torch 2.9.1+cu130 (CUDA 13.0), flash-attn 2.8.3, triton 3.5.1, transformers 5.16.1, Python 3.12.3 |
| models | target Qwen3-8B; drafts DeepSpec's `dflash_qwen3_8b_block7` and `dspark_qwen3_8b_block7` (`deepseek-ai/dspark_qwen3_8b_block7` at `03326e5`) |
| protocol | [`benchmarks/protocol-v2.1.json`](../benchmarks/protocol-v2.1.json), SHA256 `f8adddb463ffb268…` |
| panels | D48 `071befa6898b3187…`; K128b `cfd53af26e0cdcc2…`; GSM8K test set (openai/grade-school-math `3101c7d`) `3730d312f6e34405…` |
| tools | [`tools/report_v2.py`](../tools/report_v2.py) applies the protocol's pre-registered rules; [`tools/gain_factors.py`](../tools/gain_factors.py) splits DSpark's gain (§3); [`tools/plot_speedup.py`](../tools/plot_speedup.py) draws the README figure |

Each run writes one JSON file (schema `bench-v2`) with the protocol's SHA256, a manifest (GPU, driver, library versions, commit, node, and SHA256 of the installed packages and of the panel) and the GPU state before and after the run.

The DFlash results on K128 are in [v2/RESULTS.md](v2/RESULTS.md), and the September results of the 0.6B draft in [v1/RESULTS.md](v1/RESULTS.md).
