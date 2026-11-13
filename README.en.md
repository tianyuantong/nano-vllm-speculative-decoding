<div align="center">

# nano-vllm-speculative-decoding

Block speculative decoding for [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm): evolving from a 0.6B standalone draft to DFlash parallel blocks, DSpark's in-block Markov correction, and variable-length verification.

**Variable-length DSpark: single-request 2.02× · 8 concurrent 1.79× · 32 concurrent 1.48× · 64 concurrent 1.25×**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

[简体中文](README.md) | **English** · [Quick start](#quick-start) · [Results](docs/RESULTS.md) · [Usage guide](docs/DECODING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-overview-dark.svg">
  <img alt="Speedup over plain decoding on the same 128-prompt held-out test set (RTX PRO 6000 600 W, medians over 3 seeds): Qwen3-0.6B draft (k=3) reaches 1.50×, 1.21×, 1.00×, 0.85×; DFlash (k=7) reaches 1.98×, 1.66×, 1.31×, 1.02×; fixed-length DSpark (k=7) reaches 2.08×, 1.76×, 1.44×, 1.14× at 1, 8, 32, 64 concurrent requests." src="docs/assets/speedup-overview-light.svg" width="760">
</picture>

</div>

Measured on Qwen3-8B and a single RTX PRO 6000 (600 W), with standard rejection sampling preserving the target model's output distribution. The table below compares the three fixed-length draft generations; variable-length verification follows in the next section.

| Concurrent requests | 0.6B draft (k=3) | DFlash (k=7) | DSpark (k=7) | DSpark vs DFlash | Tokens / round (DFlash / DSpark) |
|---:|---|---|---|---|---|
| 1 | 1.50× | 1.98× | **2.08×** | 1.06× | 2.59 / 2.82 |
| 8 | 1.21× | 1.66× | **1.76×** | 1.06× | 2.53 / 2.81 |
| 32 | 1.00× | 1.31× | **1.44×** | 1.10× | 2.70 / 2.95 |
| 64 | 0.85× | 1.02× | **1.14×** | 1.12× | 2.65 / 2.93 |

Values are steady-state throughput multiples over plain decoding on the same 128-prompt held-out test set stratified from NVIDIA [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) (~1,000-token prompts spanning needle-in-a-haystack retrieval, code completion, book continuation, and creative writing; Colab G4 RTX PRO 6000, 600 W, medians over 3 seeds; shared plain-decoding baselines agree within 2× timing noise across jobs). The 0.6B standalone draft ([`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft)) and the variable-length table below use this same test set; earlier Nebius measurements and full per-seed ranges are in [Full results and protocol →](docs/RESULTS.md) · [0.6B baseline comparison →](docs/v2/RESULTS.md)

## Variable-length verification

In DSpark, target verification accounts for roughly three quarters of a round's duration and scales steeply with batch size: in the measured cost table, verifying 8 positions per request instead of 1 increases round time by 14%, 25%, and 49% at batch sizes 16, 32, and 64. With variable-length verification enabled, the engine ranks all candidate positions across the batch by predicted survival probability (DSpark Algorithm 1), admits positions while estimated throughput from the measured round-time table continues to rise, and rounds the total verified token count up to a captured CUDA Graph size—raising DSpark's speedup to **1.48×** (+2.2%) at 32 concurrent requests and **1.25×** (+8.9%) at 64.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-variable-dark.svg">
  <img alt="Speedup over plain decoding on the 128-prompt held-out test set (medians over 3 seeds). Fixed-length DSpark reaches 2.08×, 1.76×, 1.44×, 1.14× at 1, 8, 32, 64 concurrent requests; variable-length DSpark reaches 2.02×, 1.79×, 1.48×, 1.25×; 0.6B standalone draft reaches 1.50×, 1.21×, 1.00×, 0.85×." src="docs/assets/speedup-variable-light.svg" width="760">
</picture>

| Concurrent requests | 0.6B draft (k=3) | DSpark fixed-length | DSpark variable-length | Variable vs fixed | Positions verified per request per round |
|---:|---|---|---|---|---|
| 1 | 1.50× | 2.08× | 2.02× | −2.7% | 8.00 |
| 8 | 1.21× | 1.76× | 1.79× | +2.2% | 6.68 |
| 32 | 1.00× | 1.44× | **1.48×** | **+2.2%** | 4.62 |
| 64 | 0.85× | 1.14× | **1.25×** | **+8.9%** | 4.43 |

Medians over 3 seeds on the same 128-prompt held-out test set (Colab G4 RTX PRO 6000, 600 W). `Variable vs fixed` reports the median of per-seed paired gains (consistent positive gains across all 3 seeds at 32 and 64 concurrent requests; mixed seed directions at 8). [Variable-length results and protocol →](docs/RESULTS.md#8-variable-length-verification)

- **Verification savings at high concurrency**: At 64 concurrent requests, verification time drops from 36.51 ms to 27.96 ms per round and acceptance from 4.31 ms to 2.45 ms. Each round is 18.6% shorter while delivering only 5.6% fewer tokens, yielding a +16.1% throughput gain on decode rounds alone (+8.9% overall including prefill steps).
- **Single-request overhead**: At concurrency 1, every round verifies all 8 positions and produces token-identical output to fixed-length mode; the 2.7% overhead reflects confidence scoring and dynamic schedule lookup.

## Performance breakdown and comparisons

- **Block drafting eliminates serial autoregressive overhead.** A standalone 0.6B draft runs 3 sequential decode steps per round (~6 ms) and maintains a separate KV cache, breaking even with plain decoding at 32 concurrent requests (1.00×) and dropping to 0.85× at 64; a block draft proposes all 7 candidate tokens in a single forward pass.
- **DSpark raises acceptance via its Markov head.** It yields 7–12% more tokens per round than DFlash. At 32–64 concurrent requests, sampling within the top-k candidate window also avoids full-vocabulary softmax overhead (accounting for 2–6 percentage points of the lead in phase-timing breakdowns).
- **Speedup narrows at high concurrency.** Verification costs more per row as batch size grows, and prefill steps—which pause decoding in this engine—take 31% of DSpark's steady-state window at 32 concurrent requests.
- **Round-robin prefill bounds single-GPU stalls.** With `round_robin_prefill=True` (off by default), a step admits one request and a decode step follows it. On the same 128-prompt held-out test set, the longest gap between a request's deliveries drops from up to 1.1 s to 70–110 ms, and DSpark's inter-token latency (ITL) p99 at 32–64 concurrent requests falls to 0.58–0.73× of default scheduling with throughput at 0.97–1.00×. Arriving requests queue instead: the time to first token (TTFT) p99 of requests arriving mid-run rises to 1.0–3.8× ([results](docs/RESULTS.md#9-round-robin-prefill)).
- **Comparison with vLLM** (same machine, 48-prompt SPEED-Bench dev set, 5 seeds, end-to-end wall-clock throughput, vLLM `372477d`): on DSpark, this engine is 5.8–8.2% behind vLLM at 1–32 concurrent requests, with the extra time per round on the draft side; on plain decoding without speculation it is within 2% of vLLM ([results](docs/RESULTS.md#4-against-vllm-d48-wall-clock-toks)).

Three companion studies examine high-concurrency stalls, kernel floors, and shared prefixes:

| Direction | Key findings | Code & archives |
|---|---|---|
| **Dual-GPU 1P+1D disaggregation** | Decouples prefill and decode across 2 GPUs: across 8–64 concurrent requests, cuts whole-run ITL p99 to **0.15–0.19×** of 1-GPU default on plain decoding (13.3–23.6 ms) and **0.24–0.31×** on DSpark (31.6–81.4 ms, roughly half of 1-GPU round-robin prefill at 32–64 concurrent); at 32–64 concurrent requests, reaches **1.35–1.52×** and **1.38–1.39×** steady-state throughput (0.68–0.76× and 0.69–0.70× per GPU). | [`disagg/dual-gpu-1p1d` branch](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/disagg/dual-gpu-1p1d) · [Results §12](docs/RESULTS.md#12-dual-gpu-prefill-decode-disaggregation-1p1d) · [Archive](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-pd-disagg) |
| **Roofline & custom Triton kernels** | Plain decoding reaches 82–94% of reachable high-bandwidth memory (HBM) read bandwidth (1.52 TB/s) while 8-query verification drops to 39–61% without grouped-query attention (GQA) packing; custom Triton kernels for split-prefix verification and sparse-candidate chunked rejection shorten full-batch variable-length DSpark rounds by **10.6–13.9%** at 32–64 concurrent requests and by **31.4%** (73.5 → 50.5 ms) at 64 concurrent requests on ~3k-token contexts. | [Results §10](docs/RESULTS.md#10-speed-of-light-roofline-and-hand-written-triton-kernels) · [Roofline archive](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-verify-attention) · [Kernel archive](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-fused-kernels) |
| **Prefix caching interaction** | Across 512–2,048 shared prefix tokens (`exp/m6-prefix-cache`; `main` keeps `enable_prefix_cache=False` with a draft model), prefix caching raises steady-state throughput by **13–70%** on every line and cuts steady-state TTFT p50 to **51–59 ms**, while narrowing speculative speedup by 6–12% at 32–64 concurrent requests because L2-resident shared KV blocks accelerate 1-query decode more than 8-query verification. | [Results §11](docs/RESULTS.md#11-prefix-caching-with-speculative-decoding) · [Archive](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-prefix-cache) |

## Core implementation (four-stage evolution)

The `main` engine is under 3,000 lines of Python, built across four generations:
- **Stage 1 — 0.6B standalone draft (branch [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft))**: two separate models with independent block tables and KV caches; the 0.6B model runs k=3 autoregressive steps per round, the target verifies all positions in one fixed-length CUDA Graph pass, and a vectorized GPU sampler (`SpecSampler`) performs batched rejection sampling and KV rollback.
- **Stage 2 — DFlash parallel block draft**: replaces the standalone model with a single 7-query, non-causal pass through the decode attention path that reads 5 auxiliary target layers; a 5-layer draft KV cache shares the target's block table (`block_tables`) and stores context K/V for accepted positions after verification.
- **Stage 3 — DSpark in-block Markov correction**: adds a Markov head on top of the DFlash block features to draw the 7 candidate positions left to right, passing each position's bias-adjusted distribution to the rejection sampler. Three CUDA Graphs cover target verification, the draft block forward pass, and DSpark's 7 sampling steps.
- **Stage 4 — Variable-length verification**: uses DSpark's confidence head to compute each position's survival probability (the prefix product of calibrated confidences), admits positions across the batch while expected tokens divided by the measured round-time schedule keep rising, and pads the total verified token count up to a captured CUDA Graph size without breaking lossless rejection sampling.

Entry points: `SpeculativeDecoder.run_round` (five-step round pipeline), `VerifySchedule.verify_lengths`, `DFlashDrafter.propose`, `_markov_rows` and `after_verify`, and the decode branch of `Attention.forward`.

## Correctness verification

Candidates are verified via standard rejection sampling: accept $x$ when $u \lt \min(1, p_{\text{target}}(x) / p_{\text{draft}}(x))$, otherwise resample from $\text{normalize}(\max(p_{\text{target}} - p_{\text{draft}}, 0))$, guaranteeing the exact target output distribution (for DSpark, $p_{\text{draft}}$ is each position's distribution after its Markov bias).

Verified on GPU across four checks:
- Multi-step sampler output distribution (29 test cases)
- DSpark block and Markov-corrected logits against the DeepSpec reference implementation (0.59–1.08% relative error)
- Verification logits against step-by-step decoding (0.84 max absolute difference)
- GSM8K math reasoning accuracy (84.7% vs 85.2% plain decoding, −0.5 pt difference, 95% CI −1.7 to +0.8 pt)

Variable-length verification passes the sampler, verification-logit and GSM8K checks: 84.2% vs 85.1% plain decoding, −0.87 pt difference, 95% CI −2.14 to +0.40 pt.

On the DSpark paper's own prompts and sampling, the engine reproduces the tokens per round of the paper's Table 1: 4.10 for DFlash and 4.80 for DSpark, against 4.07 and 4.81 (9 sets, 3,030 prompts; [details](docs/RESULTS.md#53-tokens-per-round-against-the-dspark-paper)).

## Engineering pitfalls

- **In-place input overwrite in fp32**: `x.float()` returns the original tensor reference when `x` is already fp32, causing the upstream RMSNorm to overwrite its input in-place during fp32 numerical checks against DeepSpec (bf16 runs were unaffected).
- **`torch.compile` recompile fallback**: exceeding the default recompile limit (8) sent new batch shapes to the uncompiled path and broke bitwise identity between CUDA Graphs and eager execution; raising the limit to 64 restored parity.
- **vLLM baseline alignment**: vLLM (`372477d`) defaults to greedy draft sampling and requires routing DeepSpec's DFlash checkpoint through its `dspark` path to match the block layout; both sides were aligned to probabilistic sampling under the same distribution.
- **Unpacked GQA tiles in multi-query verification**: `flash_attn_with_kvcache` packs 4 query heads per KV head into one tile only when `q_len == 1`; verifying 8 candidate tokens per request drops out of the packed path and reduces verification attention to 39–61% of reachable HBM bandwidth ([details](docs/RESULTS.md#10-speed-of-light-roofline-and-hand-written-triton-kernels)).

## Quick start

Requires Python 3.10–3.12 and a CUDA GPU with enough memory for the target and draft models and their KV caches.

```bash
git clone https://github.com/tianyuantong/nano-vllm-speculative-decoding.git
cd nano-vllm-speculative-decoding
pip install torch triton flash-attn "transformers>=4.51" xxhash pytest
python -m pip install -e . --no-deps
```

Set the local model paths and generate:

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/models/Qwen3-8B",
    draft_model="/models/dspark_qwen3_8b_block7",    # deepseek-ai/dspark_qwen3_8b_block7
    draft_type="dspark",
    max_num_seqs=8,
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    seed=0,
)
params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=256)

try:
    outputs = llm.generate(["Explain KV caching."], params)
    print(outputs[0]["text"])
finally:
    llm.exit()
```

DSpark and DFlash propose a fixed block of 7 tokens per round (set by the checkpoint), so `num_speculative_tokens` is not needed. `kv_cache_memory_bytes` budgets the target's KV blocks; the 5-layer draft KV cache shares the same block table and is allocated outside that budget. For DFlash, pass `draft_model="/models/dflash_qwen3_8b_block7"` and `draft_type="dflash"`; for plain decoding, omit `draft_model`. The standalone 0.6B draft and dual-GPU 1P+1D disaggregation live on branches [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft) and [`disagg/dual-gpu-1p1d`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/disagg/dual-gpu-1p1d).
[Model configuration and API →](docs/DECODING.md)

## Reproduce

Fetch NVIDIA SPEED-Bench and build the prompt panels (`D48` = 48-prompt dev set; `K128` / `K128b` = first and second disjoint 128-prompt held-out test sets), then run plain decoding (`k0`) and the speculative lines (`dflash_k7`, `dspark_k7`) across concurrencies and seeds in one directory:

```bash
python -m pytest -q    # CPU tests, under 30 s
python tools/fetch_panels.py && python tools/build_panel_k128.py
for c in 1 8 32 64; do
  for s in 17011 110017 130031; do
    for line in k0 dflash_k7 dspark_k7; do
      python bench_spec.py --mode closed-loop --protocol benchmarks/protocol-v2.1.json --panel K128b \
          --line "$line" --concurrency "$c" --seed "$s" --models-root /path/to/models \
          --commit "$(git rev-parse --short HEAD)" --output "runs/${line}_c${c}_s${s}.json"
    done
  done
done
python tools/report_v2.py --protocol benchmarks/protocol-v2.1.json --results runs
```

For variable-length verification, pass `benchmarks/protocol-v2.3.json` and run lines `k0 dspark_k7 dspark_var` (line names, concurrencies, and seeds are defined in the protocol file). `tools/report_v2.py` summarizes throughput and speedup ratios, `tools/gain_factors.py` decomposes speedup gains by factor, and `tools/plot_speedup.py` generates the speedup bar charts. Raw result JSONs and prompt panels are packaged in [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark), [`results-variable-length`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-variable-length), and companion [Releases](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases).

## Explore the project

| Document | Contents |
|---|---|
| [Results](docs/RESULTS.md) | DSpark vs DFlash, variable-length verification, round-robin prefill, 2-GPU 1P+1D, prefix caching, hardware roofline & Triton kernels, vLLM comparison and correctness checks |
| [DFlash results](docs/v2/RESULTS.md) | DFlash vs the 0.6B standalone draft, per-round phase timings and prefill stalls |
| [Usage guide](docs/DECODING.md) | Model configuration, sampling options and timing API |
| [Tools](tools/README.md) | Measurement, verification and analysis scripts |
| [0.6B branch & design](docs/design/batched-speculative-decoding.md) | 0.6B draft architecture: dual-model execution, KV state, batching and CUDA Graphs |
| [0.6B v1 results](docs/v1/RESULTS.md) | Initial 0.6B draft release: batch-size scaling, phase timings and repeatability |
| [Optimization history](docs/PERFORMANCE.md) | Profiling the initial implementation and the bottlenecks that motivated the rewrite |

CPU tests run in [GitHub Actions](.github/workflows/cpu-tests.yml).

## Acknowledgments

Built on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). The DFlash block draft design is from [DFlash](https://github.com/z-lab/dflash) ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)), and DSpark is from DeepSeek ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)); both checkpoints and the reference implementation used for numerical verification are from [DeepSpec](https://github.com/deepseek-ai/DeepSpec). Base model: [Qwen3](https://huggingface.co/Qwen/Qwen3-8B); kernels: [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Triton](https://github.com/triton-lang/triton). The sampling algorithm follows Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023), and Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
[Upstream history →](docs/UPSTREAM.md)

## License

[MIT](LICENSE).
