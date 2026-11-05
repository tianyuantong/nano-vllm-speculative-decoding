<div align="center">

# nano-vllm-speculative-decoding

Block speculative decoding for [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm): evolving from a 0.6B standalone draft to DFlash parallel blocks and DSpark's in-block Markov correction.

**Single-request 2.10× · 8 concurrent 1.78× · 32 concurrent 1.43× · 64 concurrent 1.15×**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**English** | [简体中文](README.zh.md) · [Quick start](#quick-start) · [Results](docs/RESULTS.md) · [Usage guide](docs/DECODING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-overview-dark.svg">
  <img alt="Speedup over plain decoding on 128-prompt held-out test sets (medians over 3 seeds). Left (DFlash vs 0.6B standalone draft): 0.6B reaches 1.54×, 1.21×, 0.96×, 0.82× at 1, 8, 32, 64 concurrent requests; DFlash reaches 2.19×, 1.80×, 1.40×, 1.08×. Right (DSpark vs DFlash): DFlash reaches 2.00×, 1.68×, 1.31×, 1.01×; DSpark reaches 2.10×, 1.78×, 1.43×, 1.15×." src="docs/assets/speedup-overview-light.svg" width="760">
</picture>

</div>

Measured on Qwen3-8B and a single RTX PRO 6000 (600 W), with standard rejection sampling preserving the target model's output distribution.

| Concurrent requests | 0.6B draft (k=3) | DFlash (k=7) | DSpark (k=7) | DSpark vs DFlash | Tokens / round (DFlash / DSpark) |
|---:|---|---|---|---|---|
| 1 | 1.54× | 2.00× | **2.10×** | 1.07× | 2.59 / 2.82 |
| 8 | 1.21× | 1.68× | **1.78×** | 1.06× | 2.53 / 2.81 |
| 32 | 0.96× | 1.31× | **1.43×** | 1.10× | 2.70 / 2.95 |
| 64 | 0.82× | 1.01× | **1.15×** | 1.14× | 2.65 / 2.93 |

Values are steady-state throughput multiples over plain decoding on 128-prompt held-out test sets (medians over 3 seeds). The 0.6B standalone draft ([`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft), 2.49–2.53 tokens/round) was measured on the first 128-prompt set (where DFlash reaches 2.19× / 1.80× / 1.40× / 1.08×, left panel); all other columns were measured back-to-back on the second 128-prompt set (right panel). [Full results and protocol →](docs/RESULTS.md) · [0.6B baseline comparison →](docs/v2/RESULTS.md)

## Performance breakdown and comparisons

- **Block drafting eliminates serial autoregressive overhead.** A standalone 0.6B draft runs 3 sequential decode steps per round (~6 ms) and maintains a separate KV cache, falling below plain decoding at 32 concurrent requests (0.96×); a block draft proposes all 7 candidate tokens in a single forward pass.
- **DSpark raises acceptance via its Markov head.** It yields 7–12% more tokens per round than DFlash. At 32–64 concurrent requests, sampling within the top-k candidate window also avoids full-vocabulary softmax overhead (accounting for 2–6 percentage points of the lead in phase-timing breakdowns).
- **Speedup narrows at high concurrency.** Verification costs more per row as batch size grows, and prefill steps—which pause decoding in this engine—take 31% of DSpark's steady-state window at 32 concurrent requests.
- **Comparison with vLLM** (48-prompt dev set, end-to-end wall-clock throughput, vLLM `372477d`): on DSpark, this engine is 7.6–7.9% behind vLLM at 1–32 concurrent requests; on plain decoding without speculation, it is 1–5% faster than vLLM.

## Core implementation (three-stage evolution)

The `main` engine is 2,570 lines of Python, built across three generations:
- **Stage 1 — 0.6B standalone draft (branch [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft))**: two separate models with independent block tables and KV caches; the 0.6B model runs k=3 autoregressive steps per round, the target verifies all positions in one fixed-length CUDA Graph pass, and a vectorized GPU sampler (`SpecSampler`) performs batched rejection sampling and KV rollback.
- **Stage 2 — DFlash parallel block draft**: replaces the standalone model with a single 7-query, non-causal pass through the decode attention path that reads 5 auxiliary target layers; a 5-layer draft KV cache shares the target's block table (`block_tables`) and stores context K/V for accepted positions after verification.
- **Stage 3 — DSpark in-block Markov correction**: adds a Markov head on top of the DFlash block features to draw the 7 candidate positions left to right, passing each position's bias-adjusted distribution to the rejection sampler. Three CUDA Graphs cover target verification, the draft block forward pass, and DSpark's 7 sampling steps.

Entry points: `SpeculativeDecoder.run_round` (five-step round pipeline), `DFlashDrafter.propose`, `_markov_rows` and `after_verify`, and the decode branch of `Attention.forward`.

## Correctness verification

Candidates are verified via standard rejection sampling: accept $x$ when $u \lt \min(1, p_{\text{target}}(x) / p_{\text{draft}}(x))$, otherwise resample from $\text{normalize}(\max(p_{\text{target}} - p_{\text{draft}}, 0))$, guaranteeing the exact target output distribution (for DSpark, $p_{\text{draft}}$ is each position's distribution after its Markov bias).

Verified on GPU across four checks:
- Multi-step sampler output distribution (29 test cases)
- DSpark block and Markov-corrected logits against the DeepSpec reference implementation (0.59–1.08% relative error)
- Verification logits against step-by-step decoding (0.84 max absolute difference)
- GSM8K math reasoning accuracy (84.7% vs 85.2% plain decoding, −0.5 pt difference, 95% CI −1.7 to +0.8 pt)

## Engineering pitfalls

- **In-place input overwrite in fp32**: `x.float()` returns the original tensor reference when `x` is already fp32, causing the upstream RMSNorm to overwrite its input in-place during fp32 numerical checks against DeepSpec (bf16 runs were unaffected).
- **`torch.compile` recompile fallback**: exceeding the default recompile limit (8) sent new batch shapes to the uncompiled path and broke bitwise identity between CUDA Graphs and eager execution; raising the limit to 64 restored parity.
- **vLLM baseline alignment**: vLLM (`372477d`) defaults to greedy draft sampling and requires routing DeepSpec's DFlash checkpoint through its `dspark` path to match the block layout; both sides were aligned to probabilistic sampling under the same distribution.

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

DSpark and DFlash propose a fixed block of 7 tokens per round (set by the checkpoint), so `num_speculative_tokens` is not needed. `kv_cache_memory_bytes` budgets the target's KV blocks; the 5-layer draft KV cache shares the same block table and is allocated outside that budget. For DFlash, pass `draft_model="/models/dflash_qwen3_8b_block7"` and `draft_type="dflash"`; for plain decoding, omit `draft_model`. The standalone 0.6B draft implementation and usage live on branch [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft).
[Model configuration and API →](docs/DECODING.md)

## Reproduce

`bench_spec.py` checks that the GPU matches `protocol["gpu"]["name"]` before running. Build the prompt panels first, then run plain decoding (`k0`) and the speculative lines across concurrencies and seeds in one directory:

```bash
python -m pytest -q    # CPU tests, ~6 s
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

`tools/report_v2.py` summarizes throughput and speedup ratios, `tools/gain_factors.py` decomposes speedup gains by factor, and `tools/plot_speedup.py` generates the speedup bar charts. Raw result JSONs and prompt panels for every table are packaged in the release asset [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark).

## Explore the project

| Document | Contents |
|---|---|
| [Results](docs/RESULTS.md) | DSpark vs DFlash, speedup breakdown, vLLM comparison and correctness checks |
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
