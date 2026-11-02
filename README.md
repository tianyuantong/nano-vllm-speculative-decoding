<div align="center">

# nano-vllm-speculative-decoding

Speculative decoding for [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm): the DFlash block draft and DSpark's Markov-corrected blocks.

**2.10× at C=1 · 1.78× at C=8 · 1.43× at C=32 · 1.15× at C=64**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**English** | [简体中文](README.zh.md) · [Quick start](#usage) · [Results](docs/RESULTS.md) · [Usage guide](docs/DECODING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-overview-dark.svg">
  <img alt="Speedup over plain decoding across two held-out 128-prompt confirmation sets (medians over 3 seeds). Left (DFlash vs 0.6B draft): Qwen3-0.6B reaches 1.54×, 1.21×, 0.96×, 0.82× at C=1, 8, 32, 64; DFlash reaches 2.19×, 1.80×, 1.40×, 1.08×. Right (DSpark vs DFlash): DFlash reaches 2.00×, 1.68×, 1.31×, 1.01×; DSpark reaches 2.10×, 1.78×, 1.43×, 1.15×." src="docs/assets/speedup-overview-light.svg" width="760">
</picture>

</div>

Measured on Qwen3-8B and an RTX PRO 6000 (600 W). Standard rejection sampling preserves the target's output distribution (GSM8K accuracy −0.5 pt vs plain decoding, 95% CI −1.7…+0.8 pt).

| C | Set A: 0.6B (k=3) | Set A: DFlash (k=7) | Set A: DFlash ÷ 0.6B | Set B: DFlash (k=7) | Set B: DSpark (k=7) | Set B: DSpark ÷ DFlash | Set B: tokens/round (DFlash / DSpark) |
|---:|---|---|---|---|---|---|---|
| 1 | 1.54× | 2.19× | 1.41 | 2.00× | 2.10× | 1.07 | 2.59 / 2.82 |
| 8 | 1.21× | 1.80× | 1.48 | 1.68× | 1.78× | 1.06 | 2.53 / 2.81 |
| 32 | 0.96× | 1.40× | 1.46 | 1.31× | 1.43× | 1.10 | 2.70 / 2.95 |
| 64 | 0.82× | 1.08× | 1.31 | 1.01× | 1.15× | 1.14 | 2.65 / 2.93 |

Steady-state tok/s relative to plain decoding in the same engine and job, on two held-out 128-prompt confirmation sets (medians over 3 seeds per concurrency; Set A tokens/round are 2.49–2.53 for 0.6B and 2.75–2.85 for DFlash). Set A (`K128`, commit `10b38b4`; standalone 0.6B retained on [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft)) and Set B (`K128b`, commit `f30ed0a`) use different prompt draws, engine versions and cloud instances, so comparisons are strictly within each set; at C=64 on Set B, DFlash is essentially level with plain decoding while DSpark stays faster. [Set B ranges, protocol and gates →](docs/RESULTS.md) · [Set A 0.6B comparison →](docs/v2/RESULTS.md)

## Where the speedup comes from

- **Block drafting removes serial draft steps.** A standalone 0.6B draft runs 3 sequential decode steps per round and maintains its own KV cache; a block draft proposes all 7 tokens in a single forward pass.
- **DSpark leads on acceptance.** Its Markov head raises tokens per round by 7–12% over DFlash. At C=32–64, sampling within the top-k candidate window also avoids full-vocabulary softmax overhead (accounting for an estimated 2–6 points of the lead, inferred without a same-sampling DFlash run).
- **The gain shrinks with C.** Verification costs more per row at high concurrency, and prefill steps, which pause decoding in this engine, take 31% of DSpark's steady state at C=32.
- **Against vLLM's DSpark** (same weights, same GPU model on the previous instance where plain decoding differed by <0.5%, dev build `372477d`, 48-prompt dev set, wall-clock): 7.6–7.9% behind at C=1–32. Without speculation, nano is 1–5% faster than vLLM.

## What this adds

The engine is now 2,570 lines, grown out of the September 0.6B-draft release ([`results-batched-v1`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/results-batched-v1)):
- **A DFlash block draft**: one 7-query, non-causal pass through the decode path, reading 5 target layers, with a separate 5-layer draft KV cache that shares the target's block table (`block_tables`) to store context K/V for every verified position. 12 engine files, +434/−107.
- **DSpark on top of it**: a Markov head draws the 7 rows left to right, and each row hands the acceptance rule the distribution it was drawn from. 4 engine files, +224/−12.
- **CUDA graphs** for verification, the draft block and DSpark's 7 sampling steps.

Key code: `SpeculativeDecoder.run_round` (the five steps of a round), `DFlashDrafter.propose`, `_markov_rows` and `after_verify`, and the decode branch of `Attention.forward`.

## Correctness

Drafts are accepted by the standard rule: keep draft x when u < min(1, p_target(x) / p_draft(x)), otherwise resample from normalize(max(p_target − p_draft, 0)), so the output distribution is the target's; for DSpark, p_draft is each row's distribution after its Markov bias. Checked on the GPU: the sampler's multi-step output distribution (29 cases), DSpark's block and Markov-corrected logits against DeepSpec (relative error 0.59–1.08%, bound 2%), verification logits against step-by-step decoding (worst gap 0.84 against a bound of 3.0), and GSM8K accuracy (84.7% vs 85.2% plain decoding, difference −0.5 pt, 95% CI −1.7…+0.8 pt, gate bound −3.0 pt).

## Pitfalls we hit

- `x.float()` returns the same tensor when `x` is already fp32, so the upstream RMSNorm overwrote its input in fp32 runs. bf16 runs were unaffected; it surfaced when the draft was checked against DeepSpec in fp32.
- torch.compile's default recompile limit (8) sent new shapes to the uncompiled path, which broke graph-against-eager bit identity until the limit was raised to 64.
- A fair vLLM baseline needed its draft sampling switched from greedy to probabilistic, and DeepSpec's DFlash checkpoint run through vLLM's `dspark` path, whose block layout matches it (vLLM `372477d`).

## Usage

Python 3.10–3.12 and a CUDA GPU with room for both models and their KV caches.

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

A DSpark or DFlash draft proposes its block size (7) per round and shares the target's block table for its own 5-layer KV cache, so it needs no `num_speculative_tokens` parameter (note that `kv_cache_memory_bytes` sizes the target's KV blocks; the draft KV cache is allocated in addition to that budget); `draft_type` must match the checkpoint. For DFlash, pass `draft_model="/models/dflash_qwen3_8b_block7"` and `draft_type="dflash"`; for plain decoding, leave out `draft_model`. The 0.6B draft runs on branch `baseline/qwen3-0.6b-draft`.
[Model configuration and API →](docs/DECODING.md)

## Reproduce

`bench_spec.py` verifies that the GPU matches `protocol["gpu"]["name"]` before running. Build the prompt panels first, then run the baseline (`k0`) and speculative lines across concurrencies and seeds in one directory:

```bash
python -m pytest -q    # CPU tests, about 6 s
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

`tools/report_v2.py` applies the protocol's verdicts, `tools/gain_factors.py` splits one line's gain over another into its factors, and `tools/plot_speedup.py` draws single-panel speedup charts (`docs/assets/speedup-k128*`). The result files behind every table, with the gates' outputs and the panels, are the release asset [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark).

## Explore the project

| Document | What's inside |
|---|---|
| [Results](docs/RESULTS.md) | DSpark against DFlash, where the lead comes from, the vLLM comparison and the gates |
| [DFlash results](docs/v2/RESULTS.md) | DFlash against the 0.6B draft, a round's phases and the prefill stall |
| [Usage guide](docs/DECODING.md) | Model configuration, sampling options and timing API |
| [Tools](tools/README.md) | One line per measurement, gate and analysis script |
| [September design](docs/design/batched-speculative-decoding.md) | The 0.6B-draft release: draft/target execution, KV state, batching and CUDA Graphs |
| [September results](docs/v1/RESULTS.md) | The 0.6B-draft release: batch-size comparisons, phase timings, repeatability |
| [Optimization history](docs/PERFORMANCE.md) | Profiling the first implementation and the bottlenecks behind the rewrite |

CPU tests run in [GitHub Actions](.github/workflows/cpu-tests.yml).

## Acknowledgments

Built on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). This repository adds batched speculative decoding with a separate draft model, released in September, and everything under [What this adds](#what-this-adds). The DFlash draft design is from [DFlash](https://github.com/z-lab/dflash) ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)), and DSpark from DeepSeek ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)); both checkpoints, and the reference implementation the port is checked against, are from [DeepSpec](https://github.com/deepseek-ai/DeepSpec). Models: [Qwen3](https://huggingface.co/Qwen/Qwen3-8B); kernels: [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Triton](https://github.com/triton-lang/triton). The sampling algorithm follows Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023), and Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
[Upstream history →](docs/UPSTREAM.md)

## License

[MIT](LICENSE).
