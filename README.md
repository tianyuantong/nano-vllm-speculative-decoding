<div align="center">

# nano-vllm-speculative-decoding

Speculative decoding for [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm): a separate draft model, and the DFlash block draft.

**2.11–2.24× at C=1 · 1.75–1.85× at C=8 · 1.32–1.40× at C=32**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**English** | [简体中文](README.zh.md) · [Quick start](#usage) · [Results](docs/RESULTS.md) · [Design](docs/design/batched-speculative-decoding.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-k128-dark.svg">
  <img alt="Speedup over plain decoding on the K128 panel, median and range over 3 seeds. DFlash draft: 2.11–2.24× at 1 concurrent request, 1.75–1.85× at 8, 1.32–1.40× at 32 and 1.08× at 64. Qwen3-0.6B draft: 1.54×, 1.21×, 0.96× and 0.82×." src="docs/assets/speedup-k128-light.svg" width="760">
</picture>

</div>

DFlash draft on Qwen3-8B, RTX PRO 6000. Lossless in distribution (verified); GSM8K accuracy within −1.5…+0.9 pt.

Measured: steady-state tok/s against plain decoding in the same engine, on an unseen 128-prompt panel, 3 seeds per concurrency, median (min–max); 1.08× (1.06–1.09) at C=64 is within timing noise. The previous 0.6B draft: 1.54 / 1.21 / 0.96 / 0.82×. vLLM with the same weights on the same GPU (dev build `372477d`, dev panel, wall-clock): 155 vs 156 tok/s at C=1, 7–9% behind at C=8–32. [Protocol, gates, negative results →](docs/RESULTS.md)

| C | 0.6B draft (k=3) | DFlash (k=7) | DFlash ÷ 0.6B | tokens per round (0.6B / DFlash) |
|---:|---|---|---|---|
| 1 | 1.54× (1.54–1.57) | 2.11–2.24× | 1.37–1.45 | 2.52 / 2.85 |
| 8 | 1.21× (1.19–1.21) | 1.75–1.85× | 1.44–1.56 | 2.49 / 2.75 |
| 32 | 0.96× (0.95–0.96) | 1.32–1.40× | 1.38–1.48 | 2.53 / 2.82 |
| 64 | 0.82× (0.81–0.82) | 1.08× (1.06–1.09) | 1.29–1.33 | 2.53 / 2.82 |

The gain shrinks with C because verification cost per row climbs past about 128–256 rows, and prefill steps, which pause decoding in this engine, take 29% of the steady state at C=32. Use DFlash up to C=32; at C=64 it is no faster than plain decoding.

## What this adds

434 lines added, 107 removed, in 12 engine files on top of the 0.6B-draft release (the engine is now 2,627 lines):
- The DFlash draft block — the last committed token plus 6 mask tokens — runs as one 7-query, non-causal attention pass through the decode path.
- The target exports the hidden states (hidden + residual) of 5 layers during prefill and verification.
- Context K/V for all 8 verified positions are written right after verification; a rejected position is overwritten by the next block before anything reads it.
- The draft shares the target's block table and pad block, so there is no second KV allocator.
- Two CUDA graphs: verification (also exporting the 5 layers) and the draft block forward.

Key code: `SpeculativeDecoder.run_round` (the five steps of a round), `DFlashDrafter.propose` and `after_verify`, and the decode branch of `Attention.forward`.

## Correctness

Drafts are accepted by the standard rule: keep draft x when u < min(1, p_target(x) / p_draft(x)), otherwise resample from normalize(max(p_target − p_draft, 0)), so the output distribution is the target's. Checked on the GPU: the sampler's multi-step output distribution (29 cases), verification logits against step-by-step decoding (worst gap 0.75 / 0.84 against a bound of 3.0), and GSM8K accuracy (−1.5…+0.9 pt against plain decoding).

## Usage

Python 3.10–3.12 and a CUDA GPU with room for both models and their KV caches.

```bash
git clone https://github.com/tianyuantong/nano-vllm-speculative-decoding.git
cd nano-vllm-speculative-decoding
pip install torch triton flash-attn "transformers>=4.51" xxhash
python -m pip install -e . --no-deps
```

Set the local model paths and generate:

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/models/Qwen3-8B",
    draft_model="/models/dflash_qwen3_8b_block7",    # deepseek-ai/dflash_qwen3_8b_block7
    draft_type="dflash",
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

A DFlash draft proposes its block size (7) per round and keeps its context K/V in the target's blocks, so it needs no `num_speculative_tokens` or draft KV budget. For the 0.6B draft, pass `draft_model="/models/Qwen3-0.6B"`, `num_speculative_tokens=3` and `draft_kv_cache_memory_bytes`; for plain decoding, leave out `draft_model`.
[Model configuration and API →](docs/DECODING.md)

## Reproduce

```bash
python -m pytest -q    # CPU tests: 198 pass in about 6 s
python bench_spec.py --mode closed-loop --protocol benchmarks/protocol-v2.json --panel K128 --line dflash_k7 \
    --concurrency 8 --seed 17011 --commit <sha> --output result.json
```

The panels are built by `tools/fetch_panels.py` (D48) and `tools/build_panel_k128.py` (K128); `tools/report_v2.py` applies the protocol's verdicts and `tools/plot_speedup.py` draws the figure above.

## Explore the project

| Document | What's inside |
|---|---|
| [Results](docs/RESULTS.md) | Both panels, the vLLM comparison, the gates and where a round's time goes |
| [Design](docs/design/batched-speculative-decoding.md) | Draft/target execution, KV state, batching and CUDA Graphs |
| [Usage guide](docs/DECODING.md) | Model configuration, sampling options and timing API |
| [Tools](tools/README.md) | One line per measurement, gate and analysis script |
| [September results](docs/v1/RESULTS.md) | The 0.6B-draft release: batch-size comparisons, phase timings, repeatability |
| [Optimization history](docs/PERFORMANCE.md) | Profiling the first implementation and the bottlenecks behind the rewrite |

CPU tests run in [GitHub Actions](.github/workflows/cpu-tests.yml).

## Acknowledgments

Built on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). This repository adds batched speculative decoding with a separate draft model (the 0.6B line, released in September) and, for DFlash, the 7-query, non-causal decode mode for the DFlash draft block; the target's export of 5 layers' hidden + residual states; writing context K/V for every verified position and relying on the next block to overwrite rejected ones; a block table shared by draft and target; and two CUDA graphs, for verification and for the draft block.

The DFlash draft design is from [DFlash](https://github.com/z-lab/dflash) ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)); the checkpoint, and the reference implementation the port is checked against, are from [DeepSpec](https://github.com/deepseek-ai/DeepSpec). Models: [Qwen3](https://huggingface.co/Qwen/Qwen3-8B); kernels: [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Triton](https://github.com/triton-lang/triton). The sampling algorithm follows Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023), and Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
[Upstream history →](docs/UPSTREAM.md)

## License

[MIT](LICENSE).
