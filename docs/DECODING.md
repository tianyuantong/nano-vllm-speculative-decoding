# Offline decoding

Use local model directories with identical target/draft tokenizers and vocabularies.
The measured setup used Qwen3-8B and Qwen3-0.6B, BF16, PyTorch 2.9.1+cu130 and FlashAttention 2.8.3.

```python
from nanovllm.config import Config
from nanovllm.engine.random_llm import RandomLLM

common = dict(max_num_seqs=4, max_model_len=3328,
              max_num_batched_tokens=5328, enable_prefix_cache=False)
target = Config("/path/to/Qwen3-8B", kv_cache_memory_bytes=4 << 30, **common)
draft = Config("/path/to/Qwen3-0.6B", kv_cache_memory_bytes=1536 << 20, **common)
llm = RandomLLM(target, draft, k=4)
try:
    requests = llm.prepare_requests(["Explain KV caching."], request_ids=["example"],
                                    seed=7, max_tokens=64, temperature=1.0)
    print(llm.generate(requests))
finally:
    llm.close()
```

Omit `draft` for ordinary decoding. Use `RandomLLM(target, ngram=True, k=4)` for n-gram proposals.
Set `gpu_draft_tokens=True` with a draft model for device token continuation.
KV budgets exclude weights, activations and graph memory; they are not a total-memory guarantee.

Run `python tools/run_cpu_tests.py` for independent CPU checks. Each file runs in its own
process because several tests replace Torch/model imports with test doubles.
These checks do not establish GPU numerics, physical KV contents or performance.
GPU scripts in `tests/` are separate, explicit acceptance tools; inspect their `--help`.

## R1 results and checks

R1 caches supported multi-position verification graphs, skips unused draft logits and
reduces host checks and probability temporaries. Disable with
`performance_mode=False, verify_graphs=False`; this is not a substitute for the frozen old implementation.

Run `python -m pytest -q tests/perf_repair` with Torch and pytest installed.
Controller tests compare token IDs, RNG states and forward events using a CPU test runner;
probability tests compare the frozen sampler on available devices. Other checks cover
metadata, routing and memory admission. This is not a universal bit-exactness proof.

For real-model acceptance on an allocated CUDA GPU:

```bash
PERF_REPAIR_REQUIRE_CUDA=1 python -m pytest -q tests/perf_repair
python tools/perf_repair_gpu_gate.py --target /path/to/Qwen3-8B --draft /path/to/Qwen3-0.6B --output /path/to/new-result.json
```

The recorded run used one RTX 5090, eight requests in two batches of four, seed 17011,
maximum 512 new tokens, natural EOS and two timed repetitions after warmup.
Generation time includes the complete `generate` call, excluding model loading and warmup.
R1 S0/S1 took 11.8848/11.6837 s versus 21.2985/20.8712 s before optimization;
ordinary decoding took 11.1995 s. Same-mode before/after outputs, stops and RNG states matched
in this recorded run. Only one seed was measured; task quality was not evaluated.

[Archived inputs, timings, validation and reproduction instructions](https://github.com/tianyuantong/serve-nano-vllm/tree/8796a52/docs/experiments/r1-20260914)
remain at the original fixed commit. Their source hashes describe that historical run,
not a new GPU validation of these reorganized branches. No new GPU run is implied.
