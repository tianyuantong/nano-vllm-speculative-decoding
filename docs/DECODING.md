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
