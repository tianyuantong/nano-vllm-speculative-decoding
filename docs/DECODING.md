# Using speculative decoding

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/Qwen3-0.6B",
    num_speculative_tokens=3,
    enable_prefix_cache=False,           # required with a draft model
    kv_cache_memory_bytes=10 << 30,      # explicit budgets: two models share the GPU
    draft_kv_cache_memory_bytes=6 << 30,
    max_num_seqs=8,
    seed=0,                              # optional: reproducible sampling
)
params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=256)
try:
    outputs = llm.generate(["Explain KV caching."], params)
    print(outputs[0]["text"])
finally:
    llm.exit()
```

For ordinary decoding leave out both `draft_model` and `num_speculative_tokens` (a `k > 0` without a draft is rejected); everything else is unchanged. Greedy
decoding is `temperature=0`. `top_p` requires `top_k` (the nucleus is computed over the
top-k candidates).

## What a round does

For every sequence in a decode batch: the draft recomputes at most one missing KV entry
(one decode-graph step), proposes `k` tokens (one decode-graph step each), the target
scores `[last token, d1..dk]` in one forward through the paged varlen-attention path (a
CUDA graph per padded batch size), rejection sampling accepts a prefix and draws one more
token, and a single `[B, k+2]` copy brings the result to the host. Round-entry invariants:
target KV covers `len-1` positions, draft KV covers `len-2` or `len-1`. The design is in
`docs/design/batched-speculative-decoding.md`.

## Reproducibility

One `torch.Generator` per engine, seeded from `seed`. A run is reproducible given the
seed and the same batch composition; which requests share a batch changes the draws.
The measured greedy outputs differ across batch sizes and between speculative and
ordinary decoding. Most first divergences occur at small top-2 logit margins; the
per-divergence results and identical-KV probe are in `docs/RESULTS.md`, gate G3 and its control.

## Limits (v1)

TP = 1; prefix cache off with a draft; no n-gram proposals; a batch is all-greedy or
all-random; `prompt + max_tokens + k <= max_model_len`.

## Timing

`record_step_timings=True` records a pair of CUDA events per engine step and one event
per phase of every speculative round; `llm.step_timings()` and `llm.phase_timings()`
return them after `generate()`. `bench_spec.py`, `tools/summarize_bench.py` and
`tools/report_matrix.py` build throughput, TTFT, TPOT, per-round costs and the
break-even check from them.

## Gates

`tools/gate_smoke.py`, `tools/gate_verify_logits.py` (G1: verification logits vs decode
logits), `tools/gate_sampler_distribution.py` (G2: sampler distribution),
`tools/gate_greedy_equivalence.py` (G3: greedy match rate with per-divergence logit
margins) and `tools/gate_divergence_probe.py` (decode vs verify logits on identical KV
at G3 divergence points) run on a CUDA machine with both models.
