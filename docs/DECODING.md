# Using speculative decoding

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/dflash_qwen3_8b_block7",   # deepseek-ai/dflash_qwen3_8b_block7
    draft_type="dflash",
    enable_prefix_cache=False,           # required with a draft
    kv_cache_memory_bytes=10 << 30,      # explicit target KV budget; the draft's context K/V mirrors its blocks
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

For standard autoregressive decoding, omit `draft_model` (passing `num_speculative_tokens > 0` without a draft model raises an error). Set `temperature=0` for greedy decoding. When using `top_p`, `top_k` must also be set because nucleus filtering is applied over the top-k candidates.

## DFlash block draft

As shown in the example above, `num_speculative_tokens` defaults to the draft checkpoint's block size (7) and is fixed to that value. `kv_cache_memory_bytes` sizes the target model's KV blocks; the draft model allocates its own 5-layer KV cache and shares the target's block table (`block_tables`). Each round runs a single forward pass of the draft model over a 7-token block (the last committed token followed by 6 mask tokens, using non-causal attention over context K/V projected from 5 target hidden layers), verifies all 7 candidate tokens in one target forward pass, writes the context K/V for all 8 verified positions, and applies rejection sampling.

## DSpark block draft

```python
llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/dspark_qwen3_8b_block7",   # deepseek-ai/dspark_qwen3_8b_block7
    draft_type="dspark",
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    max_num_seqs=8,
)
```

DSpark shares the block forward pass, KV layout, and target verification path with DFlash, and adds an in-block Markov head (`draft_type` must match the checkpoint). Instead of sampling all 7 draft positions independently, DSpark samples them sequentially from left to right: position `j` adds the Markov bias of the token sampled at position `j - 1` (or the last committed token for position `0`), and passes its post-bias probability distribution to the rejection sampler. The checkpoint's confidence head is loaded for variable-length verification; in fixed-length mode, all 7 draft tokens are verified each round.

When all active requests use top-k truncation, each draft position samples within its top `max_top_k + 64` logits rather than running softmax over the full vocabulary (matching full-vocabulary truncation unless more than 64 tokens tie at the cutoff). CUDA graphs are captured for up to 4 distinct sampling configurations (`top_k` width, whether `top_p` is enabled, and whether all requests truncate); batches with other configurations fall back to eager sampling (~0.4–0.5 ms slower per round at `C = 1`).

## Variable-length verification

By default, DSpark verifies all 8 positions (the anchor plus 7 draft tokens) for every request in the batch. Setting `variable_length=True` dynamically chooses the verification length per request based on confidence:

```python
llm = LLM(
    "/models/Qwen3-8B",
    draft_model="/models/dspark_qwen3_8b_block7",
    draft_type="dspark",
    variable_length=True,
    verify_schedule="benchmarks/inputs/verify-schedule-g4.json",
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    max_num_seqs=64,
)
```

In each round, the engine ranks all draft positions across the batch by their predicted survival probability (the prefix product of calibrated confidences along each sequence), admits positions while expected tokens divided by the measured round-time table keep increasing, and pads the total verified token count up to the next captured CUDA Graph size. Because the decision to verify position `i` depends only on positions ranked ahead of it—never on the sampled token at position `i` or later—rejection sampling remains strictly lossless.

The schedule JSON stores the empirical round-time table (indexed by batch size and verified token count) and the 7 temperature-scaling parameters for the confidence head. To profile a new GPU environment, run `tools/profile_verify_cost.py` followed by `tools/fit_confidence_sts.py`.

See [RESULTS.md](RESULTS.md#8-variable-length-verification) for benchmark results.

## Round-robin prefill

A prefill step runs no decoding in this engine, so each arriving request stalls the requests being decoded, and several arrivals stall them for several steps in a row. With `round_robin_prefill=True`, a step admits at most one waiting request, and a decode step follows each prefill step while requests are being decoded:

```python
llm = LLM(
    "/models/Qwen3-8B",
    draft_model="/models/dspark_qwen3_8b_block7",
    draft_type="dspark",
    round_robin_prefill=True,
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    max_num_seqs=64,
)
```

Between two of its deliveries a stream then waits for one prompt's prefill instead of every queued prompt's. The cost falls on arrivals: a request that arrives behind others waits one decode step for each request ahead of it.

The option is for requests that arrive over time (`add_request` / `step`). `generate()` has all its prompts up front and would admit them one at a time, so leave it off there. A decode step that would need a preemption is skipped: a half-prefilled request may hold the KV blocks it needs, and finishes first. `start_measurement(admit_together=True)` holds the option back until the waiting queue has emptied; the closed-loop benchmark uses it for the `C` requests it starts with.

See [RESULTS.md](RESULTS.md#9-round-robin-prefill) for its effect on inter-token latency.

## Execution flow of a speculative round

At the start of every round, both the target KV cache and the draft context KV cache hold entries up to `len - 1` for each sequence in the batch:

1. **Propose**: The draft model runs one block forward pass over the last committed token and `k - 1` mask tokens to propose `k` candidate tokens.
2. **Verify**: The target model scores `[last_token, d_1, ..., d_k]` in a single forward pass through paged varlen attention and exports the intermediate hidden states needed by the draft model.
3. **Context store**: The draft model projects and writes the context K/V for all `k + 1` verified positions into the shared block layout.
4. **Accept and sample**: Batched rejection sampling accepts a prefix of draft tokens and samples one additional token from the residual or bonus distribution.
5. **Commit**: A single `[B, k + 2]` device-to-host copy transfers the accepted tokens back to the scheduler.

Both the draft block forward pass and the target verification pass replay pre-captured CUDA graphs for each padded batch size. For the earlier dual-model architecture using a standalone Qwen3-0.6B draft, see [`docs/design/batched-speculative-decoding.md`](design/batched-speculative-decoding.md) and branch [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft).

## Reproducibility

Each engine instance maintains a single `torch.Generator` initialized from `seed`. Outputs are reproducible for a fixed seed and identical batch composition (changing which requests are batched together alters the random number consumption order). Under greedy decoding (`temperature=0`), floating-point reductions vary slightly across batch sizes and between single-token decode and multi-token verification; most first divergences occur when the top-2 logit margin is small (see [`docs/v1/RESULTS.md`](v1/RESULTS.md) for the G3 equivalence gate and logit-margin analysis).

## Current constraints

- Tensor parallelism is single-GPU (`TP = 1`) when speculative decoding is enabled.
- Prefix caching (`enable_prefix_cache=False`) must be disabled with a draft model.
- All sequences in a batch must share either greedy (`temperature=0`) or stochastic (`temperature > 0`) sampling.
- Sequence length must satisfy `prompt_len + max_tokens + k <= max_model_len`.

## Step and phase timing API

Passing `record_step_timings=True` to `LLM(...)` records a pair of CUDA events around each engine step and around each phase of a speculative round. Call `llm.step_timings()` and `llm.phase_timings()` after `generate()` to retrieve the recorded durations; `bench_spec.py` and `tools/summarize_bench.py` use these events to compute steady-state throughput and phase breakdowns.

## Verification gates

The GPU verification scripts under `tools/` check numerical alignment and sampling distributions:
- `tools/gate_verify_logits.py` (`G1-wide`): compares target verification logits against single-token decoding across batch sizes.
- `tools/gate_sampler_distribution.py` (`G2-real`): checks empirical acceptance rates and output distributions against the analytical rejection-sampling target.
- `tools/gate_task_accuracy.py` (`G5`): runs paired GSM8K evaluation against standard decoding.
- `tools/gate_dflash.py`: verifies DFlash and DSpark block logits against DeepSpec's reference implementation and checks bitwise identity between CUDA Graph and eager execution.

The v1 scripts for the standalone 0.6B draft are archived under `tools/v1/` and run against tag `results-batched-v1` (or branch `baseline/qwen3-0.6b-draft`).
