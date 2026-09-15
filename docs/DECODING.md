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

## Sampling fast paths and draft length

The opt-in fast paths reduce noise-validation masks, repeated softmax validation and
residual-probability temporaries. Probability arithmetic and RNG consumption retain their
existing contracts; invalid rows remain blocked from commit. Defaults are unchanged.

```python
llm = RandomLLM(target, draft, k=3, gpu_draft_tokens=True,
                r2_options=("draw", "softmax", "residual"))
```

Use the same `r2_options` on ordinary decoding for a fair comparison.
`pack` and `views` remain opt-in ablation controls in the measured implementation;
they are disabled in this candidate because the tested workload showed no useful benefit.

On eight development requests (two batches of four, seed 17011), this candidate took
10.729660 s versus 11.234017 s for ordinary decoding: 4.49% less generation time.
It generated 2,078 versus 1,977 tokens: 193.669 versus 175.983 token/s, 10.05% higher.
Draft length 3 was selected from 2/3/4 on these same inputs. The expanded retest below did not sustain this advantage.
Quality was not evaluated and finite-precision equivalence to ordinary decoding remains open.

CPU: `python tools/run_cpu_tests.py` and `python -m pytest -q tests/perf_repair`.
The updated GPU tools accept `--r2`; the model gate uses all five ablation options,
so it is not an exact replay of the three-option candidate above.
[Recorded results and replay materials](https://github.com/tianyuantong/serve-nano-vllm/releases/tag/r3-development-results)
are distributed separately from the code.

## Expanded fixed-k3 results

The frozen three-option, device-token k=3 candidate was retested against ordinary decoding
on two previously seen 24-request panels, excluding the original eight requests.
Each panel used three new seeds (110017, 130031, 170041), six batches of four and two
repetitions per mode: 72 workers and 144 timed calls. Engine, models and settings were unchanged.

| Panel | Ordinary / speculative time (s) | Ordinary / speculative tokens/s |
| --- | ---: | ---: |
| A1 | 112.715 / 115.702 | 210.682 / 203.315 |
| A2 | 83.531 / 85.816 | 262.538 / 250.139 |

Speculative generation took 2.65% / 2.74% longer, with 3.50% / 4.72% lower throughput.
Neither panel passed the predeclared criterion; these are previously seen inputs, not a blind test.
The eight-request result remains local evidence, not a general performance claim.
The cause of the workload-dependent difference has not been isolated; quality is unmeasured.

The plan stops here: subsequent LM-head graphs, probability graphs and dynamic draft length
remain unimplemented proposals. No additional optimization benefit is claimed.
[Full results, frozen protocol and audit tools](https://github.com/tianyuantong/serve-nano-vllm/releases/tag/expanded-k3-results)
are published as an experiment attachment, outside the code branch.
