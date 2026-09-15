# Glossary of experiment codes

The PR descriptions, reports and docstrings use short codes for modes, stages
and panels. This table is the single place they are defined.

## Decoding modes

| Code | Meaning | Where |
|---|---|---|
| **B** | Ordinary random decoding through `RandomLLM(k=0)`: one target forward per token. This is the project baseline for every A/B comparison. It is **not** upstream `LLM.generate` — it uses the same per-request sampler as the speculative path and a drain-batch scheduler. | `random_decode.py`, `generate()` k=0 branch |
| **N** | n-gram proposals (prompt lookup): the draft tokens are copied from an earlier occurrence of the current suffix in the request's own history. Target model only. | `ngram.py`, `RandomLLM(ngram=True)` |
| **S0** | Dual-model speculation: Qwen3-0.6B proposes k tokens, Qwen3-8B verifies. Draft token ids are copied to the host after every draft step. | `RandomLLM(target, draft, k)` |
| **S1** | Same as S0, but draft tokens stay on the device between draft steps and are copied to the host once per round. | `RandomLLM(..., gpu_draft_tokens=True)` |
| **greedy** | Temperature-0 speculation: draft and target both take `argmax`; a draft token is accepted iff it equals the target's argmax at that position. | `greedy_backend.py`, `RandomLLM(sampling_mode="greedy")` |

## Terms inside a round

| Term | Meaning |
|---|---|
| **k** | Draft length: number of tokens proposed per round (1–4). |
| **q** | Number of uncached query tokens a request contributes to one forward. Draft steps and catch-up have q=1; VERIFY has q=k+1. |
| **VERIFY** | The single target forward over the k+1 positions `[last committed token, d₁ … dₖ]`, returning k+1 next-token distributions. |
| **post-p/q** | The deterministic part of verification after the target (p) and draft (q) distributions exist: acceptance test, residual distribution, correction draw and bonus draw. Captured as a CUDA graph in PR #7. |
| **catch-up** | The draft forward at the start of a round that recomputes the one draft KV entry left missing after a fully accepted round. Its q is always 1. |
| **eager fallback** | A VERIFY round whose batch has mixed query lengths. No captured graph matches, so the whole batch runs the eager (non-graph) forward — 3–4× the cost of a graph round. |
| **drain-batch** | The offline scheduler used by every measurement: a batch of ≤4 requests runs until all of them finish; finished slots are not refilled. Mean batch occupancy in the archives is 2.8–2.9 of 4. |
| **round** | One speculative iteration: catch-up + k draft steps + VERIFY + sampling + commit. Produces 1 + (accepted drafts) tokens per request. |
| **tokens/round** | Output tokens of the longest request divided by the number of rounds; equals 1 + mean accepted drafts for that request. |

## Stages and panels

| Code | Meaning |
|---|---|
| **R1** (PR #4) | First execution-overhead round: VERIFY CUDA graphs, KV-only draft forwards, fewer probability temporaries. |
| **R2** (PR #5) | Sampling fast-path ablation: `draw`, `softmax`, `residual`, `pack`, `views` switches measured one at a time. |
| **R3** (PR #5) | Draft-length selection, k ∈ {2, 3, 4}, with the three retained fast paths. |
| **R1 / R2 / R3** (PR #7) | Reused names for the *random* line of the last experiment: component timing of the post-p/q block, CUDA bridge checks, and the A1 + stress-group measurement. |
| **G1 / G2** (PR #7) | The *greedy* line: G1 = 2-group screen, G2 = A1 panel plus the fixed A2 stress group. |
| **A1** | Development panel: 24 previously seen requests in 6 groups of 4. |
| **A2** | Application panel: 24 requests in 6 groups of 4. The "fixed A2 stress group" is A2 group 4, where all four requests hit the 512-token cap. |
| **seed** | Base seed for the per-request RNG streams; the archives use 17011 (development) and 110017 / 130031 / 170041 (expanded retest). |

## Engine switches

| Switch | Values | Introduced |
|---|---|---|
| `enable_prefix_cache` | bool (must be `False` for the offline path) | PR #2 |
| `gpu_draft_tokens` | bool — S0 vs S1 | PR #3 |
| `ngram` | bool | PR #3 |
| `performance_mode` | bool — reduced validation, compact draws, merged host reads | PR #4 |
| `verify_graphs` | bool — capture VERIFY forwards as CUDA graphs | PR #4 |
| `r2_options` | subset of `{draw, softmax, residual, pack, views}` | PR #5 |
| `sampling_mode` | `"random"` / `"greedy"` | PR #7 |
| `verify_sampling` | `"off"` / `"eager"` / `"graph"` — post-p/q execution | PR #7 |
