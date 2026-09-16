# nano-vLLM speculative decoding

An experimental fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

Adds offline random decoding, n-gram proposals and target/draft speculative decoding.
The dual-model path supports host or device token continuation with separate KV caches.

Scope: single GPU, temperature 1 (random) or 0 (greedy), no sampling filters, prefix cache disabled, draft length 1–4.
The existing `LLM` entrypoint still requires separate greedy and random batches.

Install: `pip install -e .` in a compatible CUDA environment.
Usage and GPU checks: [decoding guide](docs/DECODING.md).
CPU checks: `python tools/run_cpu_tests.py` (control flow and mathematical reference checks).

Adds offline greedy speculation and opt-in deterministic verification CUDA graphs.
The 15% throughput target was not met: A1 improved 12.66% (greedy) / 5.40% (random),
while the fixed stress group regressed. See [results and scope](docs/DECODING.md#greedy-and-verification-graphs)
and the [full versioned plans](docs/plans/README.md).

This is a research implementation; finite-precision equivalence to ordinary decoding remains unproven.
See [upstream provenance](docs/UPSTREAM.md) and [MIT license](LICENSE).
