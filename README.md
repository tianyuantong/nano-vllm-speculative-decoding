# nano-vLLM speculative decoding

An experimental fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

Adds offline random decoding, n-gram proposals and target/draft speculative decoding.
The dual-model path supports host or device token continuation with separate KV caches.

Scope: single GPU, temperature 1, no sampling filters, prefix cache disabled, draft length 1–4.
The existing `LLM` entrypoint still requires separate greedy and random batches.

Install: `pip install -e .` in a compatible CUDA environment.
Usage and GPU checks: [decoding guide](docs/DECODING.md).
CPU checks: `python tools/run_cpu_tests.py` (control flow and mathematical reference checks).

R1 reduced dual-model generation time by about 44% versus its previous implementation on eight requests;
ordinary decoding remained faster. See [results and checks](docs/DECODING.md#r1-results-and-checks).

This is a research implementation; finite-precision equivalence to ordinary decoding remains unproven.
See [upstream provenance](docs/UPSTREAM.md) and [MIT license](LICENSE).
