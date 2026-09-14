# nano-vLLM speculative decoding

An experimental fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

Adds offline random decoding, n-gram proposals and target/draft speculative decoding.
The dual-model path supports host or device token continuation with separate KV caches.

Scope: single GPU, temperature 1, no sampling filters, prefix cache disabled, draft length 1–4.
The existing `LLM` entrypoint still requires separate greedy and random batches.

Install: `pip install -e .` in a compatible CUDA environment.
Usage and GPU checks: [decoding guide](docs/DECODING.md).
CPU checks: `python tools/run_cpu_tests.py` (control flow and mathematical reference checks).

With three sampling fast paths and draft length 3, generation took 4.49% less time than ordinary decoding
on eight development requests. This is a selected configuration, not an independently confirmed result.
See [configuration and evidence](docs/DECODING.md#sampling-fast-paths-and-draft-length).

This is a research implementation; finite-precision equivalence to ordinary decoding remains unproven.
See [upstream provenance](docs/UPSTREAM.md) and [MIT license](LICENSE).
