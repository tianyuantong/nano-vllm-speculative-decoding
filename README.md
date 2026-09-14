# nano-vLLM speculative decoding

An experimental fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

Adds offline random decoding, n-gram proposals and target/draft speculative decoding.
The dual-model path supports host or device token continuation with separate KV caches.

Scope: single GPU, temperature 1, no sampling filters, prefix cache disabled, draft length 1–4.
The existing `LLM` entrypoint still requires separate greedy and random batches.

Install: `pip install -e .` in a compatible CUDA environment.
Usage and GPU checks: [decoding guide](docs/DECODING.md).
CPU checks: `python tools/run_cpu_tests.py` (control flow and mathematical reference checks).

Expanded testing did not sustain the fixed-k=3 candidate's earlier eight-request speedup:
it took 2.65% and 2.74% more generation time than ordinary decoding on two 24-request panels.
See [results and scope](docs/DECODING.md#expanded-fixed-k3-results).

This is a research implementation; finite-precision equivalence to ordinary decoding remains unproven.
See [upstream provenance](docs/UPSTREAM.md) and [MIT license](LICENSE).
