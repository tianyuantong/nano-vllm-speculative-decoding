# nano-vLLM experiments

A fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) for studying inference.

Adds greedy decoding (`temperature=0`) and an `enable_prefix_cache` switch.
Greedy and random requests must be run in separate batches.

Install with `pip install -e .` in a compatible CUDA environment.
Run `python -m unittest discover -s tests -p "test_serving.py"` for sampling checks,
and use `test_prefix_cache.py` for prefix-cache checks (same command pattern).
These checks import the CUDA engine and are not a CPU-only CI suite.

See [upstream provenance](docs/UPSTREAM.md) and the retained [MIT license](LICENSE).
