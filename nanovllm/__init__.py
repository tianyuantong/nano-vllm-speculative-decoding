"""nano-vLLM with speculative decoding.

`LLM` is loaded on first access (PEP 562) so that importing a submodule — the scheduler, the
block manager, the samplers, the batch metadata — does not pull in the CUDA-only attention
dependencies (flash-attn, triton). `from nanovllm import LLM` still works unchanged.
"""
from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module 'nanovllm' has no attribute {name!r}")
