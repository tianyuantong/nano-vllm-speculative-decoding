"""CPU test environment: flash-attn and triton are GPU-only, so provide import-time stand-ins.

The engine modules import both at module level. On a machine without CUDA the stand-ins let
the pure-Python parts (scheduler, block manager, metadata, sampler, speculative decoder with
fake runners) be imported and tested; any attempt to run attention raises.
"""
import sys
import types


def _install_triton_stub() -> None:
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    language.constexpr = int
    language.dtype = type("dtype", (), {})   # transformers probes triton.language.dtype at import

    def jit(function=None, **kwargs):
        return function if function is not None else (lambda inner: inner)

    triton.jit = jit
    triton.language = language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language


def _install_flash_attn_stub() -> None:
    flash_attn = types.ModuleType("flash_attn")

    def unavailable(*args, **kwargs):
        raise RuntimeError("flash-attn is not available in the CPU test environment")

    flash_attn.flash_attn_varlen_func = unavailable
    flash_attn.flash_attn_with_kvcache = unavailable
    sys.modules["flash_attn"] = flash_attn


for module_name, install in (("triton", _install_triton_stub), ("flash_attn", _install_flash_attn_stub)):
    try:
        __import__(module_name)
    except ImportError:
        install()
