"""Stand-ins for flash_attn and triton so engine modules import on a CPU-only machine.

Only nanovllm.layers.attention imports them. The stand-ins stay in sys.modules just long enough to
import that module: left there, transformers detects a triton installation and fails on attributes
the stand-in does not have. With the real packages installed (GPU machine) this module does nothing.
"""
import importlib
import importlib.util
import sys
import types


def _gpu_only(name: str):
    def call(*args, **kwargs):
        raise RuntimeError(f"{name} needs the GPU build of flash_attn")
    return call


_stubs = {}
if importlib.util.find_spec("flash_attn") is None:
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = _gpu_only("flash_attn_varlen_func")
    flash_attn.flash_attn_with_kvcache = _gpu_only("flash_attn_with_kvcache")
    _stubs["flash_attn"] = flash_attn
if importlib.util.find_spec("triton") is None:
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    _stubs["triton"] = triton
    _stubs["triton.language"] = language

if _stubs:
    sys.modules.update(_stubs)
    importlib.import_module("nanovllm.layers.attention")
    for name in _stubs:
        del sys.modules[name]
