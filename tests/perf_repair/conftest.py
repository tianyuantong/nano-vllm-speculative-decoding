"""CPU tests deliberately do not import the CUDA engine's eager __init__."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
for name, path in (("nanovllm", ROOT / "nanovllm"),
                   ("nanovllm.engine", ROOT / "nanovllm/engine"),
                   ("nanovllm.layers", ROOT / "nanovllm/layers"),
                   ("nanovllm.utils", ROOT / "nanovllm/utils")):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m


def reference(name):
    from importlib.machinery import SourceFileLoader
    path = Path(__file__).parent / "reference" / (name + ".py")
    ident = "perf_repair_reference_" + name
    if ident in sys.modules:
        return sys.modules[ident]
    loader = SourceFileLoader(ident, str(path))
    spec = importlib.util.spec_from_loader(ident, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[ident] = m
    loader.exec_module(m)
    return m

import os
if os.environ.get("PERF_REPAIR_REQUIRE_CUDA") == "1":
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA acceptance requested but CUDA is unavailable; do not count CPU-only as GPU PASS")
