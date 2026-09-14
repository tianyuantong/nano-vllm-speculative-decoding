"""R1 memory-gate regression: mocked CUDA accounting; no model/GPU required."""
from collections import Counter
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.random_decode import PrivateKVPool, Query, Request
from nanovllm.engine.verify_graph import VerifyGraphCache


def _cache():
    cache = VerifyGraphCache.__new__(VerifyGraphCache)
    cache.device = torch.device("cuda:0")
    cache.reserve_budget_bytes = 200
    cache.min_free_bytes = 100
    cache.reserved_at_enable = 1000
    cache.memory_checks = []
    cache.counts = Counter()
    cache.enabled = True
    cache.frozen = False
    cache.entries = {}
    cache.blocked = {}
    cache.max_batch = 4
    cache.owner_stream = 17
    cache.runner = SimpleNamespace(
        block_size=256,
        config=SimpleNamespace(max_model_len=3328, num_kvcache_blocks=113,
                               hf_config=SimpleNamespace(vocab_size=32)))
    return cache


def _memory(monkeypatch, before, after=None):
    # Tuples are (reserved, free, allocated); empty_cache is the only transition.
    state = {"value": before}
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (state["value"][1], 4000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: state["value"][0])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: state["value"][2])
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: calls.append("sync"))

    def reclaim():
        calls.append("empty_cache")
        if after is not None:
            state["value"] = after

    monkeypatch.setattr(torch.cuda, "empty_cache", reclaim)
    return calls


@pytest.mark.parametrize("before", [(1100, 300, 1000), (1200, 100, 1100)])
def test_healthy_and_exact_limits_do_not_reclaim(monkeypatch, before):
    c = _cache()
    calls = _memory(monkeypatch, before)
    assert c._memory_ok(key=(4, 2), phase="pre_capture")
    assert calls == []
    assert c.memory_checks[-1]["after_reclaim"] is None
    assert c.counts["memory_guard_reclaims"] == 0


@pytest.mark.parametrize("before,violation", [
    ((1300, 300, 1000), "reserved_growth_limit"),
    ((1100, 50, 1000), "device_free_floor"),
])
def test_reclaim_rechecks_same_limits(monkeypatch, before, violation):
    c = _cache()
    calls = _memory(monkeypatch, before, (1100, 300, 1000))
    assert c._memory_ok(key=(4, 2), phase="pre_capture")
    assert calls == ["sync", "empty_cache"]
    event = c.memory_checks[-1]
    assert violation in event["before"]["violations"]
    assert event["after_reclaim"]["violations"] == []
    assert event["before"]["reserved_limit_bytes"] == 1200
    assert event["after_reclaim"]["reserved_limit_bytes"] == 1200
    assert c.reserved_at_enable == 1000 and c.reserve_budget_bytes == 200


@pytest.mark.parametrize("phase", ["pre_capture", "post_capture"])
@pytest.mark.parametrize("value,violations", [
    ((1300, 300, 1250), ["reserved_growth_limit"]),
    ((1100, 50, 1000), ["device_free_floor"]),
    ((1300, 50, 1250), ["device_free_floor", "reserved_growth_limit"]),
])
def test_live_or_unreclaimable_pressure_still_rejected(monkeypatch, phase, value, violations):
    c = _cache()
    calls = _memory(monkeypatch, value)  # no memory released
    assert not c._memory_ok(key=(4, 2), phase=phase)
    assert calls == ["sync", "empty_cache"]
    assert c.counts["memory_guard_reclaims"] == 1
    assert c.memory_checks[-1]["after_reclaim"]["violations"] == violations
    assert c.memory_checks[-1]["phase"] == phase


def _queries():
    r = Request("a", [17] * 31, 8, None)
    pool = PrivateKVPool(113, 256)
    pool.reserve(r.target, 33)
    r.target.num_cached_tokens = 31
    return [Query(r, r.tokens + [18, 19], r.target, 256)]


@pytest.mark.parametrize("kind", ["frozen", "blocked", "replay"])
def test_hot_paths_do_not_poll_or_reclaim(monkeypatch, kind):
    c = _cache()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=17))

    def forbidden(*args, **kwargs):
        pytest.fail("a replay/frozen/blocked miss must not check/reclaim memory")

    monkeypatch.setattr(c, "_memory_ok", forbidden)
    if kind == "replay":
        output = torch.arange(64).reshape(2, 32)
        c.entries[(1, 2)] = SimpleNamespace(
            check_binding=lambda runner: None, update=lambda meta: None,
            graph=SimpleNamespace(replay=lambda: None), output=output)
        result = c.run(_queries())
        assert torch.equal(result[0], output)
        assert result[0].data_ptr() != output.data_ptr()
        assert c.counts["replays"] == 1
    else:
        c.frozen = kind == "frozen"
        if kind == "blocked":
            c.blocked[(1, 2)] = "pre_capture_memory_guard"
        assert c.run(_queries()) is None
    assert c.memory_checks == []


def test_pre_admission_failure_keeps_fallback_not_capture(monkeypatch):
    c = _cache()
    calls = _memory(monkeypatch, (1300, 300, 1250))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=17))
    assert c.run(_queries()) is None
    assert c.entries == {}
    assert c.blocked[(1, 2)] == "pre_capture_memory_guard"
    assert c.counts["resource_fallback"] == 1
    assert calls == ["sync", "empty_cache"]


def _gate():
    path = Path(__file__).resolve().parents[2] / "tools/perf_repair_gpu_gate.py"
    spec = importlib.util.spec_from_file_location("r1_memory_gate_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_host_snapshot_roundtrip_is_bitwise_and_does_not_use_gpu_gather(monkeypatch, dtype):
    gate = _gate()
    kv = torch.randn(2, 3, 5, 4, 2, 8).to(dtype)
    kv[0, 0, 4, 0, 0, 0] = float("nan")
    kv[1, 2, 1, 3, 1, 7] = -0.0
    original = kv.clone()
    blocks = [4, 1, 3]

    def forbidden(*args, **kwargs):
        pytest.fail("KV gate must not create a whole selected-block GPU temporary")

    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)
    monkeypatch.setattr(torch.Tensor, "index_copy_", forbidden)
    host = gate.snapshot_kv_cpu(kv, blocks)
    assert host.device.type == "cpu"
    assert host.shape == (2, 3, 3, 4, 2, 8)
    assert host.data_ptr() != kv.data_ptr()
    for i, block in enumerate(blocks):
        assert torch.equal(host.select(2, i).contiguous().view(torch.uint8),
                           original.select(2, block).contiguous().view(torch.uint8))
        kv.select(2, block).fill_(123)
    address = kv.data_ptr()
    gate.restore_kv_cpu(kv, blocks, host)
    assert kv.data_ptr() == address
    assert torch.equal(kv.view(torch.uint8), original.view(torch.uint8))


def test_bad_host_snapshot_or_block_fails_before_writing():
    gate = _gate()
    kv = torch.zeros(2, 2, 3, 4, 1, 8, dtype=torch.bfloat16)
    original = kv.clone()
    for blocks in ([], [0, 0], [-1], [3]):
        with pytest.raises(ValueError):
            gate.snapshot_kv_cpu(kv, blocks)
    host = gate.snapshot_kv_cpu(kv, [0, 2])
    with pytest.raises(ValueError):
        gate.restore_kv_cpu(kv, [0, 2], host.float())
    with pytest.raises(ValueError):
        gate.restore_kv_cpu(kv, [0], host)
    assert torch.equal(kv, original)
