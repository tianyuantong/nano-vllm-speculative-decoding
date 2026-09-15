"""CPU checks of actual S0 resource control flow, with no third-party imports.

Run: python3 tests/test_s0_resources.py
The torch/model substitutes below do not execute tensors, CUDA, NCCL or attention.
Passing these checks is not evidence of GPU capacity, KV contents or S0 decoding.
"""

import importlib
import pickle
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class TensorPlaceholder:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        return (self, key)


class ResourceHarness:
    """Substitute only external/compute boundaries; load project modules unchanged."""

    def __init__(self):
        self.events = []
        self.dtype = SimpleNamespace(itemsize=4, name="original")
        self.original_dtype = self.dtype
        self.device = "cpu"
        self.free = 1 << 30
        self.total = 2 << 30
        self.peak = 0
        self.current = 0
        self.distributed = False
        self.fail_weights = None
        self.fail_phase = None
        self.fail_sync = False
        self.models = []
        torch = ModuleType("torch")
        torch.Tensor = TensorPlaceholder
        torch.get_default_dtype = lambda: self.dtype
        torch.set_default_dtype = lambda value: setattr(self, "dtype", value)
        torch.get_default_device = lambda: self.device
        torch.set_default_device = lambda value: setattr(self, "device", value)
        torch.inference_mode = lambda: (lambda function: function)
        torch.empty = self.allocate_tensor
        torch.cuda = SimpleNamespace(
            set_device=lambda rank: self.events.append(("device", rank)),
            synchronize=self.synchronize,
            mem_get_info=lambda: (self.free, self.total),
            memory_stats=lambda: {"allocated_bytes.all.peak": self.peak,
                                  "allocated_bytes.all.current": self.current},
        )
        dist = ModuleType("torch.distributed")
        dist.is_initialized = lambda: self.distributed
        dist.init_process_group = self.init_group
        dist.destroy_process_group = self.destroy_group
        torch.distributed = dist
        modules = {"torch": torch, "torch.distributed": dist}
        for package in ("nanovllm", "nanovllm.engine", "nanovllm.models",
                        "nanovllm.layers", "nanovllm.utils"):
            module = ModuleType(package)
            module.__path__ = [str(ROOT.joinpath(*package.split(".")))]
            modules[package] = module
        self.stub(modules, "transformers", AutoConfig=type("AutoConfig", (), {}))
        self.stub(modules, "nanovllm.models.qwen3", Qwen3ForCausalLM=self.make_model)
        self.stub(modules, "nanovllm.layers.sampler", Sampler=lambda: object())
        self.stub(modules, "nanovllm.utils.loader", load_model=self.load_weights)
        self.modules = modules

    @staticmethod
    def stub(modules, name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module

    def init_group(self, backend, address, **kwargs):
        if self.distributed:
            raise RuntimeError("process group initialized twice")
        self.distributed = True
        self.events.append(("init", kwargs["rank"], kwargs["world_size"]))

    def destroy_group(self):
        if not self.distributed:
            raise RuntimeError("process group destroyed twice")
        self.distributed = False
        self.events.append(("destroy",))

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError("injected synchronization failure")

    def make_model(self, hf_config):
        layers = [SimpleNamespace(k_cache=None, v_cache=None)
                  for _ in range(hf_config.num_hidden_layers)]
        model = SimpleNamespace(label=hf_config.label, layers=layers,
                                modules=lambda: iter(layers))
        self.models.append(model)
        return model

    def load_weights(self, model, path):
        self.events.append(("weights", model.label))
        if model.label == self.fail_weights:
            raise RuntimeError("injected weights failure")

    def allocate_tensor(self, *shape):
        self.events.append(("allocate", shape))
        return TensorPlaceholder(shape)

    def compute_phase(self, runner, phase):
        self.events.append((phase, runner.config.model))
        if self.fail_phase == (phase, runner.config.model):
            raise RuntimeError("injected " + phase + " failure")

    @staticmethod
    def config(label="target", **overrides):
        hf = SimpleNamespace(label=label, dtype=SimpleNamespace(itemsize=2),
                             num_hidden_layers=2, num_key_value_heads=1,
                             hidden_size=2, num_attention_heads=1)
        values = dict(model=label, hf_config=hf, kvcache_block_size=256,
                      enforce_eager=False, tensor_parallel_size=1,
                      enable_prefix_cache=False, kv_cache_memory_bytes=8192,
                      gpu_memory_utilization=0.8, num_kvcache_blocks=-1,
                      max_num_seqs=4, max_model_len=3328,
                      max_num_batched_tokens=4096)
        values.update(overrides)
        return SimpleNamespace(**values)


class TestS0Resources(unittest.TestCase):
    def setUp(self):
        self.h = ResourceHarness()
        # patch.dict restores the caller's modules, including real torch if present.
        self.modules = patch.dict(sys.modules, self.h.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in list(sys.modules):
            if name.startswith("nanovllm.") and name not in self.h.modules:
                del sys.modules[name]
        self.runtime = importlib.import_module("nanovllm.engine.runtime")
        self.sequence = importlib.import_module("nanovllm.engine.sequence")
        self.kv = importlib.import_module("nanovllm.engine.kv_state")
        self.runner = importlib.import_module("nanovllm.engine.model_runner")
        self.dual = importlib.import_module("nanovllm.engine.dual_model")
        self.context = importlib.import_module("nanovllm.utils.context")
        for method, phase in (("warmup_model", "warmup"),
                              ("capture_cudagraph", "graph")):
            def compute(runner, phase=phase):
                self.h.compute_phase(runner, phase)
            mocked = patch.object(self.runner.ModelRunner, method, compute)
            mocked.start()
            self.addCleanup(mocked.stop)

    def assert_defaults_restored(self):
        self.assertIs(self.h.dtype, self.h.original_dtype)
        self.assertEqual(self.h.device, "cpu")

    def assert_group_released_once(self):
        self.assertFalse(self.h.distributed)
        self.assertEqual(self.h.events.count(("destroy",)), 1)

    def test_pair_loads_both_weights_before_kv_and_owns_one_group(self):
        pair = self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
        self.addCleanup(pair.exit)
        self.assertEqual(sum(event[0] == "init" for event in self.h.events), 1)
        first_allocation = next(i for i, event in enumerate(self.h.events)
                                if event[0] == "allocate")
        self.assertLess(self.h.events.index(("weights", "target")), first_allocation)
        self.assertLess(self.h.events.index(("weights", "draft")), first_allocation)
        self.assertIsNot(pair.target.kv_cache, pair.draft.kv_cache)
        self.assertIsNot(pair.target.model.layers[0].k_cache,
                         pair.draft.model.layers[0].k_cache)
        pair.exit()
        pair.exit()
        self.assert_group_released_once()
        self.assert_defaults_restored()

    def test_borrowed_runner_exit_does_not_destroy_other_runner_runtime(self):
        runtime = self.runtime.DeviceRuntime(0, 1)
        target = self.runner.ModelRunner(self.h.config(), 0, [], runtime=runtime,
                                         defer_cache=True)
        draft = self.runner.ModelRunner(self.h.config("draft"), 0, [],
                                        runtime=runtime, defer_cache=True)
        target.exit()
        self.assertTrue(self.h.distributed)
        with self.assertRaisesRegex(RuntimeError, "Release all"):
            runtime.close()
        draft.initialize_cache()
        draft.exit()
        runtime.close()
        self.assert_group_released_once()

    def test_foreign_process_group_is_rejected_without_destroying_it(self):
        self.h.distributed = True
        with self.assertRaisesRegex(RuntimeError, "explicit owner"):
            self.runtime.DeviceRuntime(0, 1)
        self.assertTrue(self.h.distributed)
        self.assertEqual(self.h.events, [])

    def test_runtime_rejects_mismatched_client_and_attach_after_close(self):
        runtime = self.runtime.DeviceRuntime(0, 1)
        with self.assertRaises(ValueError):
            runtime.attach(SimpleNamespace(rank=1, world_size=1))
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.attach(SimpleNamespace(rank=0, world_size=1))
        self.assert_group_released_once()

    def test_second_model_load_failure_releases_first_and_restores_defaults(self):
        self.h.fail_weights = "draft"
        with self.assertRaisesRegex(RuntimeError, "weights failure"):
            self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
        self.assert_group_released_once()
        self.assert_defaults_restored()
        self.assertFalse(any(event[0] == "allocate" for event in self.h.events))

    def test_cache_initialization_failures_release_pair_and_context(self):
        for model in ("target", "draft"):
            for phase in ("warmup", "graph"):
                with self.subTest(model=model, phase=phase):
                    self.h.events.clear()
                    self.h.fail_phase = (phase, model)
                    self.context.set_context(True, slot_mapping=object())
                    with self.assertRaisesRegex(RuntimeError, phase + " failure"):
                        self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
                    self.assert_group_released_once()
                    self.assert_defaults_restored()
                    self.assertIsNone(self.context.get_context().slot_mapping)

    def test_exit_releases_both_runners_even_when_synchronization_fails(self):
        pair = self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
        self.h.fail_sync = True
        with self.assertRaisesRegex(RuntimeError, "synchronization failure"):
            pair.exit()
        self.assertTrue(pair.target._exited)
        self.assertTrue(pair.draft._exited)
        self.assertFalse(hasattr(pair.target, "kv_cache"))
        self.assertFalse(hasattr(pair.draft, "model"))
        self.assert_group_released_once()

    def test_single_model_owner_failure_also_releases_group(self):
        self.h.fail_weights = "target"
        with self.assertRaisesRegex(RuntimeError, "weights failure"):
            self.runner.ModelRunner(self.h.config(), 0, [])
        self.assert_group_released_once()
        self.assert_defaults_restored()

    def test_original_construction_error_survives_cleanup_error(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                self.h.events.clear()
                self.h.fail_weights = "draft" if paired else "target"
                self.h.fail_sync = True
                with self.assertRaisesRegex(RuntimeError, "weights failure") as raised:
                    if paired:
                        self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
                    else:
                        self.runner.ModelRunner(self.h.config(), 0, [])
                self.assertIn("synchronization failure", str(raised.exception.__cause__))
                self.assert_group_released_once()
                self.assert_defaults_restored()

    def test_context_body_error_survives_cleanup_error(self):
        pair = self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
        self.h.fail_sync = True
        with self.assertRaisesRegex(ValueError, "body failure") as raised:
            with pair:
                raise ValueError("injected body failure")
        self.assertIn("synchronization failure", str(raised.exception.__cause__))
        self.assert_group_released_once()

    def test_group_release_can_be_retried_after_runner_references_are_dropped(self):
        runner = self.runner.ModelRunner(self.h.config(), 0, [])
        with patch.object(self.runtime.dist, "destroy_process_group",
                          side_effect=RuntimeError("injected group release failure")):
            with self.assertRaisesRegex(RuntimeError, "group release failure"):
                runner.exit()
        self.assertTrue(self.h.distributed)
        self.assertFalse(hasattr(runner, "model"))
        runner.exit()
        self.assert_group_released_once()

    def test_unsupported_pair_configuration_fails_before_device_setup(self):
        configurations = [
            (self.h.config(tensor_parallel_size=2), self.h.config("draft")),
            (self.h.config(enable_prefix_cache=True), self.h.config("draft")),
            (self.h.config(kv_cache_memory_bytes=None), self.h.config("draft")),
            (self.h.config(), self.h.config("draft", enforce_eager=True)),
            (self.h.config(), self.h.config("draft", max_model_len=1024)),
        ]
        same = self.h.config()
        configurations.append((same, same))
        for target, draft in configurations:
            with self.subTest(target=target, draft=draft):
                with self.assertRaises(ValueError):
                    self.dual.DualModelRunner(target, draft)
                self.assertEqual(self.h.events, [])

    def test_explicit_budget_floors_to_whole_blocks_independent_of_peak(self):
        runtime = self.runtime.DeviceRuntime(0, 1)
        config = self.h.config(kv_cache_memory_bytes=3 * 4096 + 4095)
        runner = self.runner.ModelRunner(config, 0, [], runtime=runtime,
                                         defer_cache=True)
        self.h.total = 10 << 30
        self.h.peak = 8 << 30
        self.h.current = 1 << 30
        runner.initialize_cache()
        self.assertEqual(config.num_kvcache_blocks, 3)
        self.assertEqual(runner.kv_cache.shape, (2, 2, 3, 256, 1, 2))
        with self.assertRaises(RuntimeError):
            runner.initialize_cache()
        runner.exit()
        runtime.close()

    def test_insufficient_free_memory_fails_before_tensor_allocation(self):
        self.h.free = 4096
        with self.assertRaisesRegex(MemoryError, "free device memory"):
            self.dual.DualModelRunner(self.h.config(), self.h.config("draft"))
        self.assertFalse(any(event[0] == "allocate" for event in self.h.events))
        self.assert_group_released_once()
        self.assert_defaults_restored()

    def test_automatic_budget_keeps_ordinary_runner_policy(self):
        self.h.total, self.h.free = 10 * 4096, 7 * 4096
        self.h.peak, self.h.current = 2 * 4096, 1 * 4096
        config = self.h.config(kv_cache_memory_bytes=None)
        runner = self.runner.ModelRunner(config, 0, [])
        # 80% of ten blocks minus three used, minus one extra warmup peak = four.
        self.assertEqual(config.num_kvcache_blocks, 4)
        runner.exit()
        self.assert_group_released_once()

    def test_budget_rejects_zero_incomplete_or_noninteger_bytes(self):
        for budget in (0, -1, 4095, True, 4096.0):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    self.kv.blocks_for_budget(budget, 4096)
        for block_bytes in (0, -1, True, 4096.0):
            with self.subTest(block_bytes=block_bytes):
                with self.assertRaises(ValueError):
                    self.kv.blocks_for_budget(8192, block_bytes)

    def test_config_rejects_invalid_explicit_budget_before_model_config_load(self):
        for budget in (0, -1, True, 4096.0):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.runner.Config(model=str(ROOT), kv_cache_memory_bytes=budget)

    def test_target_compatibility_aliases_do_not_mutate_draft(self):
        seq = self.sequence.Sequence([10, 20])
        seq.draft_kv = self.kv.KVState(1, [7])
        seq.num_cached_tokens = 2
        seq.block_table = [3]
        seq.block_table.append(4)
        seq.append_token(30)
        self.assertEqual(seq.target_kv, self.kv.KVState(2, [3, 4]))
        self.assertEqual(seq.draft_kv, self.kv.KVState(1, [7]))
        self.assertEqual(seq.num_tokens, 3)
        other = self.sequence.Sequence([40])
        self.assertEqual(other.block_table, [])
        self.assertIsNone(other.draft_kv)

    def test_ordinary_ipc_roundtrip_and_dual_kv_rejection(self):
        for prefill in (True, False):
            with self.subTest(prefill=prefill):
                seq = self.sequence.Sequence([10, 20])
                seq.is_prefill = prefill
                seq.num_cached_tokens = 1
                seq.block_table = [3]
                restored = pickle.loads(pickle.dumps(seq))
                self.assertEqual(restored.num_cached_tokens, 1)
                self.assertEqual(restored.block_table, [3])
                self.assertEqual(restored.last_token, 20)
                self.assertIsNone(restored.draft_kv)
                seq.draft_kv = self.kv.KVState()
                with self.assertRaisesRegex(ValueError, "TP=1"):
                    pickle.dumps(seq)


if __name__ == "__main__":
    unittest.main(verbosity=2)
