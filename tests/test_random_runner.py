"""Actual ModelRunner query routing, mocked tensor/model boundary; no numerics."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_s0_resources import ResourceHarness

class Logits:
    def split(self, lengths): return ['all-rows', lengths]

class Model:
    def __init__(self): self.all_logits = None
    def __call__(self, ids, positions): return 'hidden'
    def compute_logits(self, hidden, *, all_logits):
        assert hidden == 'hidden'; self.all_logits = all_logits; return Logits()

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.h = ResourceHarness()
        self.p = patch.dict('sys.modules', self.h.modules); self.p.start()
        module = importlib.import_module('nanovllm.engine.model_runner')
        self.context = importlib.import_module('nanovllm.utils.context')
        self.r = module.ModelRunner.__new__(module.ModelRunner)
        self.r.world_size = 1
        self.r.verify_graph_cache = None
        self.r.config = SimpleNamespace(max_num_batched_tokens=8)
        self.r.model = Model()
        self.r.prepare_prefill = lambda queries: self.prepare(True)
        self.r.prepare_decode = lambda queries: self.prepare(False)
        self.r.run_model = lambda *args: Logits()
    def tearDown(self): self.p.stop()
    def prepare(self, flag):
        self.context.set_context(flag, max_seqlen_q=7)
        return 'ids', 'positions'
    def query(self, n, cached=0): return SimpleNamespace(num_scheduled_tokens=n, num_cached_tokens=cached)
    def test_verify_keeps_every_query_position(self):
        self.assertEqual(self.r.run_queries([self.query(3,4), self.query(2,7)], all_logits=True), ['all-rows', [3,2]])
        self.assertTrue(self.r.model.all_logits)
        self.assertEqual(self.context.get_context().max_seqlen_q, 0)
    def test_prefill_selects_last_position(self):
        self.assertEqual(self.r.run_queries([self.query(3), self.query(2)]), ['all-rows', 1])
        self.assertFalse(self.r.model.all_logits)
    def test_single_token_uses_decode_route(self):
        self.assertEqual(self.r.run_queries([self.query(1,3)], all_logits=True), ['all-rows', 1])
        self.assertIsNone(self.r.model.all_logits)
    def test_model_exception_resets_context(self):
        def fail(*args): raise RuntimeError('injected model error')
        self.r.model = fail
        with self.assertRaises(RuntimeError): self.r.run_queries([self.query(2)])
        self.assertEqual(self.context.get_context().max_seqlen_q, 0)
    def test_budget_fails_before_forward(self):
        with self.assertRaises(ValueError): self.r.run_queries([self.query(9)])
        self.assertIsNone(self.r.model.all_logits)

    def test_device_suffix_rejects_prefill_before_reading_placeholders(self):
        def forbidden(*args):
            self.fail('device suffix must not enter prefill')
        self.r.prepare_prefill = forbidden
        for queries in ([self.query(2,3)], [self.query(1,0)],
                        [self.query(1,3), self.query(2,4)]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                self.r.run_queries(queries, device_input_ids=object())
        self.assertEqual(self.context.get_context().max_seqlen_q, 0)

    def test_device_suffix_reaches_decode_input_unchanged(self):
        device_ids = object()
        seen = []
        def prepare(queries, supplied):
            self.assertIs(supplied, device_ids)
            return supplied, 'positions'
        def run(ids, positions, prefill):
            self.assertIs(ids, device_ids)
            self.assertFalse(prefill)
            seen.append(ids)
            return Logits()
        self.r.prepare_decode = prepare
        self.r.run_model = run
        self.r.run_queries([self.query(1,3)], device_input_ids=device_ids)
        self.assertEqual(len(seen), 1)

if __name__=='__main__': unittest.main(verbosity=2)
