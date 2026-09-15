"""Seed derivation/ownership metadata only; fake Generator does not sample."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
for name in ('nanovllm', 'nanovllm.engine', 'nanovllm.layers'):
    package = ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split('.')))]
    sys.modules[name] = package
fake = ModuleType('torch')
fake.Tensor = object
fake.inference_mode = lambda: (lambda f: f)
class Generator:
    def __init__(self, device): self.device = device
    def manual_seed(self, seed): self.seed = seed; return self
fake.Generator = Generator
sys.modules['torch'] = fake
m = importlib.import_module('nanovllm.engine.random_backend')

class RNGTests(unittest.TestCase):
    def test_stable_request_role_derivation(self):
        a = m.RequestRNG(1729, 'group02/request3'); b = m.RequestRNG(1729, 'group02/request3')
        self.assertEqual(a.seeds, b.seeds)
        self.assertEqual(len(set(a.seeds.values())), 5)
        self.assertTrue(all(0 <= x < 2**63 for x in a.seeds.values()))
        self.assertTrue(all(a.streams[k] is not b.streams[k] for k in m.ROLES))

    def test_neighbor_creation_does_not_change_request_streams(self):
        a = m.RequestRNG(1, 'a'); m.RequestRNG(1, 'b'); b = m.RequestRNG(1, 'a')
        self.assertEqual(a.seeds, b.seeds)
        self.assertNotEqual(a.seeds, m.RequestRNG(2, 'a').seeds)
        self.assertNotEqual(a.seeds, m.RequestRNG(1, 'b').seeds)

    def test_no_ambiguous_concatenation(self):
        self.assertNotEqual(m.role_seed(12, '3', 'draft'), m.role_seed(1, '23', 'draft'))
        self.assertEqual(m.role_seed(1729, '请求一', 'accept'), m.role_seed(1729, '请求一', 'accept'))

    def test_reject_invalid_seed_identity_role(self):
        for args in [(True, 'a', 'draft'), (1, '', 'draft'), (1, 2, 'draft'), (1, 'a', 'unknown')]:
            with self.assertRaises(ValueError): m.role_seed(*args)

if __name__ == '__main__': unittest.main(verbosity=2)
