"""Actual controller + independent physical-slot prefix oracle; no Torch/GPU."""
import importlib
from pathlib import Path
import random
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
for name in ('nanovllm', 'nanovllm.engine'):
    package = ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split('.')))]
    sys.modules[name] = package
module = importlib.import_module('nanovllm.engine.random_decode')
Request, RandomDecode = module.Request, module.RandomDecode


class PrefixOracle:
    def __init__(self, reject=None, proposal_eos=False, first_eos=False, fail=None):
        self.memory = {'target': {}, 'draft': {}}
        self.events = []
        self.reject = reject
        self.proposal_eos = proposal_eos
        self.first_eos = first_eos
        self.fail = fail
        self.rounds = 0
        self.draw_steps = 0

    def forward(self, role, queries, all_logits=False):
        if self.fail == role:
            raise RuntimeError('injected forward failure')
        for q in queries:
            slots = [q.block_table[i // q.block_size] * q.block_size + i % q.block_size for i in range(len(q))]
            actual = [self.memory[role][s] for s in slots[:q.num_cached_tokens]]
            if actual != q.tokens[:q.num_cached_tokens]:
                raise AssertionError(('wrong visible KV', role, actual, q.tokens))
            self.events.append((role, q.request.request_id, q.num_cached_tokens, len(q), tuple(q.tokens)))
            for i in range(q.num_cached_tokens, len(q)):
                self.memory[role][slots[i]] = q.tokens[i]
        return [list(range(q.num_scheduled_tokens)) if all_logits else [0] for q in queries]

    def sample_batch(self, requests, logits, role):
        return [99 if self.first_eos else 10 for _ in requests]

    def propose(self, requests, logits):
        self.draw_steps += 1
        return [99 if self.proposal_eos else 20 + self.draw_steps % 10 for _ in requests], [object() for _ in requests]

    def verify(self, requests, logits, proposals, probabilities):
        if self.fail == 'verify':
            raise FloatingPointError('injected boundary error')
        self.rounds += 1
        outputs = []
        for r, p in zip(requests, logits):
            ds = proposals[r.request_id]
            assert len(p) == len(ds) + 1
            assert len(probabilities[r.request_id]) == len(ds)
            n = len(ds) if self.reject is None else min(self.reject, len(ds))
            outputs.append(ds[:n] + [30])
        return outputs

    def synchronize(self):
        self.events.append(('sync',))


def engine(backend, **kwargs):
    options = dict(target_blocks=100, draft_blocks=100, block_size=4,
                   max_model_len=80, max_num_seqs=4, vocab_size=100, eos=99, k=3)
    options.update(kwargs)
    return RandomDecode(backend, **options)


def request(name='a', prompt=None, cap=12):
    return Request(name, [1, 2, 3, 4] if prompt is None else prompt, cap, object())


class StateTests(unittest.TestCase):
    def assert_released(self, e, requests):
        self.assertFalse(e.target_pool.used)
        if e.draft_pool:
            self.assertFalse(e.draft_pool.used)
        for r in requests:
            self.assertEqual((r.target.num_cached_tokens, r.draft.num_cached_tokens), (0, 0))
            self.assertFalse(r.target.block_table or r.draft.block_table)

    def test_full_acceptance_computes_missing_tail(self):
        b = PrefixOracle(); e = engine(b); r = request()
        out = e.generate([r])
        # m=4, h=5, k=3 => target 8, draft 7; catch-up computes [7:8].
        self.assertTrue(any(x[:4] == ('draft', 'a', 7, 8) for x in b.events))
        self.assertEqual(len(out[0]['token_ids']), 12)
        self.assert_released(e, [r])

    def test_each_rejection_position_and_many_rounds(self):
        for reject in (0, 1, 2, 3):
            with self.subTest(reject=reject):
                b = PrefixOracle(reject=reject); e = engine(b); rs = [request(cap=25)]
                self.assertEqual(len(e.generate(rs)[0]['token_ids']), 25)
                self.assert_released(e, rs)

    def test_full_accept_then_last_token_skips_draft_catchup(self):
        b = PrefixOracle(); e = engine(b); r = request(cap=6)
        out = e.generate([r])[0]
        self.assertEqual(out['token_ids'], [10, 21, 22, 23, 30, 30])
        self.assertEqual(out['stop_reason'], 'length')
        # Draft prefill + three proposal forwards only; no [7:8] catch-up.
        self.assertEqual([x[2:4] for x in b.events if x[0] == 'draft'],
                         [(0, 4), (4, 5), (5, 6), (6, 7)])
        self.assert_released(e, [r])

    def test_last_token_row_skips_catchup_while_neighbor_continues(self):
        b = PrefixOracle(); e = engine(b)
        rs = [request('short', cap=6), request('long', cap=10)]
        out = e.generate(rs)
        self.assertEqual(out[0]['token_ids'], [10, 21, 22, 23, 30, 30])
        self.assertEqual(len(out[1]['token_ids']), 10)
        self.assertFalse(any(x[:4] == ('draft', 'short', 7, 8) for x in b.events))
        self.assertTrue(any(x[:4] == ('draft', 'long', 7, 8) for x in b.events))
        self.assert_released(e, rs)

    def test_first_eos_skips_draft_prefill(self):
        b = PrefixOracle(first_eos=True); e = engine(b); rs = [request()]
        self.assertEqual(e.generate(rs)[0]['token_ids'], [99])
        self.assertFalse(any(x[0] == 'draft' for x in b.events))
        self.assert_released(e, rs)

    def test_cap_one_skips_draft_prefill(self):
        b = PrefixOracle(); e = engine(b); rs = [request(cap=1)]
        self.assertEqual(e.generate(rs)[0]['token_ids'], [10])
        self.assertFalse(any(x[0] == 'draft' for x in b.events))

    def test_accepted_eos_discards_bonus_but_consumes_fixed_draft_budget(self):
        b = PrefixOracle(proposal_eos=True); e = engine(b); rs = [request()]
        self.assertEqual(e.generate(rs)[0]['token_ids'], [10, 99])
        self.assertEqual(b.draw_steps, 3)
        self.assert_released(e, rs)

    def test_rejected_eos_does_not_stop(self):
        b = PrefixOracle(proposal_eos=True, reject=0); e = engine(b); rs = [request(cap=6)]
        out = e.generate(rs)[0]
        self.assertNotIn(99, out['token_ids'])
        self.assertEqual(out['stop_reason'], 'length')
        self.assert_released(e, rs)

    def test_ignore_eos(self):
        b = PrefixOracle(proposal_eos=True); e = engine(b); r = request(cap=7); r.ignore_eos = True
        out = e.generate([r])[0]
        self.assertEqual(len(out['token_ids']), 7)
        self.assertIn(99, out['token_ids'])

    def test_context_budget_and_remaining_one(self):
        b = PrefixOracle(); e = engine(b, max_model_len=6); r = request(cap=20)
        out = e.generate([r])[0]
        self.assertEqual(out['stop_reason'], 'context')
        self.assertEqual(len(out['token_ids']), 2)
        self.assertEqual(b.draw_steps, 0)

    def test_mixed_batch_ends_and_reuses_physical_blocks(self):
        b = PrefixOracle(reject=1); e = engine(b)
        for repeat in range(3):
            rs = [request(str(i), list(range(1, 3+i)), cap) for i, cap in enumerate([1, 4, 13, 19])]
            self.assertEqual([len(o['token_ids']) for o in e.generate(rs)], [1, 4, 13, 19])
            self.assert_released(e, rs)

    def test_ordinary_random_path_uses_no_draft(self):
        b = PrefixOracle(); e = engine(b, k=0); rs = [request(cap=8)]
        self.assertEqual(e.generate(rs)[0]['token_ids'], [10]*8)
        self.assertFalse(any(x[0] == 'draft' for x in b.events))
        self.assert_released(e, rs)

    def test_capacity_failure_releases_partial_reservation_and_poison(self):
        b = PrefixOracle(); e = engine(b, target_blocks=1); rs = [request('a'), request('b')]
        with self.assertRaises(MemoryError): e.generate(rs)
        self.assert_released(e, rs)
        with self.assertRaises(RuntimeError): e.generate([request()])

    def test_failure_in_target_draft_or_commit_boundary_aborts(self):
        for fail in ('target', 'draft', 'verify'):
            b = PrefixOracle(fail=fail); e = engine(b); rs = [request()]
            with self.assertRaises((RuntimeError, FloatingPointError)): e.generate(rs)
            self.assertTrue(e.poisoned)
            self.assert_released(e, rs)

    def test_invalid_input_rejected_before_embedding(self):
        for prompt, cap in [([], 5), ([100], 5), ([-1], 5), ([1], 0)]:
            b = PrefixOracle(); e = engine(b)
            with self.assertRaises(ValueError): e.generate([request(prompt=prompt, cap=cap)])
            self.assertFalse(any(x[0] == 'target' for x in b.events))

    def test_physical_prefix_property_varied_shapes(self):
        rng = random.Random(127)
        for _ in range(100):
            b = PrefixOracle(reject=rng.choice([None, 0, 1, 2])); e = engine(b, k=rng.randint(1, 4))
            rs = [request(str(i), [1] * rng.randint(1, 11), rng.randint(1, 35)) for i in range(rng.randint(1, 4))]
            out = e.generate(rs)
            self.assertEqual([len(x['token_ids']) for x in out], [r.max_tokens for r in rs])
            self.assert_released(e, rs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
