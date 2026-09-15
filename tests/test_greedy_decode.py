"""Greedy controller payload/KV tests with independent physical-slot oracle."""
import unittest
from test_random_decode import PrefixOracle, engine, Request


class GreedyOracle(PrefixOracle):
    def propose(self, requests, logits):
        ids, _ = super().propose(requests, logits)
        return ids, None

    def point_mass_proposals(self, *args):
        raise AssertionError("greedy fell into the probability backend")

    def verify(self, requests, logits, proposals, probabilities):
        assert probabilities is None
        if self.fail == 'verify':
            raise FloatingPointError('invalid greedy boundary')
        self.rounds += 1
        result = []
        for request, values in zip(requests, logits, strict=True):
            ds = proposals[request.request_id]
            assert len(values) == len(ds) + 1
            n = len(ds) if self.reject is None else min(self.reject, len(ds))
            result.append(ds[:n] + [30])
        return result


def request(rid='g', cap=19, prompt=None):
    return Request(rid, [1, 2, 3, 4] if prompt is None else prompt, cap, None, temperature=0.0)


class GreedyStateTests(unittest.TestCase):
    def test_all_k_and_reject_positions_with_mixed_budgets(self):
        for k in range(5):
            for reject in range(k + 1):
                b = GreedyOracle(reject=reject)
                d = engine(b, k=k, sampling_mode='greedy')
                rs = [request(str(i), cap, [1] * (i + 3)) for i, cap in enumerate([1, 6, 13, 23])]
                out = d.generate(rs)
                self.assertEqual([len(o['token_ids']) for o in out], [1, 6, 13, 23])
                self.assertFalse(d.target_pool.used or (d.draft_pool and d.draft_pool.used))

    def test_k4_full_acceptance_and_final_remaining_one(self):
        for cap, has_catchup in [(7, False), (12, True)]:
            b = GreedyOracle()
            d = engine(b, k=4, sampling_mode='greedy')
            out = d.generate([request(cap=cap)])
            self.assertEqual(len(out[0]['token_ids']), cap)
            self.assertEqual(any(e[:4] == ('draft', 'g', 8, 9) for e in b.events), has_catchup)

    def test_eos_accepted_rejected_and_ignored(self):
        for reject, ignore in [(None, False), (0, False), (None, True)]:
            b = GreedyOracle(reject=reject, proposal_eos=True)
            d = engine(b, k=4, sampling_mode='greedy')
            r = request(); r.ignore_eos = ignore
            out = d.generate([r])[0]
            self.assertEqual(out['stop_reason'], 'eos' if reject is None and not ignore else 'length')
            self.assertFalse(d.target_pool.used or d.draft_pool.used)

    def test_whole_batch_bad_temperature_before_allocation(self):
        b = GreedyOracle(); d = engine(b, sampling_mode='greedy')
        rs = [request('valid'), request('bad')]; rs[1].temperature = 1.0
        with self.assertRaises(ValueError): d.generate(rs)
        self.assertFalse(d.target_pool.used or d.draft_pool.used or b.events)
        self.assertFalse(any(r.used for r in rs))

    def test_whole_batch_bad_prompt_before_allocation(self):
        b = GreedyOracle(); d = engine(b, sampling_mode='greedy')
        with self.assertRaises(ValueError): d.generate([request('valid'), request('bad', prompt=[-1])])
        self.assertFalse(d.target_pool.used or d.draft_pool.used or b.events)

    def test_failure_releases_both_pools(self):
        b = GreedyOracle(fail='verify'); d = engine(b, sampling_mode='greedy')
        with self.assertRaises(FloatingPointError): d.generate([request()])
        self.assertTrue(d.poisoned)
        self.assertFalse(d.target_pool.used or d.draft_pool.used)

    def test_random_still_rejects_zero_and_greedy_ngram_rejected(self):
        with self.assertRaises(ValueError): engine(GreedyOracle()).generate([request()])
        with self.assertRaises(ValueError): engine(GreedyOracle(), sampling_mode='greedy', ngram=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
