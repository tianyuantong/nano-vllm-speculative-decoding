import unittest
from test_random_decode import module, PrefixOracle, engine, request


class ViewTests(unittest.TestCase):
    def test_indices_slices_and_no_hidden_history_copy(self):
        class NoCopyList(list):
            def __add__(self, other):
                raise AssertionError('full history concatenation')
        for size in (0, 1, 7):
            base = NoCopyList(range(size))
            for prefix in range(size+1):
                for tail in ([], [21], [21, 22, 23]):
                    view = module.TokenView(base, tail, prefix)
                    expected = list(base[:prefix]) + tail
                    self.assertEqual(list(view), expected)
                    for a in (None, -20, -2, 0, 1, 5, 20):
                        for b in (None, -20, -1, 0, 3, 20):
                            for step in (None, -2, -1, 1, 2):
                                sl = slice(a,b,step)
                                self.assertEqual(view[sl], expected[sl])
                    for i in (-len(view)-1, len(view)):
                        with self.assertRaises(IndexError):
                            view[i]

    def test_state_transitions_match_original_history_lists(self):
        for rejection in (None, 0, 1, 2, 3):
            for eos in (False, True):
                results = []
                for views in (False, True):
                    backend = PrefixOracle(reject=rejection, proposal_eos=eos)
                    decoder = engine(backend)
                    decoder.r2_views = views
                    requests = [request('a', cap=6), request('b', cap=19)]
                    outputs = decoder.generate(requests)
                    self.assertFalse(decoder.target_pool.used or decoder.draft_pool.used)
                    results.append((outputs, backend.events))
                self.assertEqual(*results)


if __name__ == '__main__':
    unittest.main()
