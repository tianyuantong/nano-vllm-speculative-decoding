"""Fixture oracle checks only; CUDA row selection remains a separate test."""
import unittest
from random_position_cases import (position_probabilities as probs, ALL_ACCEPT_OUTPUTS,
                                   SECOND_REJECT_OUTPUTS, SECOND_RESIDUAL_ROWS)

class PositionCases(unittest.TestCase):
    def test_all_accept_positions_and_bonus_are_distinct(self):
        for rid, expected in ALL_ACCEPT_OUTPUTS.items():
            prefix = [0, 1, 2]
            for token in expected:
                p = probs(rid, prefix, role='target')
                self.assertEqual(p, [int(i == token) for i in range(4)])
                self.assertEqual(p, probs(rid, prefix, role='draft'))
                prefix.append(token)
            self.assertNotEqual(expected[-1], expected[-2])

    def test_second_rejection_has_known_p_q_and_residual(self):
        for rid, expected in SECOND_REJECT_OUTPUTS.items():
            initial, first, correction, last = expected
            prefix = [0,1,2,initial]
            self.assertEqual(probs(rid, prefix, role='target', reject_second=True)[first], 1)
            prefix.append(first)
            p = probs(rid, prefix, role='target', reject_second=True)
            q = probs(rid, prefix, role='draft')
            self.assertEqual((p,q), SECOND_RESIDUAL_ROWS[rid])
            proposal = q.index(1)
            self.assertEqual(p[proposal], 0)  # deterministic rejection
            self.assertEqual([max(0,a-b) for a,b in zip(p,q)], p)
            self.assertEqual(p[correction], 1)
            self.assertEqual(probs(rid, prefix+[correction], role='target', reject_second=True)[last], 1)

    def test_wrong_previous_q_row_has_zero_proposal_support(self):
        for rid, expected in ALL_ACCEPT_OUTPUTS.items():
            prefix = [0,1,2] + expected[:1]
            previous = probs(rid, prefix, role='draft')
            current = probs(rid, prefix+[expected[1]], role='draft')
            self.assertEqual(previous[current.index(1)], 0)

    def test_request_rows_and_prefixes_cannot_be_interchanged(self):
        self.assertNotEqual(ALL_ACCEPT_OUTPUTS['a'], ALL_ACCEPT_OUTPUTS['b'])
        self.assertNotEqual(SECOND_RESIDUAL_ROWS['a'], SECOND_RESIDUAL_ROWS['b'])
        self.assertNotEqual(probs('a',[0,1,2,0],role='target'), probs('a',[0,1,2,1],role='target'))

if __name__ == '__main__': unittest.main(verbosity=2)
