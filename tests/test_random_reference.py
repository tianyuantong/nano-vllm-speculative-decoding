"""Independent exact/FP64 probability checks; no torch and no GPU claims.

Run: python3 tests/test_random_reference.py
These checks only validate the frozen cases and mathematical reference. The
GPU frequency harness must call the actual production proposal, acceptance,
and residual functions; passing this file cannot substitute for that test.
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import unittest

from random_sampling_cases import CASES, expected_distribution


FROZEN_CASES_SHA256 = "e918671a27e2d6688448e8e85ac8f5f1587b8ace3135a901971204229eaa8b4a"
FREQUENCY_SAMPLES = 100_000
FAMILY_FAILURE_PROBABILITY = 0.001


def exact_row(row):
    # Decimal declarations are the specification; do not rationalize binary64.
    return tuple(Fraction(str(value)) for value in row)


def row_pairs():
    for case in CASES:
        if "p" in case:
            yield case["name"], case["p"], case["q"]
        else:
            for index, (p, q) in enumerate(zip(case["p_rows"], case["q_rows"])):
                yield f"{case['name']}:{index}", p, q


def exact_components(p, q):
    """Accepted joint mass + total rejection mass times conditional residual."""
    accepted = tuple(min(px, qx) for px, qx in zip(p, q))
    positive = tuple(max(Fraction(0), px - qx) for px, qx in zip(p, q))
    rejected = 1 - sum(accepted)
    assert rejected == sum(positive)
    # p=q never invokes residual sampling. No invented uniform fallback.
    residual = tuple(value / rejected for value in positive) if rejected else None
    return accepted, rejected, residual


def exact_target_outputs(case):
    rows = tuple(exact_row(row) for row in case["p_rows"])
    outputs = {}

    def visit(prefix, probability):
        stopped = prefix and prefix[-1] == case["eos_token_id"]
        if stopped or len(prefix) == case["max_tokens"]:
            outputs[prefix] = probability
            return
        for token, value in enumerate(rows[0 if not prefix else prefix[-1] + 1]):
            visit(prefix + (token,), probability * value)

    visit((), Fraction(1))
    return outputs


def exact_speculative_outputs(case, draft_length=2):
    """Enumerate whole speculative rounds, including first rejection and bonus.

This reference terminates a proposal on EOS; accepted EOS stops immediately.
Rejected EOS is replaced and generation can continue. It is intentionally not
imported by the GPU harness as the implementation under test.
    """
    p_rows = tuple(exact_row(row) for row in case["p_rows"])
    q_rows = tuple(exact_row(row) for row in case["q_rows"])
    cap, eos = case["max_tokens"], case["eos_token_id"]

    def stopped(prefix):
        return len(prefix) == cap or bool(prefix and prefix[-1] == eos)

    def row(rows, prefix):
        return rows[0 if not prefix else prefix[-1] + 1]

    def proposals(prefix, draft=(), probability=Fraction(1)):
        if len(draft) == draft_length or stopped(prefix + draft):
            yield draft, probability
            return
        for token, mass in enumerate(row(q_rows, prefix + draft)):
            if mass:
                yield from proposals(prefix, draft + (token,), probability * mass)

    @lru_cache(None)
    def rounds(prefix):
        if stopped(prefix):
            return {prefix: Fraction(1)}
        outputs = defaultdict(Fraction)

        def continue_with(history, mass):
            if mass:
                for result, conditional in rounds(history).items():
                    outputs[result] += mass * conditional

        def verify(history, draft, index, mass):
            if stopped(history):
                outputs[history] += mass
                return
            p = row(p_rows, history)
            if index == len(draft):
                # All proposed tokens accepted; draw the extra token from p.
                for token, value in enumerate(p):
                    continue_with(history + (token,), mass * value)
                return
            q = row(q_rows, history)
            candidate = draft[index]
            assert q[candidate] > 0  # impossible proposals were excluded above
            acceptance = min(Fraction(1), p[candidate] / q[candidate])
            if acceptance:
                verify(history + (candidate,), draft, index + 1, mass * acceptance)
            if acceptance < 1:
                _, _, residual = exact_components(p, q)
                assert residual is not None
                for token, value in enumerate(residual):
                    continue_with(history + (token,), mass * (1 - acceptance) * value)

        for draft, mass in proposals(prefix):
            verify(prefix, draft, 0, mass)
        return dict(outputs)

    return rounds(())


class RandomReferenceTests(unittest.TestCase):
    def test_frozen_12_case_identity(self):
        self.assertEqual(len(CASES), 12)
        self.assertEqual(len({case["name"] for case in CASES}), 12)
        encoded = json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), FROZEN_CASES_SHA256)

    def test_case_shape_and_exact_normalization(self):
        self.assertTrue(all("p" in case for case in CASES[:8]))
        self.assertTrue(all("p_rows" in case for case in CASES[8:]))
        for name, p, q in row_pairs():
            with self.subTest(row=name):
                self.assertIn(len(p), (2, 3, 4))
                self.assertEqual(len(p), len(q))
                for values in (p, q):
                    self.assertTrue(all(0 <= value <= 1 for value in values))
                    self.assertEqual(sum(exact_row(values)), 1)
        for case in CASES[8:]:
            vocab = len(case["p_rows"][0])
            self.assertEqual(len(case["p_rows"]), vocab + 1)
            self.assertEqual(len(case["q_rows"]), vocab + 1)
            self.assertIn(case["max_tokens"], (2, 3))
            self.assertTrue(case["eos_token_id"] is None
                            or 0 <= case["eos_token_id"] < vocab)

    def test_exact_accepted_plus_residual_equals_target(self):
        for name, p_values, q_values in row_pairs():
            with self.subTest(row=name):
                p, q = exact_row(p_values), exact_row(q_values)
                accepted, rejected, residual = exact_components(p, q)
                result = tuple(value + (rejected * residual[i] if residual else 0)
                               for i, value in enumerate(accepted))
                self.assertEqual(result, p)
                self.assertEqual(sum(result), 1)

    def test_binary64_identity_with_fixed_tolerance(self):
        for name, p, q in row_pairs():
            with self.subTest(row=name):
                accepted = [min(px, qx) for px, qx in zip(p, q)]
                positive = [max(0.0, px - qx) for px, qx in zip(p, q)]
                rejected = 1.0 - math.fsum(accepted)
                normalizer = math.fsum(positive)
                result = [mass + (rejected * positive[i] / normalizer
                                  if normalizer else 0.0)
                          for i, mass in enumerate(accepted)]
                self.assertLessEqual(max(abs(a - b) for a, b in zip(result, p)), 1e-12)

    def test_unnormalized_binary_weights_use_actual_sampling_distribution(self):
        # These are exact binary float values, not the decimal case convention.
        # Weighted categorical sampling uses row / sum(row), even when a row
        # is close enough to one to pass the probability validity tolerance.
        raw_p = tuple(map(Fraction.from_float,
                          [0.7000035047531128, 0.30000150203704834]))
        raw_q = tuple(map(Fraction.from_float,
                          [0.4000000059604645, 0.6000000238418579]))
        p_mass, q_mass = sum(raw_p), sum(raw_q)
        self.assertNotEqual(p_mass, 1)
        self.assertNotEqual(q_mass, 1)
        self.assertNotEqual(p_mass, q_mass)
        p = tuple(value / p_mass for value in raw_p)
        q = tuple(value / q_mass for value in raw_q)

        accepted, rejected, residual = exact_components(p, q)
        self.assertIsNotNone(residual)
        self.assertEqual(accepted, tuple(qx * min(1, px / qx)
                                         for px, qx in zip(p, q)))
        output = tuple(value + rejected * residual[token]
                       for token, value in enumerate(accepted))
        self.assertEqual(sum(output), 1)
        self.assertEqual(output, p)

        # Using the raw ratio/residual after drawing from normalized q remains
        # a valid distribution, but it is the wrong target distribution.
        wrong_accepted = tuple(qx * min(1, px_raw / qx_raw)
                               for qx, px_raw, qx_raw in zip(q, raw_p, raw_q))
        raw_positive = tuple(max(Fraction(0), px - qx)
                             for px, qx in zip(raw_p, raw_q))
        raw_residual_mass = sum(raw_positive)
        wrong_output = tuple(value + (1 - sum(wrong_accepted))
                             * raw_positive[token] / raw_residual_mass
                             for token, value in enumerate(wrong_accepted))
        self.assertEqual(sum(wrong_output), 1)
        self.assertNotEqual(wrong_output, p)

    def test_user_70_30_hand_calculation(self):
        case = CASES[0]
        accepted, rejected, residual = exact_components(exact_row(case["p"]),
                                                       exact_row(case["q"]))
        self.assertEqual(accepted, (Fraction(2, 5), Fraction(3, 10)))
        self.assertEqual(rejected, Fraction(3, 10))
        self.assertEqual(residual, (Fraction(1), Fraction(0)))

    def test_zero_q_is_an_impossible_proposal_not_a_division(self):
        zero_entries = 0
        for _, p_values, q_values in row_pairs():
            p, q = exact_row(p_values), exact_row(q_values)
            accepted, _, _ = exact_components(p, q)
            for token, proposal_mass in enumerate(q):
                if proposal_mass == 0:
                    zero_entries += 1
                    self.assertEqual(accepted[token], 0)
                else:
                    self.assertEqual(proposal_mass * min(1, p[token] / proposal_mass),
                                     accepted[token])
        self.assertGreater(zero_entries, 0)

    def test_identical_and_disjoint_residual_branches(self):
        same = CASES[1]
        accepted, rejected, residual = exact_components(exact_row(same["p"]),
                                                       exact_row(same["q"]))
        self.assertEqual(accepted, exact_row(same["p"]))
        self.assertEqual(rejected, 0)
        self.assertIsNone(residual)
        disjoint = CASES[2]
        accepted, rejected, residual = exact_components(exact_row(disjoint["p"]),
                                                       exact_row(disjoint["q"]))
        self.assertEqual(sum(accepted), 0)
        self.assertEqual(rejected, 1)
        self.assertEqual(residual, exact_row(disjoint["p"]))

    def test_whole_speculative_rounds_match_exact_target_sequences(self):
        for case in CASES[8:]:
            # k=1 also exercises repeated correction/bonus rounds before cap.
            for draft_length in (1, 2):
                with self.subTest(case=case["name"], k=draft_length):
                    target = exact_target_outputs(case)
                    actual = exact_speculative_outputs(case, draft_length)
                    self.assertFalse(set(actual) - set(target))
                    self.assertEqual(sum(actual.values()), 1)
                    for sequence, probability in target.items():
                        self.assertEqual(actual.get(sequence, 0), probability)

    def test_binary64_expected_sequence_probabilities(self):
        for case in CASES:
            with self.subTest(case=case["name"]):
                expected = expected_distribution(case)
                self.assertLessEqual(abs(math.fsum(expected.values()) - 1), 1e-12)
                exact = ({(i,): value for i, value in enumerate(exact_row(case["p"]))}
                         if "p" in case else exact_target_outputs(case))
                self.assertEqual(set(expected), set(exact))
                for sequence, probability in exact.items():
                    self.assertLessEqual(abs(expected[sequence] - float(probability)), 1e-12)

    def test_eos_is_included_and_no_tokens_follow_it(self):
        for case in CASES[8:]:
            eos, cap = case["eos_token_id"], case["max_tokens"]
            for output in expected_distribution(case):
                self.assertTrue(1 <= len(output) <= cap)
                if eos is not None and eos in output:
                    self.assertEqual(output.index(eos), len(output) - 1)
                else:
                    self.assertEqual(len(output), cap)
        self.assertEqual(expected_distribution(CASES[9])[(2,)], 0.25)
        self.assertEqual(expected_distribution(CASES[11])[(3,)], 0.4)

    def test_frozen_hoeffding_family_includes_zero_probability_categories(self):
        counts = [len(expected_distribution(case)) for case in CASES]
        self.assertEqual(counts, [2, 3, 4, 3, 4, 3, 3, 4, 4, 15, 8, 13])
        self.assertEqual(sum(counts), 66)
        self.assertGreater(sum(value == 0.0 for case in CASES
                               for value in expected_distribution(case).values()), 0)

    def test_fixed_simultaneous_frequency_bound(self):
        # This computes the predeclared threshold; it runs no frequency test.
        categories = sum(len(expected_distribution(case)) for case in CASES)
        bound = math.sqrt(math.log(2 * categories / FAMILY_FAILURE_PROBABILITY)
                          / (2 * FREQUENCY_SAMPLES))
        self.assertAlmostEqual(2 * categories * math.exp(-2 * FREQUENCY_SAMPLES * bound**2),
                               FAMILY_FAILURE_PROBABILITY, places=14)
        self.assertTrue(0.007 < bound < 0.008)


if __name__ == "__main__":
    categories = sum(len(expected_distribution(case)) for case in CASES)
    bound = math.sqrt(math.log(2 * categories / FAMILY_FAILURE_PROBABILITY)
                      / (2 * FREQUENCY_SAMPLES))
    print(f"REFERENCE_ONLY cases={len(CASES)} M={categories} "
          f"samples_per_case={FREQUENCY_SAMPLES} bound={bound:.12f}", flush=True)
    print("Exact/FP64 mathematics only; actual CPU/GPU sampling was NOT executed.", flush=True)
    unittest.main(verbosity=2)
