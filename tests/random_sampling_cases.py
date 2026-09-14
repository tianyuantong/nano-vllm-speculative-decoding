"""Frozen small-vocabulary cases shared by reference and actual-backend checks.

This module has no torch dependency. ``expected_distribution`` enumerates the
ordinary target distribution, not a replacement implementation of the GPU
proposal/acceptance/residual path. All legal terminal categories are retained,
including those whose theoretical probability is zero.

Markov row 0 is the initial distribution; after token x, use row x + 1.
EOS belongs to the returned output and stops it immediately. Otherwise output
stops exactly at max_tokens. The target enumeration stops at EOS; a fixed-shape
GPU backend may still evaluate masked rows without committing their output.
"""

from math import fsum


CASES = [
    {"name": "user_70_30", "p": [0.7, 0.3], "q": [0.4, 0.6]},
    {"name": "identical", "p": [0.2, 0.3, 0.5], "q": [0.2, 0.3, 0.5]},
    {"name": "disjoint_support", "p": [0.6, 0.4, 0.0, 0.0],
     "q": [0.0, 0.0, 0.25, 0.75]},
    {"name": "target_zero", "p": [0.0, 0.3, 0.7], "q": [0.2, 0.5, 0.3]},
    {"name": "draft_zero", "p": [0.2, 0.3, 0.5, 0.0],
     "q": [0.5, 0.0, 0.5, 0.0]},
    {"name": "point_mass_draft", "p": [0.1, 0.2, 0.7], "q": [0.0, 1.0, 0.0]},
    {"name": "skewed", "p": [0.999, 0.0005, 0.0005],
     "q": [0.0005, 0.999, 0.0005]},
    {"name": "general_four", "p": [0.1, 0.2, 0.3, 0.4],
     "q": [0.4, 0.3, 0.2, 0.1]},
    {"name": "two_step_cap",
     "p_rows": [[0.6, 0.4], [0.9, 0.1], [0.2, 0.8]],
     "q_rows": [[0.2, 0.8], [0.3, 0.7], [0.75, 0.25]],
     "max_tokens": 2, "eos_token_id": None},
    {"name": "three_step_eos",
     "p_rows": [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25],
                [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
     "q_rows": [[0.25, 0.5, 0.25], [0.5, 0.25, 0.25],
                [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
     "max_tokens": 3, "eos_token_id": 2},
    {"name": "three_step_zero_paths",
     "p_rows": [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
     "q_rows": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
     "max_tokens": 3, "eos_token_id": None},
    {"name": "two_step_eos_four",
     "p_rows": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.25, 0.25],
                [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0]],
     "q_rows": [[0.0, 0.25, 0.5, 0.25], [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]],
     "max_tokens": 2, "eos_token_id": 3},
]


def expected_distribution(case) -> dict[tuple[int, ...], float]:
    """Enumerate ordinary target outputs using Python binary64 arithmetic.

The q tables are deliberately unused: the expected distribution is determined
by the target alone. This only supplies expected categories and probabilities
to the backend harness; it cannot demonstrate that the backend samples them.
    """
    if "p" in case:
        return {(token,): float(probability)
                for token, probability in enumerate(case["p"])}

    outputs = {}
    eos = case["eos_token_id"]

    def visit(prefix, probability):
        if len(prefix) == case["max_tokens"] or (prefix and prefix[-1] == eos):
            outputs[prefix] = probability
            return
        row = case["p_rows"][0 if not prefix else prefix[-1] + 1]
        for token, conditional in enumerate(row):
            visit(prefix + (token,), probability * conditional)

    visit((), 1.0)
    if abs(fsum(outputs.values()) - 1.0) > 1e-12:
        raise ValueError("target case is not a normalized output distribution")
    return outputs
