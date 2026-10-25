"""G1-draft for DSpark (tools/gate_dflash.py): each row's Markov bias comes from the token before it on the reference's
greedy chain, the anchor for row 0, as the engine's propose takes it from the drafts it has sampled."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gate_dflash import markov_rows  # noqa: E402

# The Markov bias by previous token, as in test_dspark_propose: token 0 adds 2 to token 1's logit.
BIAS = torch.tensor([[0.0, 2.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def test_markov_rows_follow_the_reference_chain():
    # Anchor 0 and the greedy chain 1, 0, 1 of T2's example: rows 0 and 2 follow token 0, row 1 follows token 1.
    logits = torch.tensor([2.0, 1.0, 0.0]).expand(3, 3)
    rows = markov_rows(logits, lambda prev: BIAS[prev], anchor_token=0, chain=[1, 0, 1])
    assert rows.tolist() == [[2.0, 3.0, 0.0], [2.0, 1.0, 0.0], [2.0, 3.0, 0.0]]
