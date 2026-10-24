"""DSpark's propose: the drafts are drawn left to right, each row's logits biased by the Markov head on the token
before it (the anchor for row 0), and the q handed to acceptance is each row's distribution after its bias."""
from types import SimpleNamespace

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import DFlashDrafter, SpeculativeDecoder
from nanovllm.layers.spec_sampler import probs_from_logits, sampling_tensors
from nanovllm.sampling_params import SamplingParams

CPU = torch.device("cpu")
# The Markov bias by previous token, as in the patch note's example: token 0 adds 2 to token 1's logit.
BIAS = torch.tensor([[0.0, 2.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def make_drafter(k: int) -> DFlashDrafter:
    config = SimpleNamespace(num_speculative_tokens=k, kvcache_block_size=256, max_model_len=64,
                             hf_config=SimpleNamespace(vocab_size=3))
    target = SimpleNamespace(device=CPU, generator=torch.Generator().manual_seed(0))
    decoder = SpeculativeDecoder(target, object(), config)    # builds a ModelDrafter, unused here
    drafter = object.__new__(DFlashDrafter)
    drafter.decoder, drafter.model = decoder, SimpleNamespace(markov_head=lambda prev: BIAS[prev])
    return drafter


def test_greedy_drafts_follow_the_markov_chain():
    # Base logits [2, 1, 0] at every row pick token 0; after token 0 the bias makes token 1 win, after token 1 it
    # does not. Anchor 0 gives 1, 0, 1; anchor 1 gives 0, 1, 0.
    seqs = [Sequence([2, 0], SamplingParams(temperature=0)), Sequence([2, 1], SamplingParams(temperature=0))]
    logits = torch.tensor([2.0, 1.0, 0.0]).expand(2, 3, 3)
    drafts, probs = make_drafter(k=3)._sample_markov(seqs, logits, sampling_tensors(seqs, CPU))
    assert drafts.tolist() == [[1, 0, 1], [0, 1, 0]] and probs is None


def test_q_is_each_row_distribution_after_its_bias():
    seqs = [Sequence([2, 0], SamplingParams(temperature=1.0)), Sequence([0, 1], SamplingParams(temperature=0.7, top_k=2))]
    sampling = sampling_tensors(seqs, CPU)
    logits = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(1))
    drafts, probs = make_drafter(k=4)._sample_markov(seqs, logits, sampling)
    prev = torch.tensor([0, 1])                               # the anchors
    for row in range(4):
        assert torch.equal(probs[:, row], probs_from_logits(logits[:, row] + BIAS[prev], sampling))
        prev = drafts[:, row]
