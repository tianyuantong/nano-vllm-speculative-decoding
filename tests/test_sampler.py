from types import SimpleNamespace

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.layers.spec_sampler import sampling_tensors

CPU = torch.device("cpu")


def test_greedy_batch_returns_argmax():
    sampling = sampling_tensors([SimpleNamespace(temperature=0, top_k=0, top_p=1.0)], CPU)
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert Sampler()(logits, sampling).tolist() == [1]


def test_random_batch_respects_truncation():
    sampling = sampling_tensors([SimpleNamespace(temperature=1.0, top_k=1, top_p=1.0)] * 64, CPU)
    logits = torch.tensor([[1.0, 3.0, 2.0]]).expand(64, 3)
    tokens = Sampler()(logits, sampling, torch.Generator().manual_seed(0))
    assert (tokens == 1).all()
