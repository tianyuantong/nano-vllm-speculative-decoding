import torch
from torch import nn

from nanovllm.layers.spec_sampler import SamplingTensors, probs_from_logits, sample


class Sampler(nn.Module):
    """Ordinary (k = 0) sampling; shares truncation and the exponential race with speculation."""

    def forward(self, logits: torch.Tensor, sampling: SamplingTensors,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if sampling.temperatures is None:
            return logits.argmax(dim=-1)
        return sample(probs_from_logits(logits, sampling), generator)
