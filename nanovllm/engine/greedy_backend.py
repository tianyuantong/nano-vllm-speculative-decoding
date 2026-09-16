"""Greedy sampling policy for the existing offline controller and dual KV state.

No probabilities, proposal q, or RNG. All rows are checked before host commit.
The shared backend supplies model execution and the existing return boundary.
"""
import torch

from nanovllm.engine.random_backend import CUDARandomBackend


class CUDAGreedyBackend(CUDARandomBackend):
    def _argmax(self, requests, logits, lengths, role):
        vocab = self.runners[role].config.hf_config.vocab_size
        if len(requests) != len(logits) or len(lengths) != len(requests):
            raise ValueError("greedy batch layout mismatch")
        for values, size in zip(logits, lengths, strict=True):
            if vocab <= 0 or size <= 0 or values.ndim != 2 or values.shape != (size, vocab):
                raise ValueError("greedy logits layout mismatch")
        values = torch.cat(logits) if len(logits) > 1 else logits[0]
        # Argmax owns its storage, independently of model Graph replay buffers.
        tokens = values.argmax(-1)
        invalid = ~torch.isfinite(values).all(-1)
        for request, mask in zip(requests, invalid.split(lengths), strict=True):
            self._remember(request, mask)
        return tokens.split(lengths)

    @torch.inference_mode()
    def sample_batch(self, requests, logits, role):
        ids = self._argmax(requests, logits, [1] * len(requests), "target")
        return [row[0] for row in self._materialize_checked(requests, torch.cat(ids)[:, None])]

    @torch.inference_mode()
    def propose_device(self, requests, logits):
        ids = self._argmax(requests, logits, [1] * len(requests), "draft")
        return ids, None

    @torch.inference_mode()
    def verify(self, requests, logits, proposals, saved_q):
        if saved_q is not None:
            raise ValueError("greedy VERIFY must not construct proposal probabilities")
        vocab = self.runners["target"].config.hf_config.vocab_size
        drafts = [proposals[r.request_id] for r in requests]
        for ds in drafts:
            if any(type(t) is not int or not 0 <= t < vocab for t in ds):
                raise ValueError("invalid greedy proposal ID")
        ids = self._argmax(requests, logits, [len(ds) + 1 for ds in drafts], "target")
        pairs = []
        for ds, target_ids in zip(drafts, ids, strict=True):
            k = len(ds)
            if k:
                candidate = torch.tensor(ds, dtype=torch.int64, device=target_ids.device)
                count = (target_ids[:k] == candidate).long().cumprod(0).sum().reshape(1)
            else:
                count = torch.zeros_like(target_ids)
            tail = target_ids.index_select(0, count)
            pairs.append(torch.cat((count, tail)))
        host = self._materialize_checked(requests, torch.stack(pairs))
        return [ds[:count] + [tail] for ds, (count, tail) in zip(drafts, host, strict=True)]
