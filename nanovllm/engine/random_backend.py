"""CUDA adapter for offline RandomDecode; uses the GPU-validated primitives.

One Generator per stable request ID and role. Draw shapes are per-request,
independent of other requests ending. All probability/RNG work is outside Graph.
"""

import hashlib
import json

import torch

from nanovllm.layers import random_sampler as sampler


ROLES = ("ordinary_target", "draft", "accept", "correction", "bonus")
RNG_VERSION = "nano-request-sha256-v1"


def role_seed(base_seed, request_id, role):
    if type(base_seed) is not int or not isinstance(request_id, str) or not request_id or role not in ROLES:
        raise ValueError("integer seed, nonempty stable string ID and known role required")
    encoded = json.dumps([RNG_VERSION, base_seed, request_id, role], ensure_ascii=False,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


class RequestRNG:
    def __init__(self, base_seed, request_id, device="cuda:0"):
        self.request_id = request_id
        self.seeds = {role: role_seed(base_seed, request_id, role) for role in ROLES}
        self.streams = {role: torch.Generator(device=device).manual_seed(seed)
                        for role, seed in self.seeds.items()}
        self.used = False

    def metadata(self):
        return {"derivation": RNG_VERSION, "request_id": self.request_id,
                "torch": torch.__version__, "seeds": self.seeds,
                "states_sha256": {role: hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest()
                                  for role, g in self.streams.items()}}


class CUDARandomBackend:
    def __init__(self, target, draft=None):
        self.runners = {"target": target, "draft": draft}
        self.invalid = {}

    def synchronize(self):
        torch.cuda.synchronize()

    def forward(self, role, queries, *, all_logits=False):
        return self.runners[role].run_queries(queries, all_logits=all_logits)

    def forward_draft_device(self, queries, token_ids):
        # Internal only: tokens are independently allocated argmax outputs from
        # propose_device, never arbitrary caller-supplied embedding indices.
        ids = torch.cat(token_ids)
        if ids.shape != (len(queries),) or ids.dtype != torch.int64:
            raise ValueError("device proposal layout mismatch")
        return self.runners["draft"].run_queries(queries, device_input_ids=ids)

    def materialize_proposals(self, proposals):
        sizes = [len(values) for values in proposals.values()]
        flat = [token for values in proposals.values() for token in values]
        host = torch.cat(flat).tolist() if flat else []
        result, offset = {}, 0
        for (rid, _), size in zip(proposals.items(), sizes, strict=True):
            result[rid] = host[offset:offset+size]
            offset += size
        return result

    def _remember(self, request, mask):
        value = mask.any().reshape(1)
        self.invalid[request.request_id] = self.invalid.get(request.request_id, value) | value

    def _boundary(self, requests):
        sampler.require_valid_boundary(torch.cat([self.invalid[r.request_id] for r in requests]))

    def _probabilities(self, r, logits):
        temp = torch.full((logits.shape[0],), r.temperature, device=logits.device, dtype=torch.float32)
        return sampler.from_logits(logits, temp)

    @torch.inference_mode()
    def sample_batch(self, requests, logits, role):
        result = []
        for r, values in zip(requests, logits, strict=True):
            draw = sampler.draw(self._probabilities(r, values), generator=r.rng.streams[role])
            self._remember(r, draw.invalid)
            result.append(draw.token_ids)
        self._boundary(requests)
        return torch.cat(result).tolist()

    @torch.inference_mode()
    def propose(self, requests, logits):
        ids, probabilities = self.propose_device(requests, logits)
        return torch.cat(ids).tolist(), probabilities

    @torch.inference_mode()
    def propose_device(self, requests, logits):
        ids, probabilities = [], []
        vocab = self.runners["draft"].config.hf_config.vocab_size
        for r, values in zip(requests, logits, strict=True):
            # Shape metadata only: draw reduces the unchanged vocabulary axis.
            # Together with initialization's embedding dimensions this bounds
            # every internal token without a GPU scan or host synchronization.
            if vocab <= 0 or values.ndim != 2 or values.shape != (1, vocab):
                raise ValueError("draft proposal logits must have shape [1, vocabulary]")
            q = self._probabilities(r, values)  # distinct allocation, owned until VERIFY
            draw = sampler.draw(q, generator=r.rng.streams["draft"])
            self._remember(r, draw.invalid)
            probabilities.append(q)
            ids.append(draw.token_ids)
        return ids, probabilities

    @torch.inference_mode()
    def point_mass_proposals(self, requests, logits, proposals):
        """Dense one-hot q for deterministic N proposals; no draft RNG draw."""
        saved = {}
        vocab = self.runners["target"].config.hf_config.vocab_size
        for r, values in zip(requests, logits, strict=True):
            ds = proposals[r.request_id]
            if values.ndim != 2 or values.shape != (len(ds) + 1, vocab):
                raise ValueError("N verification logits layout mismatch")
            if any(type(t) is not int or not 0 <= t < vocab for t in ds):
                raise ValueError("invalid N proposal token")
            rows = []
            for token in ds:
                weights = torch.zeros((1, vocab), dtype=torch.float32, device=values.device)
                weights[0, token] = 1.0
                rows.append(sampler.from_probs(weights))
            saved[r.request_id] = rows
        return saved

    @torch.inference_mode()
    def verify(self, requests, logits, proposals, saved_q):
        pending = []
        for r, values in zip(requests, logits, strict=True):
            ds = proposals[r.request_id]
            k = len(ds)
            if values.shape[0] != k + 1:
                raise RuntimeError("VERIFY must return k+1 position distributions")
            p = self._probabilities(r, values)
            if k == 0:
                draw = sampler.draw(p, generator=r.rng.streams["ordinary_target"])
                self._remember(r, draw.invalid)
                pending.append((ds, torch.cat([torch.zeros_like(draw.token_ids), draw.token_ids])))
                continue
            qrows = saved_q[r.request_id]
            q = sampler.Probabilities(torch.cat([q.values for q in qrows]),
                                      torch.cat([q.mass for q in qrows]), torch.cat([q.invalid for q in qrows]))
            p_proposal = sampler.Probabilities(p.values[:k], p.mass[:k], p.invalid[:k])
            token_ids = torch.tensor(ds, dtype=torch.int64, device=values.device)
            u = torch.rand(k, dtype=torch.float64, device=values.device, generator=r.rng.streams["accept"])
            decision = sampler.accept(p_proposal, q, token_ids, u)
            count = decision.accepted.long().cumprod(0).sum().reshape(1)
            # Device index_select; no Python scalar read for the rejection position.
            p_selected = sampler.Probabilities(*(t.index_select(0, count) for t in (p.values, p.mass, p.invalid)))
            padded_q = (torch.cat([q.values, p.values[-1:]]), torch.cat([q.mass, p.mass[-1:]]),
                        torch.cat([q.invalid, p.invalid[-1:]]))
            q_selected = sampler.Probabilities(*(t.index_select(0, count) for t in padded_q))
            correction = sampler.draw(sampler.residual(p_selected, q_selected, count < k),
                                      generator=r.rng.streams["correction"])
            bonus_p = sampler.Probabilities(p.values[-1:], p.mass[-1:], p.invalid[-1:])
            bonus = sampler.draw(bonus_p, generator=r.rng.streams["bonus"])
            self._remember(r, p.invalid)
            self._remember(r, decision.invalid)
            self._remember(r, correction.invalid | bonus.invalid)
            tail = torch.where(count < k, correction.token_ids, bonus.token_ids)
            pending.append((ds, torch.cat([count, tail])))
        self._boundary(requests)  # entire batch validated before host commit decisions
        host = torch.stack([pair for _, pair in pending]).tolist()
        return [ds[:count] + [tail] for (ds, _), (count, tail) in zip(pending, host, strict=True)]
