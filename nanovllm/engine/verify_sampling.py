"""Candidate k3 deterministic post-p/q block. Not installed in the engine yet.

RNG remains external. This candidate requires the existing draw/residual fast
paths; arithmetic and invalid propagation are preserved, including both draws.
"""
import torch

from nanovllm.layers import random_sampler as sampler


def _draw_with_noise(probabilities, noise):
    # Exact simplify=True draw arithmetic, with the same-shaped noise supplied.
    safe = probabilities.values.masked_fill(probabilities.invalid[:, None], 0.0)
    minimum, maximum = torch.aminmax(noise, dim=-1)
    noise_ok = (minimum > 0) & torch.isfinite(maximum)
    scores = safe.div_(noise)
    invalid = probabilities.invalid | ~noise_ok | ~torch.isfinite(scores).all(dim=-1)
    return sampler.Sample(scores.argmax(dim=-1), invalid)


@torch.inference_mode()
def post_pq(p_values, p_mass, p_invalid, q_values, q_mass, q_invalid,
            token_ids, uniforms, correction_noise, bonus_noise, prior_invalid):
    """Pure tensor block; all operands/output remain on the caller's device."""
    p = sampler.Probabilities(p_values, p_mass, p_invalid)
    q = sampler.Probabilities(q_values, q_mass, q_invalid)
    proposed = sampler.Probabilities(p.values[:3], p.mass[:3], p.invalid[:3])
    decision = sampler.accept(proposed, q, token_ids, uniforms)
    count = decision.accepted.long().cumprod(0).sum().reshape(1)
    selected_p = sampler.Probabilities(*(t.index_select(0, count) for t in (p.values, p.mass, p.invalid)))
    index = count.clamp_max(2)
    selected = [t.index_select(0, index) for t in (q.values, q.mass, q.invalid)]
    selected_q = sampler.Probabilities(
        torch.where((count < 3)[:, None], selected[0], selected_p.values),
        torch.where(count < 3, selected[1], selected_p.mass),
        torch.where(count < 3, selected[2], selected_p.invalid))
    correction = _draw_with_noise(sampler.residual(selected_p, selected_q, count < 3, simplify=True),
                                  correction_noise)
    bonus = _draw_with_noise(sampler.Probabilities(p.values[-1:], p.mass[-1:], p.invalid[-1:]), bonus_noise)
    invalid = prior_invalid | p.invalid.any().reshape(1)
    invalid = invalid | decision.invalid.any().reshape(1)
    invalid = invalid | correction.invalid | bonus.invalid
    tail = torch.where(count < 3, correction.token_ids, bonus.token_ids)
    return count, tail, invalid


class VerifySamplingGraph:
    """One private k3 graph, owned by one calling stream; inputs copied in full.

Construct only after natural model warmup/freeze, outside measured generate.
Returned small tensors own storage, so another request cannot overwrite them.
"""
    def __init__(self, *, device, vocab_size, reserve_budget_bytes=256 << 20, min_free_bytes=1 << 30):
        if vocab_size <= 0 or reserve_budget_bytes <= 0 or min_free_bytes < 0:
            raise ValueError('invalid sampling graph resource contract')
        self.device = torch.device(device)
        self.owner = torch.cuda.current_stream(self.device)
        self.graph = None
        self.inputs = None
        self.outputs = None
        self.replays = 0
        self.captures = 0
        self.closed = False
        before = torch.cuda.memory_reserved(self.device)
        try:
            with torch.inference_mode():
                p = torch.full((4, vocab_size), 1.0 / vocab_size, dtype=torch.float32, device=self.device)
                q = torch.full((3, vocab_size), 1.0 / vocab_size, dtype=torch.float32, device=self.device)
                self.inputs = (p, p.double().sum(-1), torch.zeros(4, dtype=torch.bool, device=self.device),
                    q, q.double().sum(-1), torch.zeros(3, dtype=torch.bool, device=self.device),
                    torch.zeros(3, dtype=torch.int64, device=self.device),
                    torch.full((3,), 0.5, dtype=torch.float64, device=self.device),
                    torch.ones((1, vocab_size), dtype=torch.float64, device=self.device),
                    torch.ones((1, vocab_size), dtype=torch.float32, device=self.device),
                    torch.zeros(1, dtype=torch.bool, device=self.device))
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(self.owner)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        warm = post_pq(*self.inputs)
                    del warm
                stream.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
                    self.outputs = post_pq(*self.inputs)
                self.owner.wait_stream(stream)
                self.owner.synchronize()
            self.reserved_delta = torch.cuda.memory_reserved(self.device) - before
            self.free_bytes = torch.cuda.mem_get_info(self.device)[0]
            if self.reserved_delta > reserve_budget_bytes or self.free_bytes < min_free_bytes:
                raise MemoryError('sampling graph exceeds its predeclared memory budget')
            self.captures = 1
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def __call__(self, *inputs):
        if self.closed or torch.cuda.current_stream(self.device) != self.owner:
            raise RuntimeError('sampling graph must use its live owner stream')
        if len(inputs) != len(self.inputs):
            raise ValueError('sampling graph input count mismatch')
        for actual, fixed in zip(inputs, self.inputs, strict=True):
            if actual.shape != fixed.shape or actual.dtype != fixed.dtype or actual.device != fixed.device:
                raise ValueError('sampling graph input layout mismatch')
        for actual, fixed in zip(inputs, self.inputs, strict=True):
            fixed.copy_(actual)
        self.graph.replay()
        self.replays += 1
        return tuple(value.clone() for value in self.outputs)

    def statistics(self):
        return {'captures': self.captures, 'replays': self.replays,
                'reserved_delta_bytes': self.reserved_delta, 'device_free_bytes_at_capture': self.free_bytes,
                'private_graph_count': 1, 'rng_inside_graph': False}

    def close(self):
        if not self.closed:
            self.owner.synchronize()
            self.graph = None
            self.outputs = None
            self.inputs = None
            self.closed = True
