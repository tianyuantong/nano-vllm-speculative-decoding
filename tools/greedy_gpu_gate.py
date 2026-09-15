"""Actual greedy CUDA policy tests, plus device physical-slot controller oracle."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import traceback


def main(output):
    result = {'status': 'RUNNING', 'checks': [], 'scope': 'CUDA policy and synthetic device KV; real Qwen checked separately'}
    handle = Path(output).open('x')
    try:
        import torch
        from nanovllm.engine.greedy_backend import CUDAGreedyBackend
        from nanovllm.engine.random_decode import RandomDecode, Request
        from nanovllm.engine.random_llm import RandomLLM
        assert torch.cuda.is_available()
        result['torch'] = torch.__version__
        result['device'] = torch.cuda.get_device_name()
        before_rng = torch.cuda.get_rng_state().clone()
        def runner(vocab):
            return SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab)))
        def logits(ids, vocab):
            values = torch.zeros((len(ids), vocab), device='cuda')
            values.scatter_(1, torch.tensor(ids, device='cuda')[:, None], 10)
            return values
        # Every possible first mismatch through k4, multiple query layouts.
        for k in range(5):
            for count in range(k + 1):
                b = CUDAGreedyBackend(runner(16), runner(16), performance_mode=True)
                rs = [Request(str(i), [1], 8, None, 0.0) for i in range(4)]
                ds = {r.request_id: list(range(1, k + 1)) for r in rs}
                target = list(range(1, k + 1)) + [9]
                if count < k: target[count] = 10
                out = b.verify(rs, [logits(target, 16) for _ in rs], ds, None)
                expected = list(range(1, count + 1)) + [target[count]]
                assert out == [expected] * 4
                result['checks'].append(f'k{k}_first_mismatch_{count}_batch4')
        b = CUDAGreedyBackend(runner(16), runner(16), performance_mode=True)
        rs = [Request(str(i), [1], 8, None, 0.0) for i in range(4)]
        ds = {'0': [], '1': [1, 2, 3, 4], '2': [1], '3': [1, 2]}
        targets = [[7], [1, 2, 8, 4, 9], [1, 6], [5, 2, 9]]
        assert b.verify(rs, [logits(t, 16) for t in targets], ds, None) == [[7], [1, 2, 8], [1, 6], [5]]
        result['checks'].append('mixed_k0_k4_k1_k2_no_post_reject_match')
        for vocab in [16, 151936]:
            b = CUDAGreedyBackend(runner(vocab), runner(vocab), performance_mode=True)
            r = rs[0]
            a = logits([vocab-1], vocab)
            ids, q = b.propose_device([r], [a]); saved = ids[0].clone()
            assert q is None
            b.propose_device([r], [logits([2], vocab)])
            assert torch.equal(ids[0], saved)
            tied = logits([2], vocab); tied[0, 1] = 10
            assert b.sample_batch([r], [tied], 'ordinary_target') == [1]
            for bad in [float('nan'), float('inf'), -float('inf')]:
                b.invalid.clear(); invalid = a.clone(); invalid[0, 0] = bad
                try: b.sample_batch([r], [invalid], 'ordinary_target')
                except FloatingPointError: pass
                else: raise AssertionError('invalid logits published')
            b.invalid.clear()
            b.propose_device([r], [torch.full((1, vocab), float('nan'), device='cuda')])
            try: b.verify([r], [logits([1, 2], vocab)], {r.request_id: [1]}, None)
            except FloatingPointError: pass
            else: raise AssertionError('draft invalid flag lost')
            result['checks'].append(f'vocab{vocab}_ties_lifetime_nonfinite_propagation')

        class DeviceRunner:
            def __init__(self, role, perturb):
                self.role, self.perturb = role, perturb
                self.config = runner(16).config
                self.cells = torch.full((400,), -1, dtype=torch.int64, device='cuda')
                self.events = []
            def run_queries(self, queries, *, all_logits=False, device_input_ids=None, need_logits=True):
                out = []
                for row, query in enumerate(queries):
                    slots = torch.tensor([query.block_table[p//4]*4+p%4 for p in range(len(query))], device='cuda')
                    cached = self.cells[slots[:query.num_cached_tokens]].tolist()
                    if device_input_ids is None:
                        assert cached == list(query.tokens[:query.num_cached_tokens])
                        tokens = list(query.tokens)
                    else:
                        assert query.num_scheduled_tokens == 1
                        tokens = cached + [int(device_input_ids[row])]
                    self.cells[slots] = torch.tensor(tokens, device='cuda')
                    self.events.append([query.request.request_id, query.num_cached_tokens, tokens, need_logits])
                    if not need_logits:
                        out.append(None); continue
                    positions = range(query.num_cached_tokens, len(query)) if all_logits else [len(query)-1]
                    chosen = []
                    for pos in positions:
                        value = (sum((j+1)*t for j,t in enumerate(tokens[:pos+1])) + pos) % 16
                        if self.role == 'draft' and self.perturb and pos % 3 == 0: value = (value+1)%16
                        chosen.append(value)
                    out.append(logits(chosen, 16))
                return out
        def run(k, device, perturb, eos):
            target, draft = DeviceRunner('target', perturb), DeviceRunner('draft', perturb)
            backend = CUDAGreedyBackend(target, draft if k else None, performance_mode=True)
            # Any accidental use of a probability route must fail immediately.
            def forbidden(*args, **kwargs): raise AssertionError('probability or RNG path used by greedy')
            backend._probabilities = backend._draw = backend.point_mass_proposals = forbidden
            decoder = RandomDecode(backend, target_blocks=100, draft_blocks=100 if k else 0,
                block_size=4, max_model_len=64, max_num_seqs=4, vocab_size=16, eos=eos,
                k=k, gpu_draft_tokens=device, performance_mode=True, sampling_mode='greedy')
            requests = [Request(str(i), [1, 2, 3] + [i]*i, cap, None, 0.0, eos < 0)
                        for i, cap in enumerate([1, 7, 18, 23])]
            out = decoder.generate(requests)
            assert not decoder.target_pool.used and (decoder.draft_pool is None or not decoder.draft_pool.used)
            return out, target.events, draft.events
        for perturb in [False, True]:
            for eos in [-1, 3]:
                baseline = run(0, False, perturb, eos)[0]
                for k in range(1, 5):
                    a, b = run(k, False, perturb, eos), run(k, True, perturb, eos)
                    assert a == b and a[0] == baseline
                    result['checks'].append(f'device_kv_k{k}_perturb{perturb}_eos{eos}_B_S0_S1_equal')
        # Public entry validates all requests before marking even the first used.
        llm = RandomLLM.__new__(RandomLLM)
        llm.closed = False; llm.sampling_mode = 'greedy'; llm.owner = None
        llm.backend = CUDAGreedyBackend(DeviceRunner('target', False), performance_mode=True)
        llm.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
        llm.decoder = RandomDecode(llm.backend,target_blocks=100,draft_blocks=0,block_size=4,
            max_model_len=64,max_num_seqs=4,vocab_size=16,eos=-1,k=0,sampling_mode='greedy')
        rs = [Request('good', [1], 1, None, 0.0), Request('bad', [2], 1, None, 1.0)]
        try: llm.generate(rs)
        except ValueError: pass
        else: raise AssertionError('mixed temperature accepted')
        assert not any(r.used for r in rs) and not llm.decoder.target_pool.used
        rs[1].temperature=0.0; llm.generate(rs)
        assert all(r.used for r in rs)
        try: llm.generate(rs)
        except ValueError: pass
        else: raise AssertionError('request reused')
        result['checks'].append('public_fail_before_used_and_one_shot_requests')
        assert torch.equal(torch.cuda.get_rng_state(), before_rng)
        result['checks'].append('no_global_cuda_rng_consumption')
        result['status'] = 'PASS'
    except BaseException:
        result.update(status='FAIL', traceback=traceback.format_exc())
    json.dump(result, handle, indent=2); handle.close()
    print(json.dumps({'status': result['status'], 'checks':len(result['checks'])}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    raise SystemExit(main(p.parse_args().output))
