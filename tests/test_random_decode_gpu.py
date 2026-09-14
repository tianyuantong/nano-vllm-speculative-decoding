"""New controller + actual CUDA sampling adapter, synthetic logits and KV oracle.

Not real Qwen, attention, CUDA Graph, or GPU KV acceptance. No CPU fallback.
Run only after reviewing the new integration patch, in an allocated GPU job.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from random_position_cases import (position_probabilities, ALL_ACCEPT_OUTPUTS,
                                   SECOND_REJECT_OUTPUTS, SECOND_RESIDUAL_ROWS)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open('x') as f: f.write('{}\n')
    result = {'status': 'NOT_RUN', 'scope': 'CUDA adapter + controller with synthetic logits; no Qwen/attention/Graph/GPU KV',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT/'nanovllm/engine/random_decode.py', ROOT/'nanovllm/engine/random_backend.py',
                                          ROOT/'nanovllm/layers/random_sampler.py',
                                          ROOT/'tests/random_position_cases.py', Path(__file__).resolve()]}}
    code = 2
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        result.update(status='RUNNING', torch=torch.__version__, device=torch.cuda.get_device_name(0))
        for name in ('nanovllm', 'nanovllm.engine', 'nanovllm.layers'):
            package = ModuleType(name); package.__path__ = [str(ROOT.joinpath(*name.split('.')))]; sys.modules[name] = package
        control = importlib.import_module('nanovllm.engine.random_decode')
        gpu = importlib.import_module('nanovllm.engine.random_backend')

        class SyntheticRunner:
            def __init__(self, probabilities, bad=False):
                self.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4))
                self.probabilities = probabilities
                self.memory = {}
                self.bad = bad
                self.calls = 0
            def run_queries(self, queries, all_logits=False):
                self.calls += 1
                out = []
                for q in queries:
                    slots = [q.block_table[i//q.block_size]*q.block_size+i%q.block_size for i in range(len(q))]
                    assert [self.memory[s] for s in slots[:q.num_cached_tokens]] == q.tokens[:q.num_cached_tokens]
                    for i in range(q.num_cached_tokens, len(q)): self.memory[slots[i]] = q.tokens[i]
                    positions = range(q.num_cached_tokens, len(q)) if all_logits else [len(q)-1]
                    rows = [self.probabilities(q.request.request_id, q.tokens[:position+1])
                            if callable(self.probabilities) else self.probabilities for position in positions]
                    values = torch.tensor(rows, dtype=torch.float32, device='cuda').log()
                    if self.bad: values.fill_(float('nan'))
                    out.append(values)
                return out

        def run(p, q, k=3, cap=11, eos=3, bad=False, ids=('a', 'b')):
            target = SyntheticRunner(p, bad); draft = SyntheticRunner(q)
            backend = gpu.CUDARandomBackend(target, draft)
            decoder = control.RandomDecode(backend, target_blocks=100, draft_blocks=100, block_size=4,
                                           max_model_len=40, max_num_seqs=4, vocab_size=4, eos=eos, k=k)
            requests = [control.Request(rid, [0, 1, 2], cap, gpu.RequestRNG(1729, rid)) for rid in ids]
            outputs = decoder.generate(requests)
            assert not decoder.target_pool.used
            assert decoder.draft_pool is None or not decoder.draft_pool.used
            return outputs, target, draft

        passed = []
        out, _, _ = run([1,0,0,0], [1,0,0,0])
        assert all(o['token_ids']==[0]*11 for o in out); passed.append('full_accept_bonus_catchup')
        out, _, _ = run([1,0,0,0], [0,1,0,0])
        assert all(o['token_ids']==[0]*11 for o in out); passed.append('reject_residual')
        out, _, draft = run([0,0,0,1], [1,0,0,0])
        assert all(o['token_ids']==[3] for o in out) and draft.calls==0; passed.append('first_eos')
        out, _, draft = run([1,0,0,0], [1,0,0,0], cap=1)
        assert draft.calls==0; passed.append('cap_one')
        p, q = [.7,.3,0,0], [.4,.6,0,0]
        out, _, _ = run(p,q); again, _, _ = run(p,q)
        assert out==again; passed.append('random_s0_replay')
        reverse, _, _ = run(p,q,ids=('b','a'))
        assert out==list(reversed(reverse)); passed.append('request_rng_order_independence_synthetic_logits')
        out, _, _ = run(p,q,k=0); again, _, _ = run(p,q,k=0)
        assert out==again; passed.append('random_b_replay')
        try: run(p,q,bad=True)
        except FloatingPointError: passed.append('nan_aborts')
        else: raise AssertionError('bad logits accepted')
        # Position-varying point masses make acceptance/rejection deterministic.
        # A wrong q row has zero proposal support; a wrong bonus row changes output.
        q_position = lambda rid, prefix: position_probabilities(rid, prefix, role='draft')
        p_position = lambda rid, prefix: position_probabilities(rid, prefix, role='target')
        out, _, _ = run(p_position, q_position, k=2, cap=4, eos=-1)
        assert {o['request_id']: o['token_ids'] for o in out} == ALL_ACCEPT_OUTPUTS
        passed.append('position_varying_all_accept_bonus_and_request_rows')
        reverse, _, _ = run(p_position, q_position, k=2, cap=4, eos=-1, ids=('b','a'))
        assert out == list(reversed(reverse))
        passed.append('position_varying_request_order')
        p_reject = lambda rid, prefix: position_probabilities(rid, prefix, role='target', reject_second=True)
        observed = []
        real_residual = gpu.sampler.residual
        def observed_residual(target, draft, rejected):
            # Test-only observation. Call the real implementation unchanged;
            # retain copies until after generate, so observation adds no inner read.
            observed.append((target.values.clone(), draft.values.clone(), rejected.clone()))
            return real_residual(target, draft, rejected)
        with patch.object(gpu.sampler, 'residual', observed_residual):
            out, _, _ = run(p_reject, q_position, k=2, cap=4, eos=-1)
        assert {o['request_id']: o['token_ids'] for o in out} == SECOND_REJECT_OUTPUTS
        assert len(observed) == 2  # first round's two requests; last-token round is k=0
        for rid, (p_row, q_row, rejected) in zip(('a', 'b'), observed, strict=True):
            expected_p, expected_q = SECOND_RESIDUAL_ROWS[rid]
            assert p_row.tolist() == [expected_p] and q_row.tolist() == [expected_q]
            assert rejected.tolist() == [True]
        passed.append('second_candidate_rejection_selects_its_residual_rows')
        result.update(status='PASS', checks=passed)
        code = 0
    except (ImportError, RuntimeError) as error:
        result['reason'] = str(error)
        if result['status']=='RUNNING': result['status']='FAIL'; code=1
    except Exception as error:
        result.update(status='FAIL', reason=f'{type(error).__name__}: {error}'); code=1
    finally:
        args.result.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'reason': result.get('reason')}))
    return code

if __name__ == '__main__': raise SystemExit(main())
