"""CPU arithmetic bridge against frozen old VERIFY; no CUDA claims."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import traceback


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main(output):
    import torch
    from nanovllm.layers import random_sampler as sampler
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    from conftest import reference
    old=reference('r2_random_backend')
    from nanovllm.engine import verify_sampling as candidate
    handle=Path(output).open('x');report={'status':'RUNNING','checks':[]}
    try:
        with torch.inference_mode():
            for vocab in (8,151936):
                for seed in (0,1,7,31):
                    runner=SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab)))
                    backend=old.CUDARandomBackend(runner,runner,performance_mode=True)
                    backend.r2_options=frozenset(('draw','softmax','residual'))
                    r=SimpleNamespace(request_id='case',temperature=1.,rng=old.RequestRNG(seed,'case',device='cpu'))
                    fresh=old.RequestRNG(seed,'case',device='cpu')
                    x=torch.arange(vocab,dtype=torch.float32)
                    logits=torch.stack([-(x.roll(i)%7)*.4 for i in range(4)])
                    q=sampler.from_probs(torch.stack([torch.softmax(-(x.roll(i+1)%5)*.3,0) for i in range(3)]))
                    rows=[sampler.Probabilities(q.values[i:i+1],q.mass[i:i+1],q.invalid[i:i+1]) for i in range(3)]
                    payload=[]
                    def checked(requests,values):
                        payload.append(values.clone())
                        sampler.require_valid_boundary(torch.cat([backend.invalid[r.request_id] for r in requests]))
                        return values.tolist()
                    backend._materialize_checked=checked
                    boundary_failure=False
                    try:backend.verify([r],[logits],{'case':[0,1,2]},{'case':rows})
                    except FloatingPointError:boundary_failure=True
                    p=backend._probabilities(r,logits)
                    u=torch.rand(3,dtype=torch.float64,generator=fresh.streams['accept'])
                    c=torch.empty_like(p.values[-1:],dtype=torch.float64).exponential_(generator=fresh.streams['correction'])
                    b=torch.empty_like(p.values[-1:]).exponential_(generator=fresh.streams['bonus'])
                    operands=(p.values,p.mass,p.invalid,q.values,q.mass,q.invalid,torch.tensor([0,1,2]),u,c,b,torch.zeros(1,dtype=torch.bool))
                    before=[value.clone() for value in operands]
                    count,tail,invalid=candidate.post_pq(*operands)
                    assert torch.equal(torch.cat((count,tail)),payload[0][0])
                    assert torch.equal(invalid,backend.invalid['case'])
                    assert bool(invalid.any())==boundary_failure
                    assert all(torch.equal(g.get_state(),fresh.streams[role].get_state()) for role,g in r.rng.streams.items())
                    assert all(torch.equal(a,b) for a,b in zip(operands,before,strict=True))
                    report['checks'].append({'case':f'vocab{vocab}_seed{seed}_frozen_bridge_rng_ownership',
                                            'both_failed_closed':boundary_failure})
        report['status']='PASS'
    except BaseException:
        report.update(status='FAIL',traceback=traceback.format_exc())
    finally:
        json.dump(report,handle,indent=2);handle.close()
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    raise SystemExit(main(parser.parse_args().output))


def test_post_pq_cpu_bridge(tmp_path):
    assert main(tmp_path / "bridge.json") == 0
