"""Frozen old backend versus the installed new eager/Graph final VERIFY entry.

CUDA is mandatory. Explicit-noise edge cases and real Generator-state checks
are separate tests. This does not measure throughput or prove target numerics.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import traceback
from unittest.mock import patch


def main(output):
    root=Path(__file__).resolve().parent
    handle=Path(output).open('x')
    result={'status':'RUNNING','checks':[]}
    graph=None
    try:
        import torch
        from nanovllm.engine import random_backend as current
        from nanovllm.engine.verify_sampling import post_pq, VerifySamplingGraph
        from nanovllm.layers import random_sampler as sampler
        assert torch.cuda.is_available(),'CUDA mandatory; no CPU fallback'
        spec=importlib.util.spec_from_file_location('frozen_pr5_backend',root.parent/'tests/perf_repair/reference/r2_random_backend.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        device='cuda:0'

        def run(kind, logits, q, ids, seed=11, explicit=None, prior=False):
            cls=old.CUDARandomBackend if kind=='old' else current.CUDARandomBackend
            vocab=logits.shape[1]
            runner=SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab)))
            backend=cls(runner,runner,performance_mode=True)
            backend.r2_options=frozenset(('draw','softmax','residual'))
            r=SimpleNamespace(request_id='case',temperature=1.0,rng=current.RequestRNG(seed,'case',device=device))
            backend.invalid[r.request_id]=torch.tensor([prior],dtype=torch.bool,device=device)
            calls=[];payload=[]
            if kind!='old':
                implementation=post_pq if kind=='eager' else graph
                def counted(*args):
                    calls.append(True)
                    return implementation(*args)
                backend.post_pq_verifier=counted
            boundary=backend._materialize_checked
            def checked(requests, values):
                payload.append((values.clone(),backend.invalid[r.request_id].clone()))
                return boundary(requests,values)
            backend._materialize_checked=checked
            qs=[sampler.Probabilities(q.values[i:i+1],q.mass[i:i+1],q.invalid[i:i+1]) for i in range(3)]
            failure=False;out=None
            def invoke():
                nonlocal failure,out
                try:out=backend.verify([r],[logits],{r.request_id:ids},{r.request_id:qs})
                except FloatingPointError:failure=True
            if explicit is None:
                invoke()
            else:
                uniform,correction,bonus=explicit
                noise_calls=[]
                def fixed_rand(size,**kwargs):
                    assert size==3 and kwargs['dtype']==torch.float64
                    return uniform.clone()
                def fixed_noise(tensor,lambd=1.,*,generator=None):
                    noise=correction if tensor.dtype==torch.float64 else bonus
                    assert tensor.shape==noise.shape and tensor.dtype==noise.dtype
                    noise_calls.append(tensor.dtype)
                    tensor.copy_(noise);return tensor
                with patch.object(torch,'rand',fixed_rand),patch.object(torch.Tensor,'exponential_',fixed_noise):
                    invoke()
                assert noise_calls==[torch.float64,torch.float32]
            assert len(payload)==1
            if kind!='old':assert len(calls)==1,'new final entry was not exercised'
            states={role:g.get_state().clone() for role,g in r.rng.streams.items()}
            return out,failure,payload[0],states

        def compare(logits,q,ids,seed=11,explicit=None,prior=False):
            before=(logits.clone(),q.values.clone(),q.mass.clone(),q.invalid.clone())
            baseline=run('old',logits,q,ids,seed,explicit,prior)
            for kind in ['eager','graph']:
                actual=run(kind,logits,q,ids,seed,explicit,prior)
                assert actual[:2]==baseline[:2]
                assert all(torch.equal(a,b) for a,b in zip(actual[2],baseline[2],strict=True))
                assert all(torch.equal(actual[3][role],baseline[3][role]) for role in baseline[3])
            for a,b in zip(before,(logits,q.values,q.mass,q.invalid),strict=True):
                torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)

        with torch.inference_mode():
            for vocab in [8,151936]:
                graph=VerifySamplingGraph(device=device,vocab_size=vocab)
                base=torch.arange(vocab,device=device,dtype=torch.float32)
                logits=torch.stack([-(base.roll(i)%7)*.4 for i in range(4)])
                qvalues=torch.stack([torch.softmax(-(base.roll(i+1)%5)*.3,0) for i in range(3)])
                q=sampler.from_probs(qvalues)
                for seed in [0,1,7,31]:
                    compare(logits,q,[0,1,2],seed=seed)
                    result['checks'].append(f'vocab{vocab}_seed{seed}_outputs_flags_role_states_input_ownership')
                if vocab==8:
                    uniform=torch.full((3,),.5,dtype=torch.float64,device=device)
                    correction=torch.ones((1,vocab),dtype=torch.float64,device=device)
                    bonus=torch.ones((1,vocab),dtype=torch.float32,device=device)
                    for first_reject in range(4):
                        values=torch.full((4,vocab),-100.,device=device)
                        for i in range(4):values[i,i+1]=0
                        ids=[1,2,3]
                        qpoint=torch.zeros((3,vocab),device=device)
                        for i,t in enumerate(ids):qpoint[i,t]=1
                        if first_reject<3:
                            values[first_reject].fill_(-100.);values[first_reject,7]=0
                        compare(values,sampler.from_probs(qpoint),ids,explicit=(uniform,correction,bonus))
                        result['checks'].append(f'explicit_first_reject_{first_reject}')
                    for case in ['uniform_one','bad_id','nan_logits','zero_q_mass','correction_zero',
                                 'correction_negative','correction_inf','bonus_nan','bonus_zero','prior_invalid']:
                        p=logits.clone();qq=sampler.Probabilities(q.values.clone(),q.mass.clone(),q.invalid.clone())
                        u=uniform.clone();c=correction.clone();b=bonus.clone();ids=[0,1,2]
                        if case=='uniform_one':u[1]=1
                        if case=='bad_id':ids[0]=-1
                        if case=='nan_logits':p[0,0]=float('nan')
                        if case=='zero_q_mass':qq.mass[0]=0;qq.invalid[0]=True
                        if case=='correction_zero':c[0,0]=0
                        if case=='correction_negative':c[0,0]=-1
                        if case=='correction_inf':c[0,0]=float('inf')
                        if case=='bonus_nan':b[0,0]=float('nan')
                        if case=='bonus_zero':b[0,0]=0
                        compare(p,qq,ids,explicit=(u,c,b),prior=case=='prior_invalid')
                        result['checks'].append(case)
                result.setdefault('graphs',[]).append(graph.statistics())
                graph.close();graph=None
        result.update(status='PASS',torch=torch.__version__,device=torch.cuda.get_device_name(),
            scope='final CUDA VERIFY entry, explicit noise and actual role-state equality; not frequency or model gate')
    except BaseException:
        result.update(status='FAIL',traceback=traceback.format_exc())
    finally:
        if graph is not None:
            try:graph.close()
            except BaseException:result['cleanup_error']=traceback.format_exc();result['status']='FAIL'
        json.dump(result,handle,indent=2);handle.close()
    print(json.dumps({'status':result['status'],'checks':len(result['checks'])}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    raise SystemExit(main(p.parse_args().output))
