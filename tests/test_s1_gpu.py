"""CUDA S0/S1 sampling/state comparison with position-dependent synthetic logits.

Diagnostic snapshots synchronize deliberately. This is not a timing or real
Qwen/Graph test. No CPU fallback; run only after engine patch review.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from random_position_cases import position_probabilities


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);a=ap.parse_args()
    with a.result.open('x') as f:f.write('{}\n')
    result={'status':'NOT_RUN','scope':'actual CUDA sampling/backend plus synthetic logits and physical KV oracle; no model Graph'}
    try:
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable'
        root=Path(__file__).resolve().parents[1]
        for name in ['nanovllm','nanovllm.engine','nanovllm.layers']:
            package=ModuleType(name);package.__path__=[str(root.joinpath(*name.split('.')))];sys.modules[name]=package
        control=importlib.import_module('nanovllm.engine.random_decode')
        gpu=importlib.import_module('nanovllm.engine.random_backend')
        def digest(t):
            return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        class Runner:
            def __init__(self, role, reject, random):
                self.role,self.reject,self.random=role,reject,random
                self.config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4))
                self.memory={};self.events=[]
            def run_queries(self, queries, all_logits=False, device_input_ids=None):
                device_values=device_input_ids.tolist() if device_input_ids is not None else None
                out=[]
                for i,q in enumerate(queries):
                    slots=[q.block_table[t//q.block_size]*q.block_size+t%q.block_size for t in range(len(q))]
                    prefix=[self.memory[s] for s in slots[:q.num_cached_tokens]]
                    if device_values is None:
                        assert prefix==q.tokens[:q.num_cached_tokens]
                        tokens=q.tokens
                    else:
                        assert q.num_scheduled_tokens==1
                        tokens=prefix+[device_values[i]]
                    for t in range(q.num_cached_tokens,len(q)):self.memory[slots[t]]=tokens[t]
                    self.events.append((q.request.request_id,q.num_cached_tokens,tuple(tokens),all_logits))
                    positions=range(q.num_cached_tokens,len(q)) if all_logits else [len(q)-1]
                    rows=[]
                    for pos in positions:
                        rows.append(([.7,.3,0,0] if self.role=='target' else [.4,.6,0,0]) if self.random else
                            position_probabilities(q.request.request_id,tokens[:pos+1],role=self.role,reject_second=self.reject))
                    out.append(torch.tensor(rows,device='cuda',dtype=torch.float32).log())
                return out
        def run(device, reject, random, cap, eos, mixed=False):
            target,draft=Runner('target',reject,random),Runner('draft',reject,random)
            backend=gpu.CUDARandomBackend(target,draft);saved=[];verify_events=[]
            original=backend.propose_device
            def propose(rs,logits):
                ids,qs=original(rs,logits)
                saved.append(([r.request_id for r in rs],[digest(t) for t in ids],
                              [[digest(q.values),digest(q.mass),digest(q.invalid)] for q in qs]))
                return ids,qs
            backend.propose_device=propose
            original_verify=backend.verify
            def verify(rs,logits,proposals,qs):
                out=original_verify(rs,logits,proposals,qs)
                verify_events.append((proposals.copy(),out))
                return out
            backend.verify=verify
            decoder=control.RandomDecode(backend,target_blocks=100,draft_blocks=100,block_size=4,
                max_model_len=40,max_num_seqs=4,vocab_size=4,eos=eos,k=2,gpu_draft_tokens=device)
            specs=list(zip(['a','b','c','d'],[6,3,10,1])) if mixed else [('a',cap),('b',cap+1)]
            requests=[control.Request(rid,[0,1,2],limit,gpu.RequestRNG(1729,rid)) for rid,limit in specs]
            out=decoder.generate(requests)
            assert not decoder.target_pool.used and not decoder.draft_pool.used
            if mixed:
                assert saved[0][0]==['a','b','c'] and saved[1][0]==['a','c'], 'middle request did not leave proposal batch'
                assert all('d' not in event[0] for event in saved), 'cap=1 request drafted'
            return (out,saved,verify_events,target.events,draft.events,[r.rng.metadata() for r in requests])
        checks=[]
        for reject,random,cap,eos in [(False,False,4,-1),(True,False,4,-1),(False,False,9,3),
                                    (True,False,9,3),(False,True,11,3),(False,True,1,3)]:
            assert run(False,reject,random,cap,eos)==run(True,reject,random,cap,eos)
            checks.append(dict(reject_second=reject,random=random,cap=cap,eos=eos))
        assert run(False,False,True,10,-1,mixed=True)==run(True,False,True,10,-1,mixed=True)
        checks.append(dict(mixed_caps=[6,3,10,1],first_rows=['a','b','c'],second_rows=['a','c']))
        result.update(status='PASS',checks=checks,torch=torch.__version__,device=torch.cuda.get_device_name(),
            source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [root/'nanovllm/engine/random_decode.py',root/'nanovllm/engine/random_backend.py',
                 root/'nanovllm/layers/random_sampler.py',Path(__file__).resolve()]})
        return 0
    except Exception as e:
        import traceback
        result.update(status='FAIL',reason=str(e),traceback=traceback.format_exc());return 1
    finally:a.result.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':raise SystemExit(main())
