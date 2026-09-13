"""Actual CUDA N point-mass construction and shared verifier; no model proof."""
import argparse,hashlib,importlib,json,sys,traceback
from pathlib import Path
from types import ModuleType,SimpleNamespace


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);a=ap.parse_args()
    with a.result.open('x') as f:f.write('{}\n')
    result={'status':'NOT_RUN'}
    try:
        import torch
        assert torch.cuda.is_available()
        root=Path(__file__).resolve().parents[1]
        for name in ['nanovllm','nanovllm.engine','nanovllm.layers']:
            package=ModuleType(name);package.__path__=[str(root.joinpath(*name.split('.')))];sys.modules[name]=package
        gpu=importlib.import_module('nanovllm.engine.random_backend')
        target=SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)))
        b=gpu.CUDARandomBackend(target)
        cases=[('reject_second',[0,1,2],[0,3,2,1],[0,3]),
               ('all_accept',[0,1,2],[0,1,2,3],[0,1,2,3]),
               ('reject_first',[0,1],[3,1,2],[3]),
               ('no_match',[],[2],[2])]
        rs=[];logits=[];proposals={};draft_before=[]
        for name,ds,positions,expected in cases:
            r=SimpleNamespace(request_id=name,temperature=1.,rng=gpu.RequestRNG(17011,name))
            rs.append(r);proposals[name]=ds
            logits.append(torch.nn.functional.one_hot(torch.tensor(positions,device='cuda'),4).float().log())
            draft_before.append(r.rng.metadata()['states_sha256']['draft'])
        q=b.point_mass_proposals(rs,logits,proposals)
        for r in rs:
            for token,row in zip(proposals[r.request_id],q[r.request_id],strict=True):
                expected=torch.nn.functional.one_hot(torch.tensor([token],device='cuda'),4).float()
                assert torch.equal(row.values,expected) and row.mass.item()==1 and not row.invalid.item()
        out=b.verify(rs,logits,proposals,q)
        assert out==[c[3] for c in cases],out
        assert draft_before==[r.rng.metadata()['states_sha256']['draft'] for r in rs]
        # Interior probability example uses the production q constructor and
        # the same residual primitive used by verify, without a second sampler.
        r=SimpleNamespace(request_id='interior',temperature=1.,rng=gpu.RequestRNG(17011,'interior'))
        values=torch.tensor([[.5,.3,.2,0.],[0.,0.,0.,1.]],device='cuda').log()
        qr=b.point_mass_proposals([r],[values],{'interior':[0]})['interior'][0]
        pr=b._probabilities(r,values[:1])
        decision=gpu.sampler.accept(pr,qr,torch.tensor([0],device='cuda'),
                                    torch.tensor([.75],dtype=torch.float64,device='cuda'))
        assert not decision.accepted.item()
        residual=gpu.sampler.residual(pr,qr,torch.tensor([True],device='cuda'))
        assert torch.allclose(residual.values,torch.tensor([[0.,.6,.4,0.]],dtype=torch.float64,device='cuda'),atol=1e-6,rtol=0)
        assert not residual.invalid.item()
        zero_p=b._probabilities(r,torch.tensor([[0.,0.,0.,1.]],device='cuda').log())
        endpoint=gpu.sampler.accept(zero_p,qr,torch.tensor([0],device='cuda'),
                                    torch.zeros(1,dtype=torch.float64,device='cuda'))
        assert not endpoint.accepted.item(), 'p(candidate)=0 and u=0 must reject'
        # Input probabilities and saved q remain intact across shared verifier.
        for r in rs:
            for token,row in zip(proposals[r.request_id],q[r.request_id],strict=True):
                assert row.values[0,token].item()==1 and row.values.sum().item()==1
        b.synchronize()
        # Exercise the real row splitter between controller and shared verifier.
        # Only model arithmetic/input preparation is synthetic in this test.
        control=importlib.import_module('nanovllm.engine.random_decode')
        ModelRunner=importlib.import_module('nanovllm.engine.model_runner').ModelRunner
        runner=ModelRunner.__new__(ModelRunner)
        runner.world_size=1
        runner.config=SimpleNamespace(max_num_batched_tokens=5328,hf_config=SimpleNamespace(vocab_size=4))
        def point_logits(ids):
            return torch.nn.functional.one_hot(torch.tensor(ids,device='cuda'),4).float().log()
        matrix=point_logits([0,2,1,3,3,0])
        runner.prepare_prefill=lambda queries:(torch.zeros(sum(q.num_scheduled_tokens for q in queries),device='cuda',dtype=torch.int64),None)
        runner.prepare_decode=lambda queries:(torch.zeros(len(queries),device='cuda',dtype=torch.int64),None)
        runner.run_model=lambda *args:point_logits([1])
        class Model:
            def __call__(self,*args):return matrix
            def compute_logits(self,hidden,*,all_logits):
                assert all_logits
                return hidden
        runner.model=Model()
        backend=gpu.CUDARandomBackend(runner)
        decoder=control.RandomDecode(backend,target_blocks=20,draft_blocks=0,block_size=2,
            max_model_len=30,max_num_seqs=4,vocab_size=4,eos=3,k=4,ngram=True)
        requests=[control.Request(name,[0,1],cap,gpu.RequestRNG(17011,name))
                  for name,cap in [('a',4),('b',2),('c',4),('d',1)]]
        for request in requests:
            decoder.target_pool.reserve(request.target,2);request.target.num_cached_tokens=2
            decoder._commit(request,[2])
        active=[request for request in requests if not request.stop]
        assert [r.request_id for r in active]==['a','b','c']
        proposals={'a':[0,1],'b':[],'c':[3,0]}
        decoder._verify_proposals(active,proposals,None)
        assert [len(proposals[r.request_id]) for r in active]==[2,0,1]
        assert [r.tokens[2:] for r in requests]==[[2,0,2],[2,3],[2,3],[2]]
        assert requests[0].target.num_cached_tokens==4
        assert requests[1].stop=='eos' and requests[2].stop=='eos'
        remaining=[r for r in requests if not r.stop]
        assert [r.request_id for r in remaining]==['a']
        decoder._verify_proposals(remaining,{'a':[]},None)
        assert requests[0].tokens[2:]==[2,0,2,1] and requests[0].stop=='length'
        assert not decoder.target_pool.used and decoder.draft_pool is None
        result['mixed_controller_actual_runner_split']='PASS: candidates 2/0/1, rows 3/1/2, offsets 0/3/4'
        result.update(status='N_CUDA_WIRING_PASS',outputs=out,cases=[x[0] for x in cases],
                      draft_rng_unchanged=True,torch=torch.__version__,device=torch.cuda.get_device_name())
        return 0
    except Exception as e:
        result.update(status='FAIL',reason=str(e),traceback=traceback.format_exc());return 1
    finally:
        result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        a.result.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':raise SystemExit(main())
