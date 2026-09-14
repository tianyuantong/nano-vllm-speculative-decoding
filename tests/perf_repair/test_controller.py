from types import SimpleNamespace
import pytest
import torch

from nanovllm.engine import random_decode as current
from nanovllm.engine.random_backend import CUDARandomBackend, RequestRNG
from conftest import reference

old = reference("random_decode")
old_backend = reference("random_backend")
old_backend.sampler = reference("random_sampler")


class TensorRunner:
    """CPU physical-slot oracle. S1 placeholders are NOT treated as cached IDs."""
    def __init__(self, role, vocab=8):
        self.role, self.vocab = role, vocab
        self.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab))
        self.cells, self.events = {}, []
        self.head_rows = 0

    def run_queries(self, queries, *, all_logits=False, device_input_ids=None, need_logits=True):
        result=[]
        for row, query in enumerate(queries):
            output=[]; consumed=[]
            for pos in range(query.num_cached_tokens, len(query)):
                slot=query.block_table[pos//query.block_size]*query.block_size+pos%query.block_size
                token=(int(device_input_ids[row]) if device_input_ids is not None else query.tokens[pos])
                if pos:
                    previous=query.block_table[(pos-1)//query.block_size]*query.block_size+(pos-1)%query.block_size
                    assert previous in self.cells
                    prefix=self.cells[previous][1]
                else: prefix=0
                rolling=(prefix*17+token+pos)%997
                self.cells[slot]=(token,rolling)
                preferred=(rolling+(1 if self.role=="draft" else 0))%self.vocab
                values=[-0.7*min((i-preferred)%self.vocab,(preferred-i)%self.vocab) for i in range(self.vocab)]
                output.append(values);consumed.append(token)
            self.events.append((query.request.request_id, query.num_cached_tokens, tuple(consumed),
                                tuple(query.block_table)))
            if need_logits:
                selected=output if all_logits else output[-1:]
                self.head_rows+=len(selected)
                result.append(torch.tensor(selected,dtype=torch.float32))
            else:result.append(None)
        return result


def run(module, backend_cls, mode, optimized, eos, seed):
    target=TensorRunner("target");draft=TensorRunner("draft") if mode in ("S0","S1") else None
    if backend_cls is CUDARandomBackend:
        backend=backend_cls(target,draft,performance_mode=optimized)
    else:backend=backend_cls(target,draft)
    backend.synchronize=lambda:None
    opts=dict(target_blocks=40,draft_blocks=40 if draft else 0,block_size=4,
              max_model_len=32,max_num_seqs=4,vocab_size=8,eos=eos,
              k=0 if mode=="B" else 4,gpu_draft_tokens=mode=="S1",ngram=mode=="N")
    if module is current:opts["performance_mode"]=optimized
    decoder=module.RandomDecode(backend,**opts)
    prompts=[[1,2,1],[2,2,2,2],[1,2,3,1,2],[3]]
    caps=[1,6,10,12]
    rs=[module.Request(str(i),p,cap,RequestRNG(seed,str(i),device="cpu"),ignore_eos=eos<0)
        for i,(p,cap) in enumerate(zip(prompts,caps))]
    result=decoder.generate(rs)
    assert not decoder.target_pool.used
    assert decoder.draft_pool is None or not decoder.draft_pool.used
    states={(r.request_id,role):g.get_state().clone() for r in rs for role,g in r.rng.streams.items()}
    return result,states,target,draft


@pytest.mark.parametrize("mode",["B","N","S0","S1"])
@pytest.mark.parametrize("eos",[-1,3])
@pytest.mark.parametrize("seed",[7,9217])
def test_full_controller_against_frozen_source(mode,eos,seed):
    a=run(old,old_backend.CUDARandomBackend,mode,False,eos,seed)
    b=run(current,CUDARandomBackend,mode,True,eos,seed)
    assert a[0]==b[0]
    assert all(torch.equal(a[1][k],b[1][k]) for k in a[1])
    assert a[2].events==b[2].events
    if a[3] is not None:
        assert a[3].events==b[3].events
        assert b[3].head_rows <= a[3].head_rows


def test_invalid_uncached_and_initial_history_fail_before_forward():
    backend=SimpleNamespace()
    d=current.RandomDecode(backend,target_blocks=10,draft_blocks=0,block_size=4,
                           max_model_len=20,max_num_seqs=4,vocab_size=8,eos=-1,k=0,performance_mode=True)
    r=current.Request("a",[1,8],4,None)
    with pytest.raises(ValueError):d._forward("target",[(r,r.tokens)])
    assert not d.target_pool.used
    r=current.Request("a",[1,2],4,None)
    d.target_pool.reserve(r.target,3);r.target.num_cached_tokens=2
    with pytest.raises(ValueError):d._forward("target",[(r,[1,2,-1])])


def test_new_forward_is_only_suffix_scan():
    class Tracked(list):
        def __iter__(self):raise AssertionError("full cached-prefix scan")
    class Backend:
        def forward(self, role, queries, **kwargs):return [torch.zeros((1,8)) for _ in queries]
    d=current.RandomDecode(Backend(),target_blocks=10,draft_blocks=0,block_size=4,
                           max_model_len=20,max_num_seqs=4,vocab_size=8,eos=-1,k=0,performance_mode=True)
    r=current.Request("a",[1,2],4,None);r.target.num_cached_tokens=2
    d._forward("target",[(r,Tracked([1,2,3]))])
    assert r.target.num_cached_tokens==3
