from types import SimpleNamespace
import pytest
from nanovllm.engine.random_decode import Query, Request, PrivateKVPool
from nanovllm.engine.verify_graph import exact_verify_key, build_verify_metadata


def queries(lengths=(255,256,257,31), qlens=(5,5,5,5)):
    pool = PrivateKVPool(113,256)
    result=[]
    for i,(n,q) in enumerate(zip(lengths,qlens)):
        r=Request(str(i),[17]*n,20,None)
        pool.reserve(r.target,n+q)
        r.target.num_cached_tokens=n
        result.append(Query(r,r.tokens+[18+i]*q,r.target,256))
    return result


@pytest.mark.parametrize("b",range(1,5))
@pytest.mark.parametrize("q",range(2,6))
def test_all_exact_shapes_and_actual_lengths(b,q):
    qs=queries()[:b]
    for x in qs: x.tokens=x.request.tokens+[18]*q
    m=build_verify_metadata(qs,block_size=256,max_model_len=3328,num_blocks=113,vocab_size=32)
    assert exact_verify_key(qs)==(b,q)
    assert m.page_stride==13
    assert m.cu_q==tuple(i*q for i in range(b+1))
    end=0
    for i,x in enumerate(qs):
        end+=len(x)
        assert m.cu_k[i+1]==end
        for j,p in enumerate(range(x.num_cached_tokens,len(x))):
            assert m.slots[i*q+j]==x.block_table[p//256]*256+p%256
        assert m.block_tables[i*13:i*13+len(x.block_table)]==tuple(x.block_table)


def test_mixed_shrink_fallback_and_q1():
    qs=queries((255,256,257),(3,1,2))
    assert exact_verify_key(qs) is None
    uniform=queries((255,256,257),(5,5,5))
    assert exact_verify_key([uniform[0],uniform[2]])==(2,5)
    uniform[0].kv.num_cached_tokens=0
    assert exact_verify_key(uniform) is None
    assert exact_verify_key(queries((31,), (1,))) is None


def test_bad_token_block_or_capacity_rejected():
    opts=dict(block_size=256,max_model_len=3328,num_blocks=113,vocab_size=32)
    qs=queries();qs[0].tokens[-1]=32
    with pytest.raises(ValueError):build_verify_metadata(qs,**opts)
    qs=queries();qs[0].block_table[0]=113
    with pytest.raises(ValueError):build_verify_metadata(qs,**opts)
    qs=queries();qs[0].block_table.clear()
    with pytest.raises(ValueError):build_verify_metadata(qs,**opts)
