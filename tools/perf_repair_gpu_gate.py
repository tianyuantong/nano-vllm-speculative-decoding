#!/usr/bin/env python3
"""Real Qwen/FA2 gate for the repair package. NOT a performance benchmark.

Run in the existing 2.9.1/cu130 + FA2.8.3 environment. No remote operations,
model download, optimizer, profiler, or source mutation. Exits nonzero on the
first discrepancy. Artifacts contain observations, not a fabricated PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()



def snapshot_kv_cpu(kv, blocks):
    """Exact selected-block snapshot without a full GPU index_select/clone.

    This is diagnostic code, never the generation/performance path. Copy one
    block at a time into host-owned storage; preserve all bytes, including the
    uninitialized tail already checked by the original gate.
    """
    import torch
    blocks = tuple(blocks)
    if kv.ndim < 3 or not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("nonempty distinct KV blocks required")
    if any(type(b) is not int or not 0 <= b < kv.shape[2] for b in blocks):
        raise ValueError("KV block out of range")
    shape = (*kv.shape[:2], len(blocks), *kv.shape[3:])
    host = torch.empty(shape, dtype=kv.dtype, device="cpu")
    for i, block in enumerate(blocks):
        host.select(2, i).copy_(kv.select(2, block), non_blocking=False)
    return host


def restore_kv_cpu(kv, blocks, host):
    """Restore the same addresses; no full-size GPU restoration temporary."""
    blocks = tuple(blocks)
    if (kv.ndim < 3 or not blocks or len(set(blocks)) != len(blocks)
            or any(type(b) is not int or not 0 <= b < kv.shape[2] for b in blocks)):
        raise ValueError("invalid KV blocks")
    shape = (*kv.shape[:2], len(blocks), *kv.shape[3:])
    if host.device.type != "cpu" or host.dtype != kv.dtype or tuple(host.shape) != shape:
        raise ValueError("CPU KV snapshot layout mismatch")
    for i, block in enumerate(blocks):
        kv.select(2, block).copy_(host.select(2, i), non_blocking=False)


def memory_snapshot():
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True);p.add_argument('--draft',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    # Do not overwrite an earlier result, even if this invocation later fails.
    handle=out.open('x',encoding='utf-8')
    report={'status':'RUNNING','started_ns':time.time_ns(),'cases':[], 'cleanup_complete':False}
    llm=None;error=None
    try:
        import torch
        import flash_attn
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
        if not str(torch.__version__).startswith('2.9.1') or str(flash_attn.__version__).split('+')[0]!='2.8.3':
            raise RuntimeError('This GPU gate is pinned to Torch2.9.1 / FlashAttention2.8.3')
        from nanovllm.config import Config
        from nanovllm.engine.random_llm import RandomLLM
        from nanovllm.engine.random_decode import Query
        from nanovllm.engine.verify_graph import build_verify_metadata
        from nanovllm.utils.context import reset_context
        common=dict(max_num_seqs=4,max_num_batched_tokens=5328,max_model_len=3328,enable_prefix_cache=False)
        llm=RandomLLM(Config(args.target,kv_cache_memory_bytes=4<<30,**common),
                      Config(args.draft,kv_cache_memory_bytes=1536<<20,**common),
                      k=4,performance_mode=True,verify_graphs=True)
        runner=llm.backend.runners['target'];cache=runner.verify_graph_cache
        pool=llm.decoder.target_pool
        report['environment']={'torch':torch.__version__,'flash_attn':flash_attn.__version__,
                               'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}

        def diagnostic_memory_snapshot():
            return {'allocated':torch.cuda.memory_allocated(),
                    'reserved':torch.cuda.memory_reserved(),
                    'free':torch.cuda.mem_get_info()[0],
                    'graph':cache.statistics()}
        report['memory_at_enable']=diagnostic_memory_snapshot()

        def release(rs):
            for r in rs:llm.decoder._release(r)
            assert not pool.used
            assert not llm.decoder.draft_pool.used

        def component(b,qlens,variant=0):
            report['active_case']={'b':b,'q':qlens,'variant':variant,
                                   'entry_memory':diagnostic_memory_snapshot()}
            # Variant changes actual K from capture-time short prefixes up to
            # 2774 including q5, while total B4 prefill stays within 5328.
            lens=([31,255,256,257] if not variant else [2769,1007,1027,31])[:b]
            rs=llm.prepare_requests([[17+variant]*n for n in lens],request_ids=[f'g{i}' for i in range(b)],
                                    seed=9217,max_tokens=32)
            llm.decoder._forward('target',[(r,r.prompt) for r in rs])
            queries=[]
            for i,(r,q) in enumerate(zip(rs,qlens)):
                ids=[18+variant+i+j for j in range(q)]
                pool.reserve(r.target,len(r.tokens)+q)
                queries.append(Query(r,r.tokens+ids,r.target,256))
            blocks=sorted(pool.used)
            # The old gate retained base+expected KV on GPU, and created
            # additional full-sized index_select/clone temporaries. Those
            # diagnostics polluted the process-reserved Graph admission guard.
            base=snapshot_kv_cpu(runner.kv_cache,blocks)
            def restore():
                restore_kv_cpu(runner.kv_cache,blocks,base)
                torch.cuda.synchronize()
            row={'b':b,'q':qlens,'prefix_lengths':lens,'variant':variant,
                 'status':'RUNNING','snapshot_device':'cpu',
                 'snapshot_bytes_each':base.numel()*base.element_size(),
                 'memory_before':memory_snapshot()}
            report['cases'].append(row)  # retain this case even on failure
            captures=cache.counts['captures']
            expected=None
            try:
                restore();cache.enabled=False
                reference=runner.run_queries(queries,all_logits=True)
                reference=[t.clone() for t in reference]
                expected=snapshot_kv_cpu(runner.kv_cache,blocks)
                restore();cache.enabled=True
                report['active_case']['before_graph']=diagnostic_memory_snapshot()
                actual=runner.run_queries(queries,all_logits=True)
                report['active_case']['after_graph']=diagnostic_memory_snapshot()
                # Retain returned tensors over a second replay; they must own
                # independent storage, unlike graph internal output views.
                saved=[t.clone() for t in actual]
                assert all(torch.equal(a,r) for a,r in zip(actual,reference)), 'Graph vs existing eager logits differ'
                actual_kv=snapshot_kv_cpu(runner.kv_cache,blocks)
                assert torch.equal(actual_kv.view(torch.uint8),
                                   expected.view(torch.uint8)), 'Graph vs eager KV bytes differ'
                del actual_kv
                restore();again=runner.run_queries(queries,all_logits=True)
                assert all(torch.equal(a,r) for a,r in zip(again,reference)), 'repeat logits differ'
                assert all(torch.equal(a,s) for a,s in zip(actual,saved)), 'returned logits were overwritten'
                row.update(new_captures=cache.counts['captures']-captures,
                           logits_sha256=[digest(t) for t in actual])
                if len(set(qlens))==1 and 2<=qlens[0]<=5:
                    key=(b,qlens[0])
                    assert key in cache.entries, (
                        'requested exact Graph not admitted: '
                        + repr({'key':key,'blocked':cache.blocked.get(key),
                                'memory_checks':cache.statistics()['memory_checks'][-2:]}))
                    entry=cache.entries[key]
                    meta=build_verify_metadata(queries,block_size=256,max_model_len=3328,
                                               num_blocks=runner.config.num_kvcache_blocks,
                                               vocab_size=runner.config.hf_config.vocab_size)
                    assert entry.ids.tolist()==list(meta.input_ids)
                    assert entry.positions.tolist()==list(meta.positions)
                    assert entry.slots.tolist()==list(meta.slots)
                    assert entry.cu_q.tolist()==list(meta.cu_q)
                    assert entry.cu_k.tolist()==list(meta.cu_k)
                    assert entry.tables.flatten().tolist()==list(meta.block_tables)
                    assert entry.max_seqlen_k==3328 and entry.tables.stride()==(13,1)
                    # Saved external logits must survive genuinely CHANGED graph
                    # input on the same storage, not just an identical replay.
                    changed=[Query(q.request,q.tokens[:-1]+[q.tokens[-1]+1],q.kv,256) for q in queries]
                    restore();runner.run_queries(changed,all_logits=True)
                    assert all(torch.equal(a,s) for a,s in zip(actual,saved))
                    # A -> B -> A with restored KV, changing positions/pages is
                    # additionally exercised by the second q5 pass below.
                    restore();back=runner.run_queries(queries,all_logits=True)
                    assert all(torch.equal(a,r) for a,r in zip(back,reference))
                    row['graph_verified']=True
                else:
                    assert cache.counts['captures']==captures
                    row['eager_fallback_verified']=True
                row['status']='PASS'
            finally:
                if row['status']=='RUNNING':row['status']='FAIL'
                row['memory_after']=memory_snapshot()
                report['graph']=cache.statistics()
                cache.enabled=True
                release(rs)
                del base,expected
                reset_context()

        # q5 first: the dominant B1 tail is tested, not just a B4 demo.
        for q in (5,4,3,2):
            for b in (1,2,3,4):component(b,[q]*b)
        # Same graph keys, different real lengths and different block allocation.
        for b in (1,2,3,4):
            pool.free.reverse()  # test-only free-list permutation; no active KV
            component(b,[5]*b,variant=1)
        component(3,[3,1,2])
        cache.freeze()
        capture_count=cache.counts['captures']

        def natural(optimized,device_tokens):
            llm.performance_mode=optimized
            llm.backend.performance_mode=optimized
            llm.decoder.performance_mode=optimized
            llm.decoder.gpu_draft_tokens=device_tokens
            cache.enabled=optimized
            rs=llm.prepare_requests([[17]*n for n in [255,256,257,31]],
                                    request_ids=['a','b','c','d'],seed=9217,max_tokens=12)
            for r,cap in zip(rs,[1,6,10,12]):r.max_tokens=cap
            output=llm.generate(rs)
            states={r.request_id:r.rng.metadata()['states_sha256'] for r in rs}
            assert not pool.used and not llm.decoder.draft_pool.used
            return {'output':output,'rng':states}
        # All modes consume the same per-role RNG protocol. No model hook.
        a=natural(False,False);a2=natural(False,False)
        b=natural(True,False);c=natural(True,True);c2=natural(True,True)
        assert a==a2==b==c==c2, 'natural output/stop/RNG mismatch'
        assert cache.counts['captures']==capture_count, 'capture occurred after freeze'
        report['natural_short']=a
        report['graph']=cache.statistics()
        report['memory']={'max_allocated':torch.cuda.max_memory_allocated(),
                          'max_reserved':torch.cuda.max_memory_reserved(),
                          'device_free_after':torch.cuda.mem_get_info()[0]}
        report['status']='PASS'
    except BaseException as exc:
        error=exc;report['status']='FAIL';report['error']=repr(exc)
        if llm is not None:
            report['failure_memory']=diagnostic_memory_snapshot()
    finally:
        if llm is not None:
            try:llm.close();report['cleanup_complete']=True
            except BaseException as exc:
                report['cleanup_error']=repr(exc);report['status']='FAIL';error=error or exc
        report['finished_ns']=time.time_ns()
        json.dump(report,handle,ensure_ascii=False,indent=2);handle.write('\n');handle.close()
    if error is not None:raise error


if __name__=='__main__':main()
