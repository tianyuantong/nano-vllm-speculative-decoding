#!/usr/bin/env python3
"""Same-session original-source vs repair B/N/S0/S1 natural comparison.

Strict input schema: {"groups":[{"group_id":"0","requests":[
 {"request_id":"original-stable-id", "token_ids":[...]} ... four ]}, ...two]}.
Copy IDs from the frozen old debug input; NEVER retokenize or use confirmation
inputs. Default: 2 groups * 4 modes * 2 source builds * 2 timed repeats = 32.
Each build/condition has its own fresh process, full same-seed warmup, and no
hooks/profiler. Public-backend changes are therefore tested against a new B too.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[1]
MODES=('B','N','S0','S1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted((Path(root)/'nanovllm').rglob('*.py'))}


def states(requests):
    return {r.request_id:r.rng.metadata()['states_sha256'] for r in requests}


def dynamo_snapshot():
    import torch._dynamo.utils as u
    # Snapshot counter VALUES, not a live Counter object. Raw compiler logs stay
    # in this worker's unmodified stderr, stage markers are flushed separately.
    return {str(k):{str(a):int(b) for a,b in v.items()} for k,v in u.counters.items()}


def marker(phase):
    print(json.dumps({'phase':phase,'time_ns':time.time_ns(),
                      'perf_counter_ns':time.perf_counter_ns(),'pid':os.getpid()}),flush=True)


def worker(task_path):
    task=json.loads(Path(task_path).read_text())
    root=Path(task['engine_root']).resolve()
    sys.path.insert(0,str(root))  # before ANY nanovllm import
    result={'status':'RUNNING','task':task,'rows':[], 'cleanup_complete':False}
    out=Path(task['result']);handle=out.open('x',encoding='utf-8')
    llm=None;error=None
    try:
        import torch
        import flash_attn
        from nanovllm.config import Config
        from nanovllm.engine.random_llm import RandomLLM
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if engine_hashes(root)!=task['source_hashes']:raise RuntimeError('source binding mismatch')
        common=dict(max_num_batched_tokens=5328,max_num_seqs=4,max_model_len=3328,enable_prefix_cache=False)
        target=Config(task['target'],kv_cache_memory_bytes=4<<30,**common)
        draft=Config(task['draft'],kv_cache_memory_bytes=1536<<20,**common) if task['mode'] in ('S0','S1') else None
        kwargs=dict(k=0 if task['mode']=='B' else 4,ngram=task['mode']=='N',gpu_draft_tokens=task['mode']=='S1')
        if task['build']=='repair':kwargs.update(performance_mode=True,verify_graphs=True)
        result['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'flash_attn':flash_attn.__version__,
                               'device':torch.cuda.get_device_name(0),'pid':os.getpid()}
        marker('LOAD_BEGIN');start=time.perf_counter()
        llm=RandomLLM(target,draft,**kwargs)
        decoder=llm.decoder
        if (decoder.k != kwargs['k'] or decoder.ngram != kwargs['ngram']
                or decoder.gpu_draft_tokens != kwargs['gpu_draft_tokens']
                or (decoder.draft_pool is not None) != (draft is not None)):
            raise RuntimeError('actual mode route does not match task')
        result['effective']={'mode':task['mode'],'k':decoder.k,'ngram':decoder.ngram,
             'gpu_draft_tokens':decoder.gpu_draft_tokens,'temperature':1.0,'ignore_eos':False,
             'cap':task['cap'],'eos':decoder.eos,'prefix_cache':target.enable_prefix_cache,
             'max_num_batched_tokens':target.max_num_batched_tokens,'max_model_len':target.max_model_len,
             'target_kv_bytes':target.kv_cache_memory_bytes,
             'draft_kv_bytes':draft.kv_cache_memory_bytes if draft else 0}
        result['load_seconds']=time.perf_counter()-start;marker('LOAD_END')
        result['actual_blocks']={'target':target.num_kvcache_blocks,'draft':draft.num_kvcache_blocks if draft else 0}
        result['init_peak']={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        prompts=[r['token_ids'] for r in task['requests']]
        ids=[r['request_id'] for r in task['requests']]
        def prepare():
            return llm.prepare_requests(prompts,request_ids=ids,seed=task['seed'],max_tokens=task['cap'],
                                        temperature=1.0,ignore_eos=False)
        def empty():
            if llm.decoder.target_pool.used or (llm.decoder.draft_pool is not None and llm.decoder.draft_pool.used):
                raise RuntimeError('pool not naturally released')
        marker('WARM_BEGIN');rs=prepare();before_warm=states(rs);start=time.perf_counter()
        warm=llm.generate(rs);torch.cuda.synchronize()
        result['warm_seconds']=time.perf_counter()-start;empty()
        after_warm=states(rs);marker('WARM_END')
        if task['build']=='repair':llm.freeze_performance_caches()
        perf0=llm.performance_metadata() if task['build']=='repair' else None
        result['warm_graph']=perf0
        for repeat in range(task['repeats']):
            rs=prepare();before=states(rs);compile_before=dynamo_snapshot()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            marker(f'GENERATE_{repeat}_BEGIN')
            start=time.perf_counter();outputs=llm.generate(rs);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            marker(f'GENERATE_{repeat}_END')
            after=states(rs);compile_after=dynamo_snapshot();empty()
            row={'repeat':repeat,'generate_seconds':elapsed,'outputs':outputs,
                 'actual_tokens':sum(len(o['token_ids']) for o in outputs),
                 'rng_before':before,'rng_after':after,
                 'compile_before':compile_before,'compile_after':compile_after,
                 'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved()}
            # Retain the failing row before reporting the mismatch.
            result['rows'].append(row)
            if outputs!=warm or before!=before_warm or after!=after_warm:
                raise RuntimeError('warm/timed output-stop-RNG mismatch')
            if task['build']=='repair':
                perf=llm.performance_metadata();row['performance']=perf
                a=(perf0.get('verify_graph') or {}).get('counts',{}).get('captures',0)
                b=(perf.get('verify_graph') or {}).get('counts',{}).get('captures',0)
                if a!=b:raise RuntimeError('VERIFY capture occurred in timed generation')
        if engine_hashes(root)!=task['source_hashes']:raise RuntimeError('source changed during worker')
        result['status']='PASS'
    except BaseException as exc:
        error=exc;result['status']='FAIL';result['error']=repr(exc);result['traceback']=traceback.format_exc()
    finally:
        if llm is not None:
            try:llm.close();result['cleanup_complete']=True
            except BaseException as exc:
                result['status']='FAIL';result['cleanup_error']=repr(exc);error=error or exc
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n');handle.close()
    if error is not None:raise error


def identity(row):return row['outputs'],row['rng_before'],row['rng_after']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',type=Path)
    p.add_argument('--reference-root',type=Path)
    p.add_argument('--target');p.add_argument('--draft');p.add_argument('--inputs',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--seed',type=int,default=17011)
    p.add_argument('--cap',type=int,default=512)
    p.add_argument('--repeats',type=int,choices=[1,2],default=2)
    p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--worker-seconds',type=int,default=300)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.worker:return worker(args.worker)
    if not all((args.reference_root,args.target,args.draft,args.inputs,args.output_dir)):
        p.error('comparison requires reference-root, target, draft, inputs and output-dir')
    inputs=json.loads(args.inputs.read_text());groups=inputs.get('groups')
    if not isinstance(groups,list) or len(groups)!=2:raise ValueError('exactly the frozen two debug groups required')
    if args.cap <= 0:raise ValueError('positive cap required')
    if any('group_id' not in g for g in groups) or len({str(g['group_id']) for g in groups})!=2:
        raise ValueError('two distinct group IDs required')
    seen=set()
    for group in groups:
        rs=group.get('requests')
        if not isinstance(rs,list) or len(rs)!=4:raise ValueError('each group must have four requests')
        for r in rs:
            rid=r.get('request_id');tokens=r.get('token_ids')
            if not isinstance(rid,str) or not rid or rid in seen:raise ValueError('unique original stable IDs required')
            seen.add(rid)
            if not isinstance(tokens,list) or not tokens or any(type(t) is not int or t<0 for t in tokens):
                raise ValueError('frozen integer token IDs required; text inputs not accepted')
        if sum(len(r['token_ids']) for r in rs)>5328:raise ValueError('input exceeds frozen B4 prefill budget')
        if max(len(r['token_ids']) for r in rs)+args.cap>3328:raise ValueError('input exceeds context envelope')
    reference_root=args.reference_root.resolve()
    if reference_root==ROOT:raise ValueError('reference must be the unmodified attachment snapshot, not repair root')
    source={'reference':engine_hashes(reference_root),'repair':engine_hashes(ROOT)}
    expected=ROOT/'tests/perf_repair/reference_engine_sha256.json'
    if expected.exists() and source['reference']!=json.loads(expected.read_text()):
        raise ValueError('reference source does not match the supplied attachment identity')
    if not source['reference']:raise ValueError('reference nanovllm sources missing')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plan=[]
    for gi,g in enumerate(groups):
        for mi,mode in enumerate(MODES):
            order=('reference','repair') if (gi+mi)%2==0 else ('repair','reference')
            for build in order:
                ident=f'g{gi}_{mode}_{build}'
                task={'run_id':ident,'group_id':str(g['group_id']),'mode':mode,'build':build,
                      'seed':args.seed,'cap':args.cap,'repeats':args.repeats,'requests':g['requests'],
                      'input_file_sha256':sha(args.inputs),'source_hashes':source[build],
                      'engine_root':str(reference_root if build=='reference' else ROOT),
                      'target':str(Path(args.target).resolve()),'draft':str(Path(args.draft).resolve()),
                      'result':str((args.output_dir/f'{ident}.json').resolve())}
                path=args.output_dir/f'{ident}.task.json';path.write_text(json.dumps(task,ensure_ascii=False,indent=2))
                plan.append((task,path))
    (args.output_dir/'plan.json').write_text(json.dumps([t for t,_ in plan],ensure_ascii=False,indent=2))
    if args.dry_run:
        print('DRY_RUN',len(plan),'processes',len(plan)*args.repeats,'timed calls');return
    started=time.monotonic();results={};ledger={'status':'RUNNING','completed':[],'attempts':[]}
    try:
        for task,path in plan:
            remaining=args.max_seconds-(time.monotonic()-started)
            if remaining<=60:raise TimeoutError('budget limit; no new worker started')
            log=args.output_dir/f"{task['run_id']}.log"
            with log.open('x') as handle:
                attempt={'run_id':task['run_id'],'started_ns':time.time_ns()};ledger['attempts'].append(attempt)
                proc=subprocess.run([sys.executable,str(SCRIPT),'--worker',str(path.resolve())],stdout=handle,stderr=subprocess.STDOUT,
                                    timeout=min(args.worker_seconds,remaining-30),check=False)
            attempt['returncode']=proc.returncode
            if proc.returncode:raise RuntimeError(f"worker failed: {task['run_id']}")
            data=json.loads(Path(task['result']).read_text())
            if data['status']!='PASS' or not data['cleanup_complete'] or len(data['rows'])!=args.repeats:
                raise RuntimeError('incomplete result')
            results[(task['group_id'],task['mode'],task['build'])]=data
            ledger['completed'].append({'run_id':task['run_id'],'result_sha256':sha(task['result']),
                                        'task_sha256':sha(path),'log_sha256':sha(log)})
            base=(task['group_id'],task['mode'])
            if all((*base,b) in results for b in ('reference','repair')):
                left=results[(*base,'reference')]['rows'];right=results[(*base,'repair')]['rows']
                if any(identity(a)!=identity(b) for a,b in zip(left,right)):
                    raise RuntimeError(f'original/repair trajectory mismatch at {base}; no speedup promotion')
            for build in ('reference','repair'):
                k0=(task['group_id'],'S0',build);k1=(task['group_id'],'S1',build)
                if k0 in results and k1 in results:
                    if any(identity(a)!=identity(b) for a,b in zip(results[k0]['rows'],results[k1]['rows'])):
                        raise RuntimeError(f'S0/S1 trajectory mismatch at {task["group_id"]}/{build}')
        summary={}
        for build in ('reference','repair'):
            summary[build]={}
            for mode in MODES:
                rows=[v['rows'] for k,v in results.items() if k[1:]==(mode,build)]
                T=sum(sum(r['generate_seconds'] for r in cond)/len(cond) for cond in rows)
                N=sum(sum(r['actual_tokens'] for r in cond)/len(cond) for cond in rows)
                summary[build][mode]={'mean_repeat_then_sum_seconds':T,'mean_repeat_then_sum_tokens':N,'tokens_per_second':N/T}
        ledger['summary']=summary;ledger['status']='PASS'
    except BaseException as exc:
        ledger['status']='FAIL';ledger['error']=repr(exc)
        raise
    finally:
        ledger['wall_seconds']=time.monotonic()-started
        (args.output_dir/'completion.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
