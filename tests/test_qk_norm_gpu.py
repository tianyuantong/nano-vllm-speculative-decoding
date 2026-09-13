"""Actual patched RMSNorm CUDA regression. No CPU fallback, no full model claim."""
import argparse,hashlib,importlib.util,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,default=Path('/projects/nanovllm/models'));a=ap.parse_args()
    with a.result.open('x') as f:f.write('{}\n')
    source=Path(__file__).resolve().parents[1]/'nanovllm/layers/layernorm.py'
    d={'status':'NOT_RUN','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':[]};code=2
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        d.update(status='RUNNING',torch=torch.__version__,device=torch.cuda.get_device_name(0))
        spec=importlib.util.spec_from_file_location('tested_norm',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from safetensors import safe_open
        gen=torch.Generator(device='cuda').manual_seed(9217)
        d['weights']=[]
        with torch.inference_mode():
            for model_name in ['Qwen3-8B','Qwen3-0.6B']:
                folder=a.model_root/model_name
                config=json.loads((folder/'config.json').read_text())
                for role,heads in [('q',config['num_attention_heads']),('k',config['num_key_value_heads'])]:
                    key=f'model.layers.0.self_attn.{role}_norm.weight'
                    weight=None
                    for shard in sorted(folder.glob('*.safetensors')):
                        with safe_open(str(shard),framework='pt',device='cpu') as f:
                            if key in f.keys():
                                weight=f.get_tensor(key);break
                    if weight is None:raise RuntimeError(f'missing real weight: {model_name}/{key}')
                    assert weight.dtype==torch.bfloat16 and tuple(weight.shape)==(128,)
                    d['weights'].append({'model':model_name,'key':key,'heads':heads,
                        'dtype':str(weight.dtype),'sha256':hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()})
                    norm=module.RMSNorm(128,compile_rms=False).to(device='cuda',dtype=torch.bfloat16)
                    norm.weight.copy_(weight)
                    row=torch.randn((1,heads,128),device='cuda',dtype=torch.bfloat16,generator=gen)
                    expected=norm(row).clone()
                    for rows in [1,4,262]:
                        for layout in ['contiguous','strided']:
                            x=row.repeat(rows,1,1)
                            if layout=='strided':
                                storage=torch.zeros((rows,heads*128+512),device='cuda',dtype=torch.bfloat16)
                                x=storage[:,256:256+heads*128].view(rows,heads,128);x.copy_(row.expand_as(x))
                            before=x.clone();actual=norm(x).clone()
                            assert actual.dtype==x.dtype==norm.weight.dtype==torch.bfloat16
                            assert torch.equal(actual,expected.expand_as(actual)) and torch.equal(x,before)
                            for _ in range(3):norm(x)
                            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):captured=norm(x)
                            other=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=gen)
                            saved=[]
                            for label,new in [('A',before),('B',other),('A',before)]:
                                x.copy_(new);reference=norm(x).clone()
                                g.replay();torch.cuda.synchronize();observed=captured.clone()
                                assert torch.equal(observed,reference) and torch.equal(x,new)
                                saved.append(observed)
                            assert torch.equal(saved[0],saved[2]) and not torch.equal(saved[0],saved[1])
                            d['cases'].append({'model':model_name,'role':role,'heads':heads,'rows':rows,'layout':layout,
                                'eager_batch_exact':True,'graph_A_B_A_exact':True,'input_unchanged':True,'dtype':'torch.bfloat16'})
        d['status']='PASS';code=0
    except Exception as error:
        d['reason']=str(error)
        if d['status']=='RUNNING':d['status']='FAIL';code=1
    finally:a.result.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'status':d['status'],'reason':d.get('reason')}));return code
if __name__=='__main__':raise SystemExit(main())
