"""Execute the ACTUAL ModelRunner class without CUDA-only imports on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from nanovllm.utils.context import set_context,get_context,reset_context


def runner_class():
    path=Path(__file__).resolve().parents[2]/"nanovllm/engine/model_runner.py"
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ModelRunner")
    env=dict(torch=torch,Config=object,Sequence=object,DeviceRuntime=object,Event=object,
             set_context=set_context,get_context=get_context,reset_context=reset_context)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),"exec"),env)
    return env["ModelRunner"]


def bare():
    r=runner_class().__new__(runner_class())
    r.world_size=1;r.config=SimpleNamespace(max_num_batched_tokens=100)
    r.verify_graph_cache=None
    r.model=Mock(return_value=torch.zeros((5,8)))
    r.model.compute_logits=Mock(return_value=torch.zeros((5,4)))
    r.prepare_prefill=Mock(return_value=(torch.zeros(5,dtype=torch.int64),torch.arange(5)))
    r.prepare_decode=Mock(return_value=(torch.zeros(1,dtype=torch.int64),torch.zeros(1,dtype=torch.int64)))
    r.run_model=Mock(return_value=torch.zeros((1,4)))
    return r


def test_kv_only_prefill_omits_head():
    r=bare();q=SimpleNamespace(num_scheduled_tokens=5,num_cached_tokens=0)
    assert r.run_queries([q],need_logits=False)==[None]
    assert r.model.call_count==1
    r.model.compute_logits.assert_not_called()


def test_kv_only_single_uses_existing_model_path():
    r=bare();q=SimpleNamespace(num_scheduled_tokens=1,num_cached_tokens=4)
    assert r.run_queries([q],need_logits=False)==[None]
    assert r.run_model.call_args.kwargs=={"need_logits":False}


def test_verify_graph_route_and_mixed_fallback():
    r=bare();q=SimpleNamespace(num_scheduled_tokens=5,num_cached_tokens=4)
    wanted=[torch.ones((5,4))]
    r.verify_graph_cache=SimpleNamespace(run=Mock(return_value=wanted))
    assert r.run_queries([q],all_logits=True) is wanted
    r.model.assert_not_called()
    r.verify_graph_cache.run.return_value=None
    got=r.run_queries([q],all_logits=True)
    assert got[0].shape==(5,4)
    assert r.model.compute_logits.call_args.kwargs=={"all_logits":True}


def test_device_continuation_rejects_prefill():
    import pytest
    r=bare();q=SimpleNamespace(num_scheduled_tokens=5,num_cached_tokens=4)
    with pytest.raises(ValueError):r.run_queries([q],device_input_ids=torch.ones(1,dtype=torch.int64))
    r.model.assert_not_called()
