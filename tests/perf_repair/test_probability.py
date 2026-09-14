from types import SimpleNamespace
import pytest
import torch

from nanovllm.layers import random_sampler as new
from nanovllm.engine.random_backend import CUDARandomBackend, RequestRNG
from conftest import reference

old = reference("random_sampler")
old_backend = reference("random_backend")
old_backend.sampler = old
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid", [False, True])
def test_compact_draw_exact_rng_and_storage(device, dtype, invalid):
    weights = torch.tensor([[.5, .3, .2, 0], [0, 0, 0, 1]], device=device, dtype=dtype)
    before = weights.clone()
    p = old.from_probs(weights)
    if invalid:
        p = old.Probabilities(p.values, p.mass, torch.tensor([True, False], device=device))
    g1 = torch.Generator(device=device).manual_seed(92821)
    g2 = torch.Generator(device=device).manual_seed(92821)
    for _ in range(25):
        a = old.draw(p, generator=g1)
        b = new.draw(p, generator=g2, compact=True)
        assert torch.equal(a.token_ids, b.token_ids)
        assert torch.equal(a.invalid, b.invalid)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert torch.equal(weights, before)


def runners(device):
    runner = SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)))
    return runner, runner


def requests(device, ids=("a", "b", "c")):
    return [SimpleNamespace(request_id=rid, temperature=1.0,
                            rng=RequestRNG(7711, rid, device=device)) for rid in ids]


def all_states(rs):
    return {(r.request_id, role): g.get_state().clone() for r in rs for role, g in r.rng.streams.items()}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("point_mass", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 7, 19, 38])
def test_backend_mixed_verify_exact_reference(device, point_mass, seed):
    target, draft = runners(device)
    a = old_backend.CUDARandomBackend(target, draft)
    b = CUDARandomBackend(target, draft, performance_mode=True)
    ra, rb = requests(device), requests(device)
    ds = {"a": [0, 2], "b": [], "c": [1]}
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = [torch.randn((len(ds[r.request_id])+1, 4), generator=gen, device=device) for r in ra]
    if point_mass:
        qa = a.point_mass_proposals(ra, logits, ds)
        qb = b.point_mass_proposals(rb, logits, ds)
    else:
        qa, qb = {}, {}
        for r in ra:
            vals = [torch.softmax(torch.randn((1,4), generator=gen, device=device), -1) for _ in ds[r.request_id]]
            qa[r.request_id] = [old.from_probs(v.clone()) for v in vals]
            qb[r.request_id] = [new.from_probs(v.clone()) for v in vals]
    originals = {rid: [q.values.clone() for q in rows] for rid, rows in qb.items()}
    xa = a.verify(ra, logits, ds, qa)
    xb = b.verify(rb, logits, ds, qb)
    assert xa == xb
    sa, sb = all_states(ra), all_states(rb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    for rid, rows in qb.items():
        assert all(torch.equal(q.values, v) for q, v in zip(rows, originals[rid]))


@pytest.mark.parametrize("device", DEVICES)
def test_batch_boundary_fails_before_return(device):
    target, draft = runners(device)
    backend = CUDARandomBackend(target, draft, performance_mode=True)
    rs = requests(device, ("a", "b"))
    good = torch.tensor([[1., 2., 3., 4.]], device=device)
    bad = torch.full((1,4), float("nan"), device=device)
    with pytest.raises(FloatingPointError):
        backend.sample_batch(rs, [good, bad], "ordinary_target")


@pytest.mark.parametrize("device", DEVICES)
def test_point_mass_endpoints_and_residual(device):
    p = new.from_probs(torch.tensor([[0.,1.,0.,0.], [.5,.3,.2,0.], [1.,0.,0.,0.]], device=device))
    q = new.from_probs(torch.tensor([[1.,0.,0.,0.]]*3, device=device))
    ids = torch.zeros(3, dtype=torch.int64, device=device)
    u = torch.tensor([0., .75, .999], dtype=torch.float64, device=device)
    a = new.accept(p, q, ids, u)
    assert a.accepted.tolist() == [False, False, True]
    r = new.residual(p,q,~a.accepted)
    assert not r.invalid.any().item()
    torch.testing.assert_close(r.values[1], torch.tensor([0.,.6,.4,0.], dtype=torch.float64, device=device), atol=1e-7, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_point_mass_exact_construction_and_bad_id(device):
    target, draft = runners(device)
    a=old_backend.CUDARandomBackend(target,draft)
    b=CUDARandomBackend(target,draft,performance_mode=True)
    rs=requests(device)
    ds={"a":[0,3,1,2],"b":[],"c":[2]}
    logits=[torch.zeros((len(ds[r.request_id])+1,4),device=device) for r in rs]
    qa=a.point_mass_proposals(rs,logits,ds);qb=b.point_mass_proposals(rs,logits,ds)
    for rid in ds:
        for x,y in zip(qa[rid],qb[rid]):
            for field in ("values","mass","invalid"):
                assert torch.equal(getattr(x,field),getattr(y,field))
    ds["c"]=[4]
    with pytest.raises(ValueError): b.point_mass_proposals(rs,logits,ds)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fault", [0.0, float("nan"), float("inf")])
def test_compact_noise_failures_preserve_invalid(device, dtype, fault, monkeypatch):
    # Diagnostic-only fault injection: every original noise/score check remains.
    def faulty_exponential(tensor, *args, **kwargs):
        tensor.fill_(1.0)
        tensor[0, 0] = fault
        return tensor
    monkeypatch.setattr(torch.Tensor, "exponential_", faulty_exponential)
    values=torch.tensor([[.5,.3,.2,0.]],device=device,dtype=dtype)
    before=values.clone();p=old.from_probs(values)
    a=old.draw(p,generator=torch.Generator(device=device).manual_seed(1))
    b=new.draw(p,generator=torch.Generator(device=device).manual_seed(1),compact=True)
    assert a.invalid.tolist()==b.invalid.tolist()==[True]
    assert torch.equal(a.token_ids,b.token_ids)
    assert torch.equal(values,before)
