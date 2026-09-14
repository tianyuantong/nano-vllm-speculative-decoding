"""R2 candidate/reference arithmetic; run on CPU and actual CUDA when available."""
import pytest
import torch
from nanovllm.layers import random_sampler as s

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('vocab', [32, 151936])
def test_softmax_constructor_preserves_values_mass_errors(device, vocab):
    g = torch.Generator(device=device).manual_seed(53)
    logits = torch.randn((4,vocab),device=device,generator=g)
    logits[1,0] = float('nan')
    logits[2,:] = float('-inf')
    t = torch.tensor([1.,1.,1.,0.],device=device)
    old, new = s.from_logits(logits,t), s.from_logits(logits,t,simplify=True)
    torch.testing.assert_close(old.values,new.values,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(old.mass,new.mass,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(old.invalid,new.invalid)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_draw_and_residual_keep_rng_and_valid_tokens(device,dtype):
    values = torch.tensor([[.1,.2,.3,.4],[1.,0.,0.,0.]],device=device,dtype=dtype)
    probabilities = s.from_probs(values)
    for seed in range(10):
        g1 = torch.Generator(device=device).manual_seed(seed)
        g2 = torch.Generator(device=device).manual_seed(seed)
        old = s.draw(probabilities,generator=g1,compact=True)
        new = s.draw(probabilities,generator=g2,simplify=True)
        assert torch.equal(old.token_ids,new.token_ids)
        assert torch.equal(old.invalid,new.invalid)
        assert torch.equal(g1.get_state(),g2.get_state())
    q = s.from_probs(values.flip(-1).contiguous())
    for active in ([True,True],[False,False],[True,False]):
        rejected = torch.tensor(active,device=device)
        old = s.residual(probabilities,q,rejected)
        new = s.residual(probabilities,q,rejected,simplify=True)
        assert torch.equal(old.invalid,new.invalid)
        assert torch.equal(old.values[rejected],new.values[rejected])
        g1 = torch.Generator(device=device).manual_seed(42)
        g2 = torch.Generator(device=device).manual_seed(42)
        a = s.draw(old,generator=g1,compact=True)
        b = s.draw(new,generator=g2,simplify=True)
        assert torch.equal(a.token_ids,b.token_ids)
        assert torch.equal(a.invalid,b.invalid)
        assert torch.equal(g1.get_state(),g2.get_state())


@pytest.mark.parametrize('device', DEVICES)
def test_new_draw_retains_noise_and_score_failure_flags(device,monkeypatch):
    p = s.from_probs(torch.tensor([[.25,.75]],device=device))
    for noise in ([0.,1.],[float('inf'),1.],[float('nan'),1.],[-1.,1.],[1e-45,1.]):
        def injected(self,*args,**kwargs):
            return self.copy_(torch.tensor([noise],device=device))
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'exponential_',injected)
            old=s.draw(p,generator=None,compact=True)
            new=s.draw(p,generator=None,simplify=True)
        assert torch.equal(old.invalid,new.invalid) and new.invalid.all()
