"""CPU state evidence only: device tokens are opaque handles, not CUDA validation."""
import unittest
from test_random_decode import PrefixOracle, engine, request, module


class DeviceToken:
    def __init__(self, value):
        self.value = value


class DeviceOracle(PrefixOracle):
    def propose_device(self, requests, logits):
        values, q = super().propose(requests, logits)
        return [DeviceToken(v) for v in values], q

    def forward_draft_device(self, queries, token_ids):
        actual = []
        for q, token in zip(queries, token_ids, strict=True):
            # The GPU consumes the previous token; reconstruct the prefix from
            # physical KV, independently of the controller's placeholder view.
            prefix = [self.memory['draft'][q.block_table[i//q.block_size]*q.block_size+i%q.block_size]
                      for i in range(q.num_cached_tokens)]
            actual.append(module.Query(q.request, prefix+[token.value], q.kv, q.block_size))
        return self.forward('draft', actual)

    def materialize_proposals(self, proposals):
        return {rid: [t.value for t in values] for rid, values in proposals.items()}


class S1StateTests(unittest.TestCase):
    def compare(self, **kwargs):
        b0, b1 = PrefixOracle(**kwargs), DeviceOracle(**kwargs)
        e0, e1 = engine(b0), engine(b1, gpu_draft_tokens=True)
        r0 = [request(str(i),prompt=list(range(1,n+1)),cap=cap)
              for i,(n,cap) in enumerate([(3,1),(4,6),(5,12),(7,17)])]
        r1 = [request(r.request_id,prompt=r.prompt,cap=r.max_tokens) for r in r0]
        self.assertEqual(e0.generate(r0),e1.generate(r1))
        self.assertEqual(b0.events,b1.events)
        self.assertEqual(b0.draw_steps,b1.draw_steps)
        for e in [e0,e1]:
            self.assertFalse(e.target_pool.used or e.draft_pool.used)

    def test_rejections_full_accept_caps_and_cross_block(self):
        for reject in [None,0,1,2,3]:
            with self.subTest(reject=reject):
                self.compare(reject=reject)

    def test_eos_accepted_rejected_and_first(self):
        for kw in [dict(proposal_eos=True),dict(proposal_eos=True,reject=0),dict(first_eos=True)]:
            with self.subTest(kw=kw):
                self.compare(**kw)

    def test_device_failure_reclaims_all_kv(self):
        b=DeviceOracle(fail='verify');e=engine(b,gpu_draft_tokens=True)
        with self.assertRaises(FloatingPointError):
            e.generate([request()])
        self.assertTrue(e.poisoned)
        self.assertFalse(e.target_pool.used or e.draft_pool.used)


if __name__=='__main__':
    unittest.main(verbosity=2)
