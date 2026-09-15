"""History proposal semantics and target-only physical KV state; no CUDA proof."""
import unittest
from test_random_decode import PrefixOracle, engine, request
from nanovllm.engine.ngram import history_proposal


class NOracle(PrefixOracle):
    def point_mass_proposals(self, requests, logits, proposals):
        self.histories = getattr(self, 'histories', []) + [{r.request_id:list(r.tokens) for r in requests}]
        self.proposals = getattr(self, 'proposals', []) + [dict((k,list(v)) for k,v in proposals.items())]
        return {r.request_id: [object() for _ in proposals[r.request_id]] for r in requests}


class NTests(unittest.TestCase):
    def test_longest_before_earliest_and_overlap(self):
        self.assertEqual(history_proposal([7,8,3,7,8],3),[3,7,8])
        self.assertEqual(history_proposal([7,8,3,7,8,4,7,8],3),[3,7,8])
        self.assertEqual(history_proposal([1,2,9,2,8,1,2],3),[9,2,8])
        self.assertEqual(history_proposal([1,1,1],4),[1])

    def test_budget_and_no_match(self):
        self.assertEqual(history_proposal([1,2,3],4),[])
        self.assertEqual(history_proposal([7,8,3,7,8],1),[3])
        self.assertEqual(history_proposal([7,8,3,7,8],0),[])
        with self.assertRaises(ValueError):history_proposal([1],-1)

    def run_case(self, cap=20, reject=None, fail=None):
        b=NOracle(reject=reject,fail=fail)
        e=engine(b,ngram=True,draft_blocks=0)
        r=request(prompt=[10,20,30,10,20],cap=cap)
        if fail:
            with self.assertRaises(FloatingPointError):e.generate([r])
            self.assertTrue(e.poisoned)
        else:
            out=e.generate([r]);self.assertEqual(len(out[0]['token_ids']),cap)
        self.assertIsNone(e.draft_pool)
        self.assertFalse(e.target_pool.used or r.target.block_table or r.draft.block_table)
        self.assertFalse(any(x[0]=='draft' for x in b.events))
        self.assertEqual(b.draw_steps,0)
        return b

    def test_rejection_positions_cross_blocks_and_bonus(self):
        for reject in [0,1,2,3,None]:
            with self.subTest(reject=reject):
                b=self.run_case(reject=reject)
                self.assertTrue(any(p['a'] for p in b.proposals))

    def test_cap_one_and_two(self):
        self.assertFalse(hasattr(self.run_case(cap=1),'proposals'))
        self.assertEqual(self.run_case(cap=2).proposals,[{'a':[]}])

    def test_verify_failure_releases_target(self):
        self.run_case(fail='verify')

    def test_rejected_candidates_not_added_to_history(self):
        b=self.run_case(reject=0)
        self.assertEqual(b.histories[0]['a'],[10,20,30,10,20,10])
        self.assertEqual(b.proposals[0]['a'],[20,30,10])
        self.assertEqual(b.histories[1]['a'],[10,20,30,10,20,10,30])

    def test_proposal_eos_accept_reject_and_ignore(self):
        for reject, ignore in [(None,False),(0,False),(None,True)]:
            b=NOracle(reject=reject);e=engine(b,ngram=True,draft_blocks=0)
            r=request(prompt=[10,99,20],cap=8);r.ignore_eos=ignore
            out=e.generate([r])[0]
            if reject is None and not ignore:
                self.assertEqual(out['token_ids'],[10,99])
                self.assertEqual(out['stop_reason'],'eos')
            else:
                self.assertEqual(len(out['token_ids']),8)
            self.assertEqual(b.proposals[0]['a'],[99,20,10] if ignore else [99])
            self.assertFalse(e.target_pool.used)

    def test_no_match_keeps_request(self):
        b=NOracle();e=engine(b,ngram=True,draft_blocks=0)
        r=request(prompt=[1,2,3],cap=2)
        self.assertEqual(e.generate([r])[0]['token_ids'],[10,30])
        self.assertEqual(b.proposals,[{'a':[]}])

    def test_reject_incompatible_resources(self):
        with self.assertRaises(ValueError):engine(NOracle(),ngram=True)
        with self.assertRaises(ValueError):engine(NOracle(),ngram=True,draft_blocks=0,gpu_draft_tokens=True)


if __name__=='__main__':unittest.main()
