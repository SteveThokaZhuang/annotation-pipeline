import unittest
from common import match_words,structural
from router import choose,features,consistent_with_primary

class AlignmentTests(unittest.TestCase):
    def test_repeat_and_missing(self):
        m=match_words([{'text':t} for t in ['we','we','go']],[{'text':t} for t in ['we','go']])
        self.assertEqual(len(m['pairs']),2)
        self.assertEqual(len(m['deleted']),1)
    def test_substitution_not_timing_match(self):
        m=match_words([{'text':'cat'}],[{'text':'dog'}])
        self.assertEqual(m['pairs'],[])
        self.assertEqual(m['substituted'],[(0,0)])
    def test_zero_is_not_missing(self):
        self.assertIn((0,'zero_duration'),structural([dict(text='a',start=0,end=0)],1))
    def test_nonfinite(self):
        self.assertIn((0,'nonfinite'),structural([dict(text='a',start=float('nan'),end=0)],1))
    def test_reject_invalid_fallback(self):
        q=dict(words=[dict(text='a',start=0,end=0)])
        n=dict(words=[dict(text='a',start=-1,end=.1)])
        policy={'rules':[dict(feature='hard',threshold=1,backend='nyra')]}
        out=choose('a',1,dict(qwen=q,nyra=n),policy)
        self.assertEqual(out['selected_backend'],'qwen');self.assertTrue(out['needs_review'])
    def test_accept_complete_fallback(self):
        q=dict(words=[dict(text='a',start=0,end=0)])
        n=dict(words=[dict(text='a',start=.1,end=.3)])
        policy={'rules':[dict(feature='hard',threshold=1,backend='nyra')]}
        out=choose('a',1,dict(qwen=q,nyra=n),policy)
        self.assertEqual(out['selected_backend'],'nyra');self.assertFalse(out['needs_review'])
    def test_guard_rejects_new_long_word(self):
        self.assertFalse(consistent_with_primary([dict(text='yes',start=0,end=.3)],[dict(text='yes',start=0,end=2)]))
    def test_guard_rejects_jump_at_stable_anchor(self):
        self.assertFalse(consistent_with_primary([dict(text='yes',start=0,end=.3)],[dict(text='yes',start=1,end=1.3)]))
    def test_guard_accepts_small_correction(self):
        self.assertTrue(consistent_with_primary([dict(text='yes',start=0,end=.3)],[dict(text='yes',start=.05,end=.35)]))
if __name__=='__main__':unittest.main()
