import unittest
from benchmark_single_h100 import accuracy
from evaluate import word_rows

class AccuracyMetricTests(unittest.TestCase):
    def row(self,pred):
        r=dict(id='x',base_id='x',meeting='test',condition='headset',duration=1,
               words=[dict(id='w',text='yes',start=.1,end=.2)])
        return word_rows(r,dict(words=pred))
    def test_missing_fails_denominator(self):
        m=accuracy(self.row([]))
        self.assertEqual(m['reference_words'],1)
        self.assertEqual(m['valid_both_boundaries_acc_100ms'],0)
    def test_zero_not_accurate_even_if_small_mae(self):
        m=accuracy(self.row([dict(text='yes',start=.15,end=.15)]))
        self.assertEqual(m['valid_both_boundaries_acc_100ms'],0)
        self.assertEqual(m['legacy_mae_le100ms_accuracy'],1)
    def test_both_boundaries_stricter_than_mean(self):
        m=accuracy(self.row([dict(text='yes',start=.1,end=.35)]))
        self.assertEqual(m['valid_both_boundaries_acc_100ms'],0)
        self.assertEqual(m['legacy_mae_le100ms_accuracy'],1)
    def test_exact_threshold_and_valid_word(self):
        m=accuracy(self.row([dict(text='yes',start=.2,end=.3)]))
        self.assertEqual(m['valid_both_boundaries_acc_100ms'],1)
        self.assertEqual(m['valid_both_boundaries_acc_50ms'],0)
if __name__=='__main__':unittest.main()
