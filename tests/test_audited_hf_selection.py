import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_audited_european_hf import eligible,FIELDS

class SelectionTests(unittest.TestCase):
    def test_only_four_yes_passes(self):
        good=dict.fromkeys(FIELDS,'yes');good['decision']='pass'
        self.assertTrue(eligible('done',good))
        for status in ('failed','running','pending'):
            self.assertFalse(eligible(status,good))
        self.assertFalse(eligible('done',good,excluded=True))
        for field in FIELDS:
            self.assertFalse(eligible('done',dict(good,**{field:'uncertain'})))
            self.assertFalse(eligible('done',dict(good,**{field:'no'})))
        self.assertFalse(eligible('done',dict(good,decision='flag')))
        self.assertFalse(eligible('done',None))
if __name__=='__main__':unittest.main()
