from collections import Counter
import itertools
import unittest
from scripts.balance_pairs import balanced_subset


class BalanceTests(unittest.TestCase):
    config = dict(groups=[dict(name='articles', families=['article_gender','article_number'],max_fraction=.25),
                          dict(name='spelling',families=['spelling'],max_fraction=.6)])

    def pairs(self, articles, spelling, other, split='train'):
        return [dict(pair_id=f'{split}-{family}-{i}',split=split,edits=[dict(corruption_type=family)])
                for family,n in [('article_gender',articles),('spelling',spelling),('subject_verb_agreement',other)]
                for i in range(n)]

    def test_caps_optimality_and_retention_of_scarce_families(self):
        for a,s,o in [(44,49,7),(5,4,12),(8,3,1),(0,0,5)]:
            pairs=self.pairs(a,s,o)
            selected,_=balanced_subset(pairs,self.config)
            n=len(selected); c=Counter(p['edits'][0]['corruption_type'] for p in selected)
            self.assertLessEqual(c['article_gender']*4,n)
            self.assertLessEqual(c['spelling']*5,n*3)
            self.assertEqual(c['subject_verb_agreement'],o)
            # Exhaustive independent oracle for the largest possible subset.
            possible=[x+y+z for x,y,z in itertools.product(range(a+1),range(s+1),range(o+1))
                      if x*4<=x+y+z and y*5<=3*(x+y+z)]
            self.assertEqual(n,max(possible))

    def test_order_independence_and_split_isolation(self):
        pairs=self.pairs(40,50,10)+self.pairs(20,20,5,'test')
        a,_=balanced_subset(pairs,self.config)
        b,_=balanced_subset(list(reversed(pairs)),self.config)
        self.assertEqual(a,b)
        self.assertTrue(set(p['pair_id'] for p in a)<=set(p['pair_id'] for p in pairs))

    def test_impossible_caps_and_multiple_edits_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'No nonempty'):
            balanced_subset(self.pairs(10,10,0),self.config)
        pairs=self.pairs(0,0,2);pairs[0]['edits']*=2
        with self.assertRaisesRegex(ValueError,'single-edit'):
            balanced_subset(pairs,self.config)

    def test_prioritized_article_number_survives_gender_downsampling(self):
        import copy
        config=copy.deepcopy(self.config)
        config['groups'][0]['priority_families']=['article_number']
        pairs=self.pairs(40,50,10)
        for p in pairs[:5]:p['edits'][0]['corruption_type']='article_number'
        selected,_=balanced_subset(pairs,config)
        self.assertEqual(sum(p['edits'][0]['corruption_type']=='article_number' for p in selected),5)
