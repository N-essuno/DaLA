import copy
import random
import unittest
import spacy
from dala.profiles import load_profile
from dala.rule_engine import run_rule


class ProfileTests(unittest.TestCase):
    def test_mapping_is_input_not_language_code(self):
        nlp=spacy.blank('xx')
        rule={'operator':'lookup','branches':[{'when':{},'mapping':{'alpha':'beta'}}]}
        self.assertEqual(run_rule(rule,nlp,'alpha gamma.'),(True,'beta gamma.'))
        changed=copy.deepcopy(rule);changed['branches'][0]['mapping']['alpha']='delta'
        self.assertEqual(run_rule(changed,nlp,'alpha gamma.'),(True,'delta gamma.'))

    def test_suffix_operator_works_for_new_vocabulary(self):
        nlp=spacy.blank('xx')
        rule={'operator':'suffix','gate':{},'case':'preserve_stem',
              'branches':[{'when':{'suffix':'xyz'},'remove':3,'append':'abc'}]}
        self.assertEqual(run_rule(rule,nlp,'NEWXYZ.'),(True,'NEWABC.'))

    def test_unknown_predicate_fails(self):
        nlp=spacy.blank('xx')
        rule={'operator':'suffix','gate':{'typo':[]},'case':'preserve_stem','branches':[]}
        with self.assertRaises(ValueError):run_rule(rule,nlp,'word')

    def test_danish_registry_order(self):
        rules=load_profile('da')['rules']
        self.assertEqual(len(rules),15)
        self.assertEqual(rules[0]['operator'],'genitive')
        self.assertEqual(rules[-1]['operator'],'token_fallback')

    def test_custom_pack_is_not_limited_by_language_registry(self):
        import json
        import tempfile
        from pathlib import Path
        profile=load_profile('da');profile.pop('_path');profile['language']='zz'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'custom.json';path.write_text(json.dumps(profile))
            self.assertEqual(load_profile(path)['language'],'zz')

    def test_task_prompts_and_language_are_supplied(self):
        from dala.pair_pipeline import task_rows
        pair=dict(pair_id='p',document_id='d',language='zz',original='a b',corrupted='a c',
                  edits=[dict(rule_id='r',corruption_type='spelling',original='b',replacement='c')])
        rows=list(task_rows(pair,dict(acceptability='CUSTOM ACCEPT',correction='CUSTOM CORRECT')))
        self.assertEqual(rows[0][1]['direction'],'CUSTOM ACCEPT')
        self.assertEqual(rows[1][2]['direction'],'CUSTOM CORRECT')
        self.assertEqual(rows[0][1]['language'],'zz')
