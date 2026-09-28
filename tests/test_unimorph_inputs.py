import json
from pathlib import Path
import unittest

from scripts.prepare_unimorph_inputs import convert
from dala.rule_compiler import compile_inflections


def recipe(language):
    return json.loads(Path(f'config/european-expansion/morphology/{language}.json').read_text())


class LexicalExpansionTests(unittest.TestCase):
    def test_personal_infinitive_remains_nonfinite(self):
        row=convert('falar','falares','V;NFIN;2;SG',recipe('pt-PT'))
        self.assertEqual(row['features']['VerbForm'],'Inf')
        self.assertEqual(row['features']['Person'],'2')

    def test_locative_and_essive_conventions_are_explicit(self):
        self.assertEqual(convert('x','x','N;ESS;SG',recipe('cs'))['features']['Case'],'Loc')
        self.assertEqual(convert('x','x','N;FRML;SG',recipe('fi'))['features']['Case'],'Ess')
        self.assertFalse(convert('x','x','N;ESS;SG',recipe('fi'))['generation_eligible'])

    def test_unknown_clitic_and_register_tags_are_veto_only(self):
        for tag in ['PRO','FORM','LGSPEC1']:
            row=convert('x','x','V;PRS;2;SG;'+tag,recipe('es'))
            self.assertFalse(row['generation_eligible'])
            self.assertIn(tag,row['unmapped_tags'])

    def test_possessors_preserved(self):
        row=convert('x','x','N;GEN;SG;PSS1P',recipe('fi'))
        self.assertEqual(row['features']['Person[psor]'],'1')
        self.assertEqual(row['features']['Number[psor]'],'Plur')

    def test_ambiguous_analysis_vetoes_replacement(self):
        rows=[dict(lemma='x',pos='NOUN',forms=[form],features={'Case':case},normative=True,generation_eligible=eligible)
              for form,case,eligible in [('aa','Nom',True),('bb','Acc',True),('bb','Acc,Nom',False)]]
        rule=compile_inflections('xx',rows,[dict(family='case',context='nominal_agreement',feature='Case',pos=['NOUN'])])[0]
        self.assertNotIn('aa',rule['mappings'])

    def test_unmapped_analysis_cannot_generate_but_still_vetoes(self):
        rows=[dict(lemma='x',pos='NOUN',forms=[form],features=feats,normative=True,generation_eligible=eligible)
              for form,feats,eligible in [('aa',{'Case':'Nom'},True),('bb',{'Case':'Acc'},True),('bb',{},False)]]
        rule=compile_inflections('xx',rows,[dict(family='case',context='nominal_agreement',feature='Case',pos=['NOUN'])])[0]
        self.assertEqual(rule['mappings'],{'bb':['aa']})

    def test_lexical_opt_in_does_not_promote_single_ud_attestations(self):
        rows=[dict(lemma='x',pos='NOUN',forms=[form],features={'Number':number},**extra)
              for form,number,extra in [('aa','Sing',{'attestations':{'aa':1}}),('bb','Plur',{'evidence_kind':'unimorph_descriptive_paradigm'})]]
        rule=compile_inflections('xx',rows,[dict(family='number',context='nominal_agreement',feature='Number',pos=['NOUN'])],{'allow_lexical_evidence':True})[0]
        self.assertFalse(rule['active'])


if __name__=='__main__':unittest.main()
