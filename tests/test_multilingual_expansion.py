import json
import gzip
from pathlib import Path
import unittest
from unittest.mock import patch

from dala.character_operations import replacements
from dala.profiles import load_profile, resource
from dala.pair_pipeline import task_rows, dataset_card
from dala.dynaword import snapshots
from scripts.prepare_inflection_lexicons import polish_features, icelandic_features


class ExpansionContractTests(unittest.TestCase):
    def test_six_distinct_standards_and_prompts(self):
        profiles = [load_profile(l) for l in ['pl','sv','nb','nn','fo','is']]
        self.assertEqual(len({p['prompts']['acceptability'] for p in profiles}), 6)
        self.assertEqual(len({p['prompts']['correction'] for p in profiles}), 6)
        coverage=json.loads(Path('wiki/artifacts/six-language-expansion/configured-coverage.json').read_text())
        for p in profiles:
            active = [r for r in coverage[p['language']] if r['active']]
            self.assertTrue(any(r['family'] == 'spelling' for r in active))
            self.assertTrue(any(r['family'] != 'spelling' for r in active))
            self.assertEqual(p['release_policy']['status'],'candidate_only')

    def test_unicode_is_opt_in_and_valid_letters_preserved(self):
        r=dict(min_length=5,max_length=24,operation='transpose_internal')
        self.assertEqual(replacements('føroyskt',r),())
        candidates=replacements('føroyskt',dict(r,unicode_letters=True))
        self.assertTrue(candidates)
        self.assertTrue(all(sorted(w)==sorted('føroyskt') for w in candidates))

    def test_diacritic_rules_do_not_touch_unconfigured_letters(self):
        r=dict(min_length=5,max_length=24,operation='substitute_character',unicode_letters=True,substitutions={'ó':['o']})
        self.assertEqual(replacements('szkoła',r),())
        self.assertEqual(replacements('królowa',r),('krolowa',))

    def test_current_targets_keep_faroese_uncapped_and_cap_other_languages(self):
        targets=json.loads(Path('config/multilingual_targets.json').read_text())['languages']
        self.assertIsNone(targets['fo']['target_pairs'])
        for code in ['is','nn']:self.assertEqual(targets[code]['target_pairs'],478930)
        for code in ['pl','sv','nb']:self.assertEqual(targets[code]['target_pairs'],478930)

    def test_candidate_card_does_not_claim_independent_grammar_checks(self):
        text=dataset_card(dict(verification={'pairs':1},quality_status='candidate_rule_and_dictionary_screened'))
        self.assertIn('no independent',text)
        self.assertNotIn('diagnostic at every injected edit',text)

    def test_clean_control_preserves_written_standard_and_text(self):
        text='Vi har skrive eit langt brev til den nye skulen.'
        pair=dict(pair_id='p',document_id='d',language='nn',original=text,corrupted='dummy',edits=[])
        first=next(task_rows(pair,load_profile('nn')['prompts']))
        self.assertEqual(first[2]['samples']['response'],text)
        self.assertIn('nynorsk',first[2]['direction'])
        self.assertEqual(first[2]['language'],'nn')

    def test_native_polish_syncretism_is_not_lost(self):
        analyses=polish_features('adj:sg:nom.voc:m1.m2.m3:pos')
        self.assertEqual(len(analyses),6)
        self.assertEqual({f['Case'] for _,f in analyses},{'Nom','Voc'})
        self.assertEqual({f['Animacy'] for _,f in analyses},{'Hum','Anim','Inan'})

    def test_ud_alternative_genders_match_both_lexical_analyses(self):
        from dala.language_packs.morphology import features_compatible
        for gender in ['Fem','Masc']:
            self.assertTrue(features_compatible({'Gender':'Fem,Masc','Number':'Sing'},{'Gender':gender,'Number':'Sing'}))
        self.assertFalse(features_compatible({'Gender':'Fem,Masc'},{'Gender':'Neut'}))
        self.assertFalse(features_compatible({'Gender':'Fem,Masc','Number':'Sing'},{'Gender':'Masc','Number':'Plur'}))

    def test_icelandic_case_and_quirky_subjects(self):
        self.assertEqual(icelandic_features('kk','ÞGFETgr'),('NOUN',{'Gender':'Masc','Case':'Dat','Number':'Sing','Definite':'Def'}))
        self.assertIsNone(icelandic_features('so','OP-ÞGF-GM-VH-ÞT-1P-ET'))

    def test_grammar_selection_does_not_starve_later_families(self):
        from dala.language_packs.morphology import MorphologyPack
        from dala.edits import Corruption
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.profile={'selection':{'strategy':'hash_rotate_grammar_then_spelling','priority':['first','second','spelling']}}
        pack.rule_priority={}
        edits=[Corruption(f,f,0,1,'a','b',(0,)) for f in ['first','second','spelling']]
        selected={pack.select_edits(str(i),edits,42,1)[0].corruption_type for i in range(40)}
        self.assertEqual(selected,{'first','second'})

    def test_long_paragraphs_keep_exact_offsets(self):
        from dala.language_packs.morphology import MorphologyPack
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.profile={'curation':{'max_paragraph_chars':90}}
        sentence='This is a sufficiently long complete sentence.'
        text='  '+('  '.join([sentence]*8))+'\n'
        chunks=list(pack.paragraphs({'text':text}))
        self.assertEqual(len(chunks),8)
        for chunk,start in chunks:self.assertEqual(text[start:start+len(chunk)],chunk)

    def test_parallel_document_fragments_preserve_all_paragraphs_and_offsets(self):
        from dala.language_packs.morphology import MorphologyPack
        from dala.batch_pipeline import task_documents
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.profile={'curation':{'max_paragraph_chars':90}}
        sentence='This is a sufficiently long complete sentence.'
        document={'document_id':'stable','text':'  '+'  '.join([sentence]*19)+'\n'}
        expected=list(pack.paragraphs(document));actual=[]
        fragments=list(task_documents([document],pack,200))
        self.assertGreater(len(fragments),1)
        for d in fragments:
            self.assertEqual(d['text'],document['text'])
            start=d['_fragment_start'];end=d['_fragment_end']
            for text,offset in pack.paragraphs(dict(d,text=d['text'][start:end])):
                actual.append((text,offset+start))
        self.assertEqual(actual,expected)

    def test_syncretic_noun_cannot_license_number_error(self):
        from spacy.tokens import Doc
        from spacy.vocab import Vocab
        from dala.language_packs.morphology import MorphologyPack
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.language='sv'
        pack.profile={}
        doc=Doc(Vocab(),words=['stort','hus'],heads=[1,1],deps=['amod','ROOT'],
                pos=['ADJ','NOUN'],lemmas=['stor','hus'],
                morphs=['Number=Sing','Number=Sing'])
        rule=dict(feature='Number',pos=['ADJ'],context='nominal_agreement',relations=['amod'])
        pair=dict(lemma='stor',before={'Number':'Sing'},after={'Number':'Plur'})
        pack.analyses={('hus','NOUN'):[{'features':{'Number':n}} for n in ['Sing','Plur']]}
        self.assertFalse(pack.licensed_context(doc[0],rule,pair))
        pack.analyses={('hus','NOUN'):[{'features':{'Number':'Sing'}}]}
        self.assertTrue(pack.licensed_context(doc[0],rule,pair))
        pack.analyses[('hus','NOUN')].append({'features':{}})
        self.assertFalse(pack.licensed_context(doc[0],rule,pair))

    def test_source_agreement_error_is_rejected_before_other_edits(self):
        from spacy.tokens import Doc
        from spacy.vocab import Vocab
        from dala.language_packs.morphology import MorphologyPack
        pack=MorphologyPack.__new__(MorphologyPack);pack.language='nn';pack.profile={}
        pack.analyses={('kommunal','ADJ'):[{'lemma':'kommunal','features':{'Definite':'Ind','Number':'Sing','Gender':'Masc'}}],
                       ('bustadpolitikken','NOUN'):[{'lemma':'bustadpolitikk','features':{'Definite':'Def','Number':'Sing','Gender':'Masc'}}]}
        doc=Doc(Vocab(),words=['den','kommunal','bustadpolitikken'],heads=[2,2,2],deps=['det','amod','ROOT'],
                pos=['DET','ADJ','NOUN'],lemmas=['den','kommunal','bustadpolitikk'],
                morphs=['Definite=Def','Definite=Ind|Gender=Masc|Number=Sing','Definite=Def|Gender=Masc|Number=Sing'])
        self.assertEqual(pack.sentence_rejection(doc[:],set()),'source_nominal_agreement_conflict')


if __name__=='__main__':unittest.main()
