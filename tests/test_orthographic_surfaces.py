import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from collections import defaultdict
from dala.character_operations import replacements
from dala.language_packs.morphology import MorphologyPack, permits
from dala.morphology_check import MorphologyCheck
from scripts.prepare_unimorph_inputs import convert
from scripts.prepare_european_pilots import attested_morphology

class OrthographyTests(unittest.TestCase):
    def test_spelling_validator_does_not_conflate_sharp_s(self):
        rule=dict(operator='character_edit',operation='delete_internal',min_length=4,max_length=24,unicode_letters=True)
        self.assertFalse(permits(rule,SimpleNamespace(original='Floß',replacement='Flss')))
        self.assertTrue(permits(rule,SimpleNamespace(original='Floß',replacement='Foß')))
        self.assertNotIn('Flss'.lower(),replacements('Floß',rule))

    def test_greek_validator_preserves_final_sigma(self):
        rule=dict(operator='dictionary_inflection',mappings={'της':['τις']})
        self.assertTrue(permits(rule,SimpleNamespace(original='Της',replacement='Τις')))
        self.assertFalse(permits(rule,SimpleNamespace(original='Της',replacement='Τισ')))

    def test_unimorph_surface_and_lemma_keys_are_separate(self):
        recipe={'tags':{'N':{'pos':'NOUN'}}}
        entry=convert('κόσμος','κόσμος','N',recipe)
        self.assertEqual(entry['forms'],['κόσμος'])
        self.assertEqual(entry['lemma'],'κόσμοσ')
        self.assertEqual(convert('Maß','Maße','N',recipe)['forms'],['maße'])

    def test_training_attestations_do_not_merge_distinct_spellings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.conllu'
            p.write_text(''.join(f'# sent_id = x{i}\n# text = {form}\n1\t{form}\tmaß\tNOUN\t_\tNumber=Sing\t0\troot\t_\t_\n\n' for i,form in enumerate(['Maß','Mass','Maß'])))
            entries,_=attested_morphology(p,{})
            self.assertEqual(entries[0]['attestations'],{'mass':1,'maß':2})

    def test_actual_generator_and_validator_agree_for_unicode(self):
        pack=MorphologyPack.__new__(MorphologyPack)
        pack.profile={};pack.by_word=defaultdict(list)
        for word in ['Straße','κόσμος']:
            pack.recognized=lambda w:w==word.lower()
            token=SimpleNamespace(text=word,is_alpha=True,pos_='NOUN',ent_type_='',lemma_=word.lower(),idx=0,i=0)
            class Sentence(list):pass
            sent=Sentence([token]);sent.text=word;sent.start_char=0
            pack.edit=lambda s,t,replacement,rule:SimpleNamespace(original=t.text,replacement=replacement)
            rule=dict(id='xx_delete',operator='character_edit',operation='delete_internal',min_length=4,max_length=24,unicode_letters=True)
            result=pack.candidates(sent,{'rules':[rule]})
            self.assertTrue(result)
            self.assertTrue(all(permits(rule,e) for e in result))
            self.assertTrue(all(len(e.replacement)==len(word)-1 for e in result))

    def test_reduced_export_retains_orthographic_rule_endpoints(self):
        import json
        from dala.pair_pipeline import export_dataset, validate_pairs
        rule=dict(id='el_case',family='case',operator='dictionary_inflection',mappings={'ένα':['ενός']})
        book=dict(surface_normalization='unicode_lower',edit_validator='dala.language_packs.morphology:permits',rules=[rule])
        pair=dict(pair_id='p',document_id='d',split='train',language='el',source_name='fixture',url='fixture',
                  original='ένα',corrupted='ενός',edits=[dict(rule_id='el_case',corruption_type='case',original='ένα',replacement='ενός',start=0,end=3,corrupted_start=0,corrupted_end=4,protected_tokens=[])])
        manifest=dict(profile={'export_used_rule_mappings':True},quality_status='test',prompts={'acceptability':'Correct?','correction':'Correct it.'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dataset'
            export_dataset([pair],root,manifest,book,[{'document_id':'d'}])
            exported=json.loads((root/'rules.json').read_text())
            self.assertEqual(exported['rules'][0]['mappings'],{'ένα':['ενός']})
            self.assertEqual(validate_pairs([pair],exported)['pairs'],1)

    def test_checker_preserves_surfaces_and_quarantines_only_exact_text(self):
        checker=MorphologyCheck.__new__(MorphologyCheck)
        checker.adapter=SimpleNamespace(language='el',recognized=Mock(side_effect=lambda w:w=='κόσμος'))
        checker.source_checker=None
        evidence,reason=checker.screen('source','bad',[dict(original='Κόσμος',replacement='κόσμς',corruption_type='spelling',rule_id='r',corrupted_start=0,corrupted_end=5)])
        self.assertIsNone(reason);self.assertIsNotNone(evidence)
        checker.source_failure_exclusions={hashlib.sha256('bad source'.encode()).hexdigest():'reproduced receipt'}
        checker.source_checker=Mock()
        self.assertEqual(checker.screen('bad source','bad',[]),(None,'documented_source_checker_failure'))
        checker.source_checker.check.assert_not_called()
        checker.source_checker.check.side_effect=RuntimeError('new server failure')
        with self.assertRaisesRegex(RuntimeError,'new server failure'):checker.screen('other source','bad',[])

if __name__=='__main__':unittest.main()
