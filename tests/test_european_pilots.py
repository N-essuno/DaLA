import json
from pathlib import Path
import tempfile
import unittest

from dala.conllu_source import records, eligible, documents, AnnotatedParser
from dala.rule_compiler import compile_inflections
from dala.language_packs.morphology import MorphologyPack


class EuropeanPilotTests(unittest.TestCase):
    def test_language_inventory_is_separate(self):
        codes={'de','fr','es','it','cs','pt-PT','fi','et','ca','el','ro','uk'}
        specs=[json.loads(Path(f'config/european/{c}.json').read_text()) for c in codes]
        self.assertEqual({s['language'] for s in specs},codes)
        for s in specs:
            self.assertTrue(s['grammar']);self.assertTrue(s['spelling'])
            self.assertIn(s['name'],s['prompts']['correction'])
        pt=next(s for s in specs if s['language']=='pt-PT')
        self.assertEqual(pt['sentence_id_pattern'],'^CP')
        self.assertEqual(pt['dictionary'],'pt-PT')

    def test_compiler_preserves_possessor_and_other_features(self):
        morphology=[dict(lemma='poss',pos='DET',forms=[word],normative=True,features={'Number':number,'Person[psor]':person})
                    for word,number,person in [('aa','Sing','1'),('bb','Plur','3'),('cc','Plur','1')]]
        definition=dict(family='number',context='nominal_agreement',feature='Number',pos=['DET'])
        rule=compile_inflections('xx',morphology,[definition],{'preserve_all_features':True,'ignore_gender_animacy_features':[]})[0]
        self.assertEqual(rule['mappings']['aa'],['cc'])

    def test_syncretism_veto(self):
        morphology=[dict(lemma='v',pos='VERB',forms=forms,normative=True,features={'Number':number,'VerbForm':'Fin'})
                    for number,forms in [('Sing',['aa','bb']),('Plur',['bb'])]]
        definition=dict(family='number',context='finite_agreement',feature='Number',pos=['VERB'])
        rule=compile_inflections('xx',morphology,[definition])[0]
        self.assertNotIn('aa',rule['mappings'])

    def test_rulebook_and_embedded_rules_cannot_cross_languages(self):
        pack=MorphologyPack.__new__(MorphologyPack);pack.language='fr';pack.resource_receipt={'language':'fr'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rules.json'
            book=dict(language='de',sources={'morphology':pack.resource_receipt},rules=[])
            path.write_text(json.dumps(book))
            with self.assertRaisesRegex(ValueError,'standard mismatch'):pack.load_rulebook(path)
            book.update(language='fr',rules=[dict(id='de_agreement',language='de',sources=['morphology'])])
            path.write_text(json.dumps(book))
            with self.assertRaisesRegex(ValueError,'Rule language mismatch'):pack.load_rulebook(path)

    def test_resource_receipt_cannot_cross_languages(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'receipt.json').write_text(json.dumps({'language':'de'}))
            (root/'exclude.json').write_text('[]')
            profile=dict(language='fr',_path=str(root/'profile.json'),resource_directory=str(root),exclusions=str(root/'exclude.json'))
            with self.assertRaisesRegex(ValueError,'resource language mismatch'):
                MorphologyPack(profile)

    def test_mixed_portuguese_and_known_errors_are_rejected(self):
        record=dict(sentence_id='CF1-1',rows=[['1','a','a','DET','_','_','2','det','_','_']])
        self.assertFalse(eligible(record,{'sentence_id_pattern':'^CP'}))
        record['sentence_id']='CP1-1';self.assertTrue(eligible(record,{'sentence_id_pattern':'^CP'}))
        record['rows'][0][5]='Typo=Yes';self.assertFalse(eligible(record,{}))
        record['rows'][0][5]='_';record['rows'][0][0]='1-2';self.assertFalse(eligible(record,{}))

    def test_unknown_document_boundaries_stay_together(self):
        text='# sent_id = x1\n# text = Alpha.\n1\tAlpha\talpha\tNOUN\t_\t_\t0\troot\t_\t_\n\n# sent_id = x2\n# text = Beta.\n1\tBeta\tbeta\tNOUN\t_\t_\t0\troot\t_\t_\n'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.conllu';path.write_text(text)
            source=dict(name='fixture',language='xx',url='fixture',license='fixture',revision='fixture')
            docs=list(documents([(source,{'data':path},{})]))
            self.assertEqual(len(docs),1)
            self.assertEqual(docs[0]['document_boundary_status'],'unavailable_file_grouped')

    def test_annotated_offsets_are_exact(self):
        import spacy
        parser=AnnotatedParser.__new__(AnnotatedParser);parser.vocab=spacy.blank('xx').vocab;parser.rejections={}
        text='They work.'
        rows=[['1','They','they','PRON','_','Number=Plur','2','nsubj','_','_'],['2','work','work','VERB','_','VerbForm=Fin','0','root','_','_'],['3','.','.','PUNCT','_','_','2','punct','_','_']]
        parser.by_text={text:[dict(rows=rows)]}
        doc=parser(text).doc
        self.assertEqual(doc.text,text);self.assertEqual(doc[1].idx,5);self.assertEqual(doc[0].head,doc[1])
        self.assertIsNone(parser('They  work.'))


if __name__=='__main__':unittest.main()
