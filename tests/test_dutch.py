import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dala.dynaword import documents
from dala.language_check import LanguageCheck
from dala.language_packs.dutch import DutchPack
from dala.profiles import load_profile


class DutchCheckTests(unittest.TestCase):
    def test_opt_in_long_paragraphs_preserve_source_offsets(self):
        profile = load_profile('nl_scale')
        text = '  ' + 'De rechtbank heeft de zaak zorgvuldig beoordeeld. ' * 150 + '\n'
        document = dict(document_id='long-source', text=text)
        self.assertEqual(list(DutchPack(profile).paragraphs(document)), [])
        profile['curation']['max_paragraph_chars'] = 60000
        blocks = list(DutchPack(profile).paragraphs(document))
        self.assertEqual(len(blocks), 1)
        block, offset = blocks[0]
        self.assertEqual(offset, 2)
        self.assertEqual(text[offset:offset + len(block)], block)

    def test_uncategorized_requires_explicit_rule_id(self):
        data={'matches':[dict(offset=0,length=8,message='Agreement',rule=dict(id='JIJ_LOOP',issueType='uncategorized'))]}
        self.assertEqual(LanguageCheck.hard_matches(data,'Hij word'),[])
        self.assertEqual(LanguageCheck.hard_matches(data,'Hij word',['JIJ_LOOP'])[0]['issue_type'],'grammar')
        self.assertEqual(LanguageCheck.hard_matches(data,'Hij word',['SOME_OTHER_RULE']),[])

    def test_reviewed_grammar_rule_can_override_spelling_category(self):
        data={'matches':[dict(offset=0,length=8,message='Agreement',rule=dict(id='NL_SIMPLE_REPLACE_IK_VINDT',issueType='misspelling'))]}
        self.assertEqual(LanguageCheck.hard_matches(data,'Ik vindt')[0]['issue_type'],'misspelling')
        self.assertEqual(LanguageCheck.hard_matches(data,'Ik vindt',['NL_SIMPLE_REPLACE_IK_VINDT'])[0]['issue_type'],'grammar')

    def test_dynaword_annotation_join_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);text='De overheid heeft de nieuwe regels gisteren bekendgemaakt.'
            pq.write_table(pa.Table.from_pylist([dict(id='a',source='gov',text=text),dict(id='b',source='gov',text='Bad source'),dict(id='c',source='gov',text=text)]),p/'data.parquet')
            pq.write_table(pa.Table.from_pylist([dict(id='a',quality='good'),dict(id='b',quality='bad'),dict(id='c',quality='good')]),p/'meta.parquet')
            source=dict(name='gov',source_value='gov',repo_id='org/repo',revision='a'*40,data_file='data/gov/data.parquet',license='CC0',license_evidence_url='https://example.org/license',annotation_filters=dict(quality=['good']))
            rows=list(documents([(source,dict(data=p/'data.parquet',metadata=p/'meta.parquet'),{})]))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['text'],text)
            self.assertEqual(rows[0]['upstream_id'],'a')
            self.assertEqual(rows[0]['url_kind'],'upstream_dataset_file_not_original_article')
            self.assertIsNone(rows[0]['author'])
            self.assertEqual(rows[0]['source_quality_annotation']['quality'],'good')

    def test_curated_source_date_and_annotation_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            source=json.loads(Path('config/dutch_sources_curated.json').read_text())['sources'][1]
            rows=[];annotations=[]
            for id,year,kind,description in [
                ('keep',2008,['legal'],'A regulation about product safety'),
                ('old',1998,['legal'],'A regulation about product safety'),
                ('conversation',2008,['legal','conversational'],'A dialogue'),
                ('interview',2008,['legal'],'An INTERVIEW with an official'),
                ('undated',None,['legal'],'A regulation about product safety'),
            ]:
                title=f'Besluit van 21 maart {year}\n' if year else 'Besluit zonder datum\n'
                rows.append(dict(id=id,source='eurlex',text=title+'De regering heeft de nieuwe regels bekendgemaakt. '*10,
                                 created='1958-01-01, 2016-12-31'))
                annotations.append(dict(id=id,content_integrity='complete',content_ratio='complete_content',
                    content_length='substantial',content_quality='excellent',content_type=kind,one_sentence_description=description))
            pq.write_table(pa.Table.from_pylist(rows),p/'data.parquet')
            pq.write_table(pa.Table.from_pylist(annotations),p/'meta.parquet')
            result=list(documents([(source,dict(data=p/'data.parquet',metadata=p/'meta.parquet'),{})]))
            self.assertEqual([r['upstream_id'] for r in result],['keep'])
            self.assertEqual(result[0]['document_year_from_text'],2008)
            self.assertEqual(result[0]['text'],rows[0]['text'])


class DutchGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy
        try:cls.nlp=spacy.load('nl_core_news_md')
        except OSError:raise unittest.SkipTest('Install nl_core_news_md for parser integration tests')
        cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name)
        words='word\nwordt\nben\nbent\nsysteem\nregering\nmensen\nonmiddellijk\n'
        (p/'words.txt').write_text(words)
        profile=load_profile('nl');profile['lexicon']=str(p/'words.txt');profile['lexicon_sha256']=hashlib.sha256(words.encode()).hexdigest()
        cls.adapter=DutchPack(profile);cls.book=cls.adapter.load_rulebook('config/dutch_rules.json')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def edits(self,text):
        return self.adapter.candidates(next(iter(self.nlp(text).sents)),self.book)

    def grammar(self,text):
        return {(e.corruption_type,e.original.lower(),e.replacement.lower()) for e in self.edits(text) if e.corruption_type!='spelling'}

    def test_pronoun_agreement_and_dt(self):
        self.assertIn(('verb_dt','word','wordt'),self.grammar('Ik word morgen door de nieuwe minister ontvangen.'))
        self.assertIn(('verb_dt','wordt','word'),self.grammar('Hij wordt morgen door de nieuwe minister ontvangen.'))
        self.assertIn(('subject_verb_agreement','hebben','heeft'),self.grammar('Wij hebben vandaag een belangrijke beslissing genomen.'))

    def test_ambiguous_pronouns_and_inversion_abstain(self):
        for text in ['Zij is morgen alweer bij ons op bezoek.','Zij zijn morgen alweer bij ons op bezoek.','U hebt vandaag heel goed aan deze zaak gewerkt.','Word jij morgen door de nieuwe minister ontvangen?','Ik en mijn broer worden morgen door de minister ontvangen.']:
            self.assertFalse(any(e[0] in {'verb_dt','subject_verb_agreement'} for e in self.grammar(text)),text)

    def test_dual_gender_and_proper_names_abstain(self):
        self.assertNotIn(('article_gender','de','het'),self.grammar('De deksel ligt vandaag weer op de juiste plaats.'))
        self.assertNotIn(('article_gender','het','de'),self.grammar('Het deksel ligt vandaag weer op de juiste plaats.'))

    def test_unambiguous_determiners(self):
        self.assertIn(('article_number','de','het'),self.grammar('De mensen moeten morgen hun nieuwe paspoort ophalen.'))
        self.assertIn(('article_gender','het','de'),self.grammar('Het systeem werkt vandaag gelukkig weer zonder problemen.'))
        self.assertIn(('demonstrative_agreement','deze','dit'),self.grammar('Deze regering heeft de nieuwe wet gisteren aangenomen.'))

    def test_demonstrative_complementizer_ambiguity_abstains(self):
        # Both "Ik hoor die mensen ..." and "Ik hoor dat mensen ..." are valid.
        self.assertNotIn(('demonstrative_agreement','die','dat'),
                         self.grammar('Ik hoor die mensen in de tuin zingen.'))

    def test_paragraph_cap_preserves_offsets_and_is_deterministic(self):
        text='\n'.join(f'De overheid heeft de nieuwe regels voor sector {i} gisteren bekendgemaakt.' for i in range(30))
        document=dict(document_id='test-document',text=text)
        blocks=list(self.adapter.paragraphs(document))
        self.assertEqual(len(blocks),12)
        self.assertEqual(blocks,list(self.adapter.paragraphs(document)))
        self.assertEqual([offset for _,offset in blocks],sorted(offset for _,offset in blocks))
        for block,offset in blocks:
            self.assertEqual(text[offset:offset+len(block)],block)

    def test_source_fragments_and_spacing_are_rejected(self):
        for text in [
            'Of omdat de gemeente te weinig capaciteit heeft om plannen te ontwikkelen.',
            'Zonder dat iemand de deur voor je hoeft open te doen of je een drempel over moet tillen.',
            "Als overheid moeten we de centrale duider proberen te blijven', gaf Leonie aan.",
            'Het Centraal Orgaan opvang Asielzoekers, het COA,legt hiervoor een maatregel op.',
        ]:
            sent=next(iter(self.nlp(text).sents))
            excluded=self.adapter.exclusions
            try:
                self.adapter.exclusions=set()
                self.assertIsNotNone(self.adapter.sentence_rejection(sent,set()),text)
            finally:
                self.adapter.exclusions=excluded
        text="De collega's komen 's avonds samen om het plan te bespreken."
        self.assertIsNone(self.adapter.sentence_rejection(next(iter(self.nlp(text).sents)),set()))

    def test_plural_rule_requires_changed_noun_lemma(self):
        text='De woning ligt op een terp aan het eind van de strekdam die de vaargeul van het Zwarte Water in de Zuiderzee begeleide.'
        self.assertNotIn(('article_number','de','het'),self.grammar(text))

    def test_plural_object_can_become_valid_nominalized_infinitive(self):
        # Both de reizen (trips) and het reizen (travelling) are grammatical here.
        self.assertNotIn(('article_number','de','het'),
                         self.grammar('Wij genieten van de reizen naar andere landen.'))
        # A plural predicate blocks that singular nominalization interpretation.
        self.assertIn(('article_number','de','het'),
                      self.grammar('De reizen naar andere landen duren meestal een week.'))

    def test_real_words_never_become_spelling_negatives(self):
        book={'rules':[dict(id='danger',family='spelling',correct='mensen',incorrect='regering')]}
        sent=next(iter(self.nlp('De mensen moeten morgen hun nieuwe paspoort ophalen.').sents))
        self.assertEqual(next(t for t in sent if t.text=='mensen').pos_,'NOUN')
        self.assertEqual(self.adapter.candidates(sent,book),[])


if __name__=='__main__':unittest.main()
