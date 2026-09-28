import json
from pathlib import Path
import unittest

from dala.language_packs.dutch import DutchPack, permits
from dala.profiles import load_profile, resource
from dala.pair_pipeline import order_documents, source_functions, validate_pairs
from dala.edits import Corruption


class ExpandedDutchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy
        cls.profile=load_profile('nl_expanded')
        if not resource(cls.profile,'morphology').is_file():
            raise unittest.SkipTest('Run scripts.prepare_dutch_morphology for resource integration tests')
        try: cls.nlp=spacy.load('nl_core_news_md')
        except OSError: raise unittest.SkipTest('Install Dutch parser')
        cls.adapter=DutchPack(cls.profile)
        cls.book=cls.adapter.load_rulebook(resource(cls.profile,'rulebook'))

    def grammar(self,text):
        s=next(iter(self.nlp(text).sents))
        return {(e.corruption_type,e.original.lower(),e.replacement.lower()) for e in self.adapter.candidates(s,self.book) if e.corruption_type!='spelling'}

    def test_expanded_lexicon_and_adjectival_interveners(self):
        self.assertGreater(len(self.adapter.noun_articles),100000)
        self.assertNotIn('deksel',self.adapter.noun_articles)
        self.assertIn(('article_gender','het','de'),self.grammar('Wij bespreken morgen het nieuwe voorstel van de regering.'))
        self.assertIn(('possessive_agreement','onze','ons'),self.grammar('Onze regering neemt morgen een nieuwe belangrijke beslissing.'))

    def test_productive_verb_not_in_original_inventory(self):
        self.assertIn(('subject_verb_agreement','beoordelen','beoordeelt'),self.grammar('Wij beoordelen morgen het nieuwe voorstel van de regering.'))
        self.assertFalse(any(x[0]=='subject_verb_agreement' for x in self.grammar('Zij beoordelen morgen het nieuwe voorstel van de regering.')))
        self.assertFalse(any(x[0]=='subject_verb_agreement' for x in self.grammar('Beoordeel jij morgen het nieuwe voorstel van de regering?')))

    def test_indefinite_neuter_adjective_only(self):
        self.assertIn(('adjective_inflection','belangrijk','belangrijke'),self.grammar('De regering heeft gisteren een belangrijk besluit genomen.'))
        for text in ['Het openbaar vervoer blijft vandaag gewoon normaal rijden.','Een groot man heeft gisteren een belangrijke prijs gekregen.']:
            self.assertFalse(any(x[0]=='adjective_inflection' for x in self.grammar(text)))

    def test_nominalized_infinitive_article_abstention(self):
        # Both De suiker toevoegen ... and Het suiker toevoegen ... can be parsed
        # as subject infinitival phrases; a noun gender change is unsafe here.
        self.assertNotIn(('article_gender','de','het'),self.grammar('De suiker toevoegen kost ons vandaag veel extra tijd.'))

    def test_human_relative_and_complementizer_abstention(self):
        for text in ['Het meisje dat hier woont heeft gisteren haar diploma gekregen.',
                     'Het feit dat hij hier woont heeft geen bijzondere betekenis.']:
            self.assertFalse(any(x[0]=='relative_pronoun_agreement' for x in self.grammar(text)))

    def test_remaining_source_error_patterns(self):
        for text in ['Dat maakt dat de beperkende maatregelen van de lockdown voor hen extra zwaar.',
                     'Diverse Nederlandse bedrijven, kennisinstellingen en de Nederlandse ambassade in Yangon helpt om de aardappelteelt een impuls te geven.',
                     'Deze ligt , aan de achterzijde van het pand met daarnaast een berghok.']:
            self.assertIsNotNone(self.adapter.sentence_rejection(next(iter(self.nlp(text).sents)),set()),text)

    def test_validation_source_guards_and_counterexamples(self):
        adapter=DutchPack(load_profile('nl_validation'))
        rejected=[
            'Waarborgsom Partijen moeten een waarborgsom betalen aan het gemeentelijk centraal stembureau.',
            'De plannen worden morgen opnieuw beoor-deeld door de gemeenteraad.',
            'Het jaar werd gedomineerd door een pandemie die wij alleen uit de geschiedenisboeken kende.',
            'I. Laboratoria moeten de resultaten van het onderzoek zorgvuldig controleren.',
        ]
        for text in rejected:
            self.assertIsNotNone(adapter.sentence_rejection(next(iter(self.nlp(text).sents)),set()),text)
        for text in [
            'De studenten presenteren vandaag hun nieuwe plannen voor de stad.',
            'Wij kenden de plannen al voordat de minister zijn besluit bekendmaakte.',
            'Hij heeft gisteren de nieuwe plannen voor de stad gepresenteerd.',
            'Mijn broer en ik bekijken morgen de nieuwe plannen van de minister.',
        ]:
            self.assertIsNone(adapter.sentence_rejection(next(iter(self.nlp(text).sents)),set()),text)


class SharedSourceTests(unittest.TestCase):
    def test_round_robin_keeps_all_documents(self):
        docs=[dict(document_id=str(i),source_name='a' if i<6 else 'b') for i in range(8)]
        ordered=order_documents(docs,42,dict(source_sampling='round_robin'))
        self.assertEqual([r['source_name'] for r in ordered[:4]],['a','b','a','b'])
        self.assertEqual({d['document_id'] for d in ordered},{d['document_id'] for d in docs})
        self.assertEqual(ordered,order_documents(reversed(docs),42,dict(source_sampling='round_robin')))
        self.assertEqual(source_functions(load_profile('nl_expanded'))[0].__module__,'dala.dynaword')

    def test_dictionary_mapping_validation(self):
        rule=dict(operator='dictionary_mapping',mappings={'beoordelen':'beoordeelt'})
        self.assertTrue(permits(rule,Corruption('r','subject_verb_agreement',0,10,'beoordelen','beoordeelt',())))
        self.assertFalse(permits(rule,Corruption('r','subject_verb_agreement',0,10,'beoordelen','beoordelen',())))

    def test_dictionary_mapping_export_validation(self):
        rule=dict(id='agreement',family='subject_verb_agreement',
                  operator='dictionary_mapping',mappings={'beoordelen':'beoordeelt'})
        edit=Corruption('agreement','subject_verb_agreement',4,14,'beoordelen','beoordeelt',(0,1)).record()
        edit.update(corrupted_start=4,corrupted_end=4+len(edit["replacement"]))
        pair=dict(pair_id='p',document_id='d',split='train',
                  original='Wij beoordelen het voorstel.',corrupted='Wij beoordeelt het voorstel.',edits=[edit])
        book=dict(rules=[rule],edit_validator='dala.language_packs.dutch:permits')
        self.assertEqual(validate_pairs([pair],book)['pairs'],1)
        edit['replacement']='beoordeel'
        with self.assertRaisesRegex(ValueError,'licensed rule'):
            validate_pairs([pair],book)


if __name__=='__main__':unittest.main()
