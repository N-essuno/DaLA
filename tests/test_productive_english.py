import copy
import unittest
import spacy
from dala.profiles import load_profile,resource
from dala.language_packs.english import EnglishPack,permits
from dala.edits import apply_edits


class ProductiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=load_profile('en');cls.pack=EnglishPack(cls.profile)
        cls.book=cls.pack.load_rulebook(resource(cls.profile,'rulebook'))
        cls.nlp=spacy.load(cls.profile['parser'])

    def family(self,text,family):
        return [e for e in self.pack.candidates(next(self.nlp(text).sents),self.book) if e.corruption_type==family]

    def test_generalizes_beyond_attested_word_pairs(self):
        for text,family,before,after in [
            ('They admire the beautiful flowers in the garden.','subject_verb_agreement','admire','admires'),
            ('They should negotiate a better agreement with the government.','modal_verb_form','negotiate','negotiating'),
            ('They did not write the final report about the pollution.','do_support_form','write','wrote'),
            ('They have written the final report about the pollution.','perfect_participle','written','wrote'),
            ('Many buildings have remained empty since the factory closed.','noun_number','buildings','building'),
        ]:
            edits=self.family(text,family)
            self.assertTrue(any(e.original==before and e.replacement==after for e in edits),(text,edits))
            for e in edits:self.assertTrue(permits(next(r for r in self.book['rules'] if r['id']==e.rule_id),e))

    def test_incomplete_reverse_dictionary_is_not_a_false_rejection(self):
        edits=self.family('He gave many lectures and collected seaweed from Key West to Halifax.','noun_number')
        self.assertTrue(edits)
        rule=next(r for r in self.book['rules'] if r['id']==edits[0].rule_id)
        self.assertTrue(permits(rule,edits[0]))

    def test_syncretic_forms_are_not_corrupted(self):
        for text,family in [
            ('They have cut the paper into several small pieces.','perfect_participle'),
            ('They did not put the newspaper on the kitchen table.','do_support_form'),
            ('Many sheep have remained outside during the cold winter.','noun_number'),
        ]:self.assertFalse(self.family(text,family))

    def test_collective_and_notional_agreement_abstention(self):
        for text in ['A school is investigating the reports of poor teaching.',
                     'The team is investigating the reports of poor teaching.',
                     'She and I have enough time to finish the work.']:
            self.assertFalse(self.family(text,'subject_verb_agreement'))

    def test_demonstratives_do_not_change_valid_syncretic_number(self):
        for text in ['We caught this fish in the river yesterday.',
                     'They published this series about the history of science.',
                     'They considered these means of improving public services.']:
            self.assertFalse(self.family(text,'demonstrative_number'))

    def test_irrealis_is_preserved(self):
        edits=self.family('I wish she were able to work with the government.','subject_verb_agreement')
        self.assertFalse(any(e.original=='were' for e in edits))

    def test_distributive_subject_and_plural_subject(self):
        self.assertTrue(self.family('Every researcher examines the results of the experiment.','subject_verb_agreement'))
        self.assertTrue(self.family('These researchers examine the results of the experiment.','subject_verb_agreement'))

    def test_no_unknown_word_inflection(self):
        self.assertFalse(self.family('They should zorbulate the material before the experiment.','modal_verb_form'))

    def test_context_word_lists_are_inputs(self):
        profile=copy.deepcopy(self.profile);profile['grammar']['modals']=[]
        pack=EnglishPack(profile)
        sent=next(self.nlp('They should negotiate a better agreement with the government.').sents)
        self.assertFalse([e for e in pack.candidates(sent,self.book) if e.corruption_type=='modal_verb_form'])

    def test_expanded_spelling_has_traceable_evidence(self):
        rules=[r for r in self.book['rules'] if r['family']=='spelling' and 'correct' in r]
        self.assertGreaterEqual(len({r['correct'] for r in rules}),45)
        self.assertEqual(len(rules),len({(r['correct'],r['incorrect']) for r in rules}))
        for r in rules:
            if r.get('evidence_kind')=='published_list':
                self.assertIsNone(r['distinct_sentence_support'])
                self.assertTrue(r['published_sources'])
                for source in r['published_sources']:
                    self.assertIn(source,self.book['published_sources'])
        self.assertFalse({'costumers','wether','their','there','colour','color'} & {r['incorrect'] for r in rules})

    def test_new_spelling_candidates_and_case(self):
        for text,before,after in [
            ('The committee published its final report yesterday.','committee','commitee'),
            ('Accommodation remains expensive in the city centre.','Accommodation','Acomodation'),
            ('They reached a consensus after several hours of discussion.','consensus','concensus')]:
            edits=self.family(text,'spelling')
            self.assertTrue(any(e.original==before and e.replacement==after for e in edits),(text,edits))

    def test_untraceable_published_spelling_is_rejected(self):
        import json,tempfile
        book=copy.deepcopy(self.book)
        rule=next(r for r in book['rules'] if r.get('evidence_kind')=='published_list')
        rule['published_sources']=['nonexistent']
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
            json.dump(book,f);f.flush()
            with self.assertRaisesRegex(ValueError,'lacks evidence'):
                self.pack.load_rulebook(f.name)
