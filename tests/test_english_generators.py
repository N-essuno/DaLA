import copy
import unittest
from unittest.mock import Mock
import spacy
from dala.character_operations import replacements
from dala.edits import apply_edits
from dala.language_check import LanguageCheck
from dala.language_packs.english import EnglishPack, permits
from dala.language_packs.english_fallback import candidates
from dala.profiles import load_profile, resource


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=load_profile('en');cls.pack=EnglishPack(cls.profile)
        cls.book=cls.pack.load_rulebook(resource(cls.profile,'rulebook'))
        cls.nlp=spacy.load(cls.profile['parser'])
        cls.rules={r['id']:r for r in cls.book['rules']}

    def test_generalizes_and_licenses_only_recorded_operation(self):
        for op,good,bad in [('delete_doubled_consonant','butterfly','buterfly'),
                            ('duplicate_consonant','hospital','hosppital'),
                            ('transpose_internal','computer','comptuer')]:
            rule=self.rules['spelling_pattern:'+op]
            self.assertIn(bad,replacements(good,rule))
            self.assertNotIn(good,replacements(good,rule))
            self.assertFalse(replacements('form',rule))
            self.assertFalse(replacements('COVID19',rule))

    def test_doubling_variants_are_protected_before_inflectional_endings(self):
        rule=self.rules['spelling_pattern:delete_doubled_consonant']
        for source,variant in [('reprogrammed','reprogramed'),('travelled','traveled'),
                               ('traveller','traveler'),('focussed','focused'),
                               ('fulfilment','fulfillment')]:
            self.assertNotIn(variant,replacements(source,rule))
        rule=self.rules['spelling_pattern:duplicate_consonant']
        for source,variant in [('focused','focussed'),('traveled','travelled')]:
            self.assertNotIn(variant,replacements(source,rule))

    def test_names_abbreviations_and_known_words_are_not_character_candidates(self):
        sent=next(self.nlp('Microsoft and NASA publish reports about nuclear research.').sents)
        edits=[e for e in self.pack.candidates(sent,self.book) if e.rule_id.startswith('spelling_pattern:')]
        self.assertTrue(edits)
        self.assertFalse(any(e.original in {'Microsoft','NASA'} for e in edits))
        self.assertTrue(all(permits(self.rules[e.rule_id],e) for e in edits))
        self.assertTrue(all(not sent.doc.vocab[e.replacement.lower()].has_vector for e in edits))

    def test_real_word_transposition_is_vetoed(self):
        rule=self.rules['spelling_pattern:transpose_internal']
        self.assertIn('casual',replacements('causal',rule))
        sent=next(self.nlp('The causal relationship remains unclear after the latest experiment.').sents)
        self.assertFalse(any(e.original.lower()=='causal' and e.replacement.lower()=='casual'
                             for e in self.pack.candidates(sent,self.book)))

    def test_attested_spelling_is_not_crowded_out_by_generated_candidates(self):
        sent=next(self.nlp('The government supports research into renewable energy.').sents)
        edits=self.pack.candidates(sent,self.book)
        self.assertTrue(any(e.rule_id.startswith('spelling_pattern:') for e in edits))
        for seed in (1,42,4242):
            chosen=self.pack.select_edits(sent.text,edits,seed,1)
            self.assertEqual(chosen[0].rule_id,'spelling:government:goverment')

    def test_required_progressive_auxiliary_deletion(self):
        sent=next(self.nlp('The boys are reading books in the quiet room.').sents)
        edits=[e for e in candidates(sent,self.book['rules']) if e.corruption_type=='word_deletion']
        self.assertEqual(len(edits),1)
        self.assertEqual(apply_edits(sent.text,edits),'The boys reading books in the quiet room.')
        self.assertTrue(permits(self.rules[edits[0].rule_id],edits[0]))
        for text in ['They have been reading books in the room.',
                     'They can read books in the room.',
                     'The boys are here in the quiet room.',
                     'The boys are reading and she writes books.']:
            self.assertFalse([e for e in candidates(next(self.nlp(text).sents),self.book['rules']) if e.corruption_type=='word_deletion'])

    def test_article_swap_has_boundaries_and_exact_inverse(self):
        sent=next(self.nlp('They put the box on the floor in the room.').sents)
        edits=[e for e in candidates(sent,self.book['rules']) if e.corruption_type=='word_swap']
        self.assertTrue(edits)
        self.assertTrue(any(e.original=='the box' and e.replacement=='box the' for e in edits))
        for e in edits:
            self.assertTrue(permits(self.rules[e.rule_id],e))
            text=apply_edits(sent.text,[e])
            restored=text[:e.start]+e.original+text[e.start+len(e.replacement):]
            self.assertEqual(restored,sent.text)
        sent=next(self.nlp('They read the history book after lunch.').sents)
        self.assertFalse([e for e in candidates(sent,self.book['rules']) if e.original=='the history'])

    def test_fallback_does_not_override_targeted_grammar(self):
        sent=next(self.nlp('They have written the report in the room.').sents)
        edits=self.pack.candidates(sent,self.book)
        self.assertTrue(any(e.corruption_type=='perfect_participle' for e in edits))
        self.assertFalse(any(e.rule_id.startswith('fallback:') for e in edits))

    def test_article_swap_validator_accepts_accented_english_words(self):
        from dataclasses import replace
        for text,span in [('They met at the café after work.','the café'),
                          ('The building has a façade of marble.','a façade')]:
            sent=next(self.nlp(text).sents)
            edits=[e for e in candidates(sent,self.book['rules']) if e.original==span]
            self.assertEqual(len(edits),1)
            rule=self.rules[edits[0].rule_id]
            self.assertTrue(permits(rule,edits[0]))
            self.assertFalse(permits(rule,replace(edits[0],original='the café_2',replacement='café_2 the')))
            self.assertFalse(permits(rule,replace(edits[0],replacement=span)))

    def test_seeded_choice_and_single_fallback(self):
        sent=next(self.nlp('They put the box on the floor in the room.').sents)
        edits=self.pack.candidates(sent,self.book)
        a=self.pack.select_edits(sent.text,edits,4242,2)
        self.assertEqual(a,self.pack.select_edits(sent.text,list(reversed(edits)),4242,2))
        self.assertEqual(len(a),1)
        self.assertTrue(a[0].rule_id.startswith('fallback:'))

    def test_valid_dialect_word_is_rejected_even_if_other_dialect_flags_it(self):
        checker=object.__new__(LanguageCheck);checker.dialects=('en-US','en-GB')
        def response(flag,word):
            return {'matches':[dict(offset=12,length=len(word),message='spelling',rule=dict(id='SPELL',issueType='misspelling'))] if flag else []}
        checker.check=Mock(side_effect=[response(False,'traveler'),response(True,'traveller'),response(False,'traveler'),response(False,'traveller')])
        verdict,reason=checker.screen('A traveler arrived.','A traveller arrived.',[dict(rule_id='pattern',corruption_type='spelling',original='traveler',replacement='traveller',requires_lexical_screen=True,corrupted_start=2,corrupted_end=11)])
        self.assertIsNone(verdict);self.assertEqual(reason,'checker_lexical_guard')
