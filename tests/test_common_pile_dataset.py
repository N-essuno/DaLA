import copy
import csv
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import spacy

from dala.build_english import export_dataset, split_for, task_rows, validate_pairs
from dala.common_pile import documents, sha256_file, snapshots
from dala.curation import NearDuplicates, sentence_rejection
from dala.dataset_review import review_sheet, reviewed_export
from dala.english_rules import apply_edits, candidates, load_rulebook, select_edits
from dala.language_check import LanguageCheck
from dala.validate_dataset import validate


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.nlp = spacy.load('en_core_web_md')
        except OSError:
            raise unittest.SkipTest('Install en_core_web_md to run parser integration tests')
        cls.book = load_rulebook()

    def edits(self, text, family):
        return [e for e in candidates(next(self.nlp(text).sents), self.book) if e.corruption_type == family]

    def test_positive_grammar_contexts(self):
        examples = [
            ('They have enough time to finish the work.', 'subject_verb_agreement', 'They has'),
            ('These books provide a useful introduction to the subject.', 'demonstrative_number', 'This books'),
            ('Many children enjoy reading books about wild animals.', 'noun_number', 'Many child'),
            ('They can work together to solve the difficult problem.', 'modal_verb_form', 'can working'),
            ('They did not know the answer to that question.', 'do_support_form', 'not knew'),
            ('They have seen the results of the latest experiment.', 'perfect_participle', 'have saw'),
        ]
        for text, family, fragment in examples:
            with self.subTest(family=family):
                edits = self.edits(text, family)
                self.assertTrue(edits)
                self.assertIn(fragment, apply_edits(text, [edits[0]]))

    def test_audited_source_error_is_excluded(self):
        text = ('This system was inaccessible for many because of rations generally did not have '
                'access to the internet or smartphones in order to apply for the e-coupons.')
        self.assertEqual(sentence_rejection(next(self.nlp(text).sents), set()), 'audited_source_exclusion')

    def test_no_lexical_auxiliary_confusion(self):
        self.assertFalse(self.edits('They do the work in the government library.', 'do_support_form'))
        self.assertFalse(self.edits('The can contains enough food for the whole family.', 'modal_verb_form'))
        self.assertFalse(self.edits('The known problems include pollution in the local environment.', 'perfect_participle'))

    def test_collectives_coordination_and_irrealis_not_changed(self):
        for text in ['The team have enough time to finish the work.',
                     'She and I have enough time to finish the work.',
                     'I wish she were able to work with the government.']:
            self.assertFalse(self.edits(text, 'subject_verb_agreement'))

    def test_no_noun_compound_changes(self):
        self.assertFalse(self.edits('Many animal rights groups support the new policy.', 'noun_number'))
        self.assertFalse(self.edits('These government policies have improved the local environment.', 'demonstrative_number'))

    def test_spelling_is_attested_not_random_noise(self):
        text = 'The government protects the environment because pollution causes harm.'
        edits = self.edits(text, 'spelling')
        self.assertTrue(any(e.original == 'government' and e.replacement == 'goverment' for e in edits))
        self.assertFalse(self.edits('The colour of the theatre remains unchanged after the renovation.', 'spelling'))

    def test_composition_preserves_independent_error_count(self):
        text = 'They have enough money because the government protects the environment.'
        edits = candidates(next(self.nlp(text).sents), self.book)
        choices = [select_edits(text, edits, seed, 3) for seed in range(20)]
        self.assertTrue(any(len(x) == 3 for x in choices))
        for chosen in choices:
            self.assertLessEqual(sum(e.corruption_type != 'spelling' for e in chosen), 1)
            protected = set()
            for e in chosen:
                self.assertFalse(protected.intersection(e.protected_tokens))
                protected.update(e.protected_tokens)
            self.assertNotEqual(text, apply_edits(text, chosen))
        self.assertEqual(select_edits(text, edits, 42, 3), select_edits(text, edits, 42, 3))

    def test_overlap_and_source_mismatch_fail(self):
        text = 'The government protects the environment because pollution causes harm.'
        e = self.edits(text, 'spelling')[0]
        with self.assertRaises(ValueError):
            apply_edits(text, [e, e])
        with self.assertRaises(ValueError):
            apply_edits('Different sentence.', [e])

    def test_source_filters(self):
        bad = [
            'The goverment protects the environment because pollution causes harm.',
            'These book explains the importance of protecting the natural environment.',
            'The minister said “this government will protect the environment”.',
        ]
        for text in bad:
            self.assertIsNotNone(sentence_rejection(next(self.nlp(text).sents), {'goverment'}))
        text = 'The government protects the environment because pollution causes harm.'
        self.assertIsNone(sentence_rejection(next(self.nlp(text).sents), {'goverment'}))

    def pair(self, text, doc='doc', split='train'):
        edits = select_edits(text, candidates(next(self.nlp(text).sents), self.book), seed=4242, max_errors=2)
        records, delta = [], 0
        for e in edits:
            r = asdict(e)
            r.update(corrupted_start=e.start+delta, corrupted_end=e.start+delta+len(e.replacement))
            records.append(r); delta += len(e.replacement)-len(e.original)
        return dict(pair_id=doc, document_id=doc, split=split, source_name='fixture',
            source_dataset='fixture', source_revision='a'*40, document_sha256='b'*64,
            url='https://example.org/'+doc, license='fixture', original=text,
            corrupted=apply_edits(text, edits), edits=records, quality_status='automatic_screening_only')

    def test_task_views_and_identity_controls(self):
        p = self.pair('They have enough money because the government protects the environment.')
        rows = list(task_rows(p))
        self.assertEqual([r[0]['label'] for r in rows], ['correct', 'incorrect'])
        self.assertEqual(rows[0][2]['samples']['content'], rows[0][2]['samples']['response'])
        self.assertEqual(rows[1][2]['samples']['response'], p['original'])
        self.assertNotIn('corruption_type', rows[1][2]['direction'])

    def test_document_leakage_and_duplicate_text_rejected(self):
        a = self.pair('They have enough money because the government protects the environment.')
        b = self.pair('These books provide a useful introduction to the subject.', doc='second', split='test')
        b['document_id'] = a['document_id']
        with self.assertRaises(ValueError):
            validate_pairs([a, b], self.book)
        with self.assertRaises(ValueError):
            validate_pairs([a, copy.deepcopy(a)], self.book)
        self.assertEqual(split_for('article', 17), split_for('article', 17))

    def test_export_review_roundtrip_and_tamper_detection(self):
        pair = self.pair('They have enough money because the government protects the environment.')
        document = {k: pair[k] for k in ['document_id','source_dataset','source_revision','document_sha256','url','license']}
        manifest = dict(seed=4242, quality_status='automatic_screening_only', max_errors=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'dataset'
            export_dataset([pair], root, manifest, self.book, [document])
            self.assertTrue(validate(root)['balanced_labels'])
            sheet = Path(tmp)/'all.csv'; review_sheet(root, sheet)
            with self.assertRaises(ValueError):
                reviewed_export(root, sheet, Path(tmp)/'blank')
            with sheet.open(newline='') as f:
                rows = list(csv.DictReader(f))
            rows[0].update(original_correct='yes', corrupted_incorrect='yes', edits_valid='yes', reviewer='test:reviewer')
            with sheet.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            reviewed_export(root,sheet,Path(tmp)/'accepted')
            self.assertTrue(validate(Path(tmp)/'accepted')['balanced_labels'])
            (root/'train/acceptability_it.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                validate(root)

    def test_jsonl_unicode_separators_are_not_record_boundaries(self):
        from dala.dataset_review import load_pairs
        pair=self.pair('They have enough money because\u2028the government protects the environment.')
        document={k:pair[k] for k in ['document_id','source_dataset','source_revision','document_sha256','url','license']}
        manifest=dict(seed=4242,quality_status='automatic_screening_only',max_errors=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'
            export_dataset([pair],root,manifest,self.book,[document])
            self.assertEqual(load_pairs(root)[0]['original'],pair['original'])
            self.assertTrue(validate(root)['task_views_match_pairs'])


class InfrastructureTests(unittest.TestCase):
    def test_exact_and_near_duplicate_filter(self):
        d = NearDuplicates()
        self.assertTrue(d.add('The government has decided to protect the natural environment today.'))
        self.assertFalse(d.add('THE government has decided to protect the natural environment today!'))
        self.assertFalse(d.add('The government has decided to protect the natural environment tomorrow.'))
        self.assertTrue(d.add('These students studied the history of medieval literature in France.'))

    def test_source_provenance_and_cache_integrity(self):
        source = dict(name='fixture',repo_id='fixture/repo', revision='a'*40, files=['one.jsonl.gz'],
                      domains=['example.org'], source_values=['fixture'],license_contains='CC BY')
        row = dict(id=1,text='This is original source text.',source='fixture',
                   metadata=dict(url='https://example.org/article',license='CC BY'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'sources.json'; config.write_text(json.dumps({'sources':[source]}))
            shard=root/'cache/fixture'/('a'*40)/'one.jsonl.gz';shard.parent.mkdir(parents=True)
            with gzip.open(shard,'wt') as f:f.write(json.dumps(row)+'\n')
            shard.with_suffix('.gz.sha256').write_text(sha256_file(shard)+'\n')
            docs=list(documents(snapshots(config,root/'cache',offline=True)))
            self.assertEqual(docs[0]['text'],row['text'])
            self.assertEqual(docs[0]['source_revision'],'a'*40)
            shard.write_bytes(b'changed')
            with self.assertRaises(ValueError):snapshots(config,root/'cache',offline=True)

    def test_utf16_checker_offsets(self):
        data={'matches':[dict(offset=3,length=4,rule=dict(id='x',issueType='misspelling'),message='bad')]}
        hit=LanguageCheck.hard_matches(data,'🧠 taht works.')[0]
        self.assertEqual((hit['start'],hit['end']),(2,6))

    def test_checker_requires_each_edit_and_ignores_style(self):
        checker=LanguageCheck.__new__(LanguageCheck)
        checker.check=Mock(return_value={'matches':[]})
        verdict, reason=checker.screen('That works.', 'Taht works.', [dict(rule_id='x',corruption_type='spelling',corrupted_start=0,corrupted_end=4)])
        self.assertIsNone(verdict);self.assertEqual(reason,'checker_missed_edit')
        checker.check=Mock(side_effect=[{'matches':[]},{'matches':[dict(offset=0,length=4,rule=dict(id='spell',issueType='misspelling'),message='bad')]}])
        verdict,reason=checker.screen('That works.','Taht works.',[dict(rule_id='x',corruption_type='spelling',corrupted_start=0,corrupted_end=4)])
        self.assertIsNone(reason);self.assertEqual(verdict['source_hard_diagnostics'],0)
