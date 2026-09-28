import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dala.english import candidates, choose_edit
from dala.languages import get_language
from dala.multilingual import create_english, prepare_english
from dala.ud_annotations import parse_conllu


def sentence(text, specs):
    """Fixtures use explicit gold tags/dependencies, independent of NLP models."""
    lines = [f"# text = {text}", "# sent_id = fixture"]
    for idx, spec in enumerate(specs, 1):
        form, lemma, upos, xpos, feats, head, dep = spec.split()
        lines.append('\t'.join([str(idx), form, lemma, upos, xpos, feats, head, dep, '_', '_']))
    return parse_conllu('\n'.join(lines)).iloc[0].to_dict()


HE_WORKS = ["He he PRON PRP Case=Nom|Number=Sing 2 nsubj",
            "works work VERB VBZ Mood=Ind|VerbForm=Fin|Tense=Pres 0 root",
            ". . PUNCT . _ 2 punct"]


class EnglishTests(unittest.TestCase):
    def check_edit(self, text, specs, rule, expected):
        row = sentence(text, specs)
        edits = [e for e in candidates(row) if e.corruption_type == rule]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].apply(text), expected)
        return row

    def test_agreement_and_case(self):
        row = self.check_edit('He works.', HE_WORKS, 'subject_verb_agreement', 'He work.')
        self.assertIn('Him works.', [e.apply(row['doc']) for e in candidates(row)])

    def test_whitespace_preserved(self):
        self.check_edit('He  works.\t', HE_WORKS, 'subject_verb_agreement', 'He  work.\t')

    def test_internal_i_does_not_introduce_capitalization_error(self):
        row = sentence('Today I work.', [
            'Today today ADV RB _ 3 advmod', 'I I PRON PRP Case=Nom 3 nsubj',
            'work work VERB VBP Mood=Ind|VerbForm=Fin 0 root', '. . PUNCT . _ 3 punct'])
        self.assertIn('Today me work.', [e.apply(row['doc']) for e in candidates(row)])

    def test_demonstrative(self):
        self.check_edit('These books fell.', [
            'These this DET DT Number=Plur 2 det',
            'books book NOUN NNS Number=Plur 3 nsubj',
            'fell fall VERB VBD Mood=Ind|VerbForm=Fin 0 root',
            '. . PUNCT . _ 3 punct'], 'demonstrative_number', 'This books fell.')

    def test_auxiliary_forms(self):
        for aux, lemma, tag, verb, vlemma, vtag, rule, bad in [
            ('will', 'will', 'MD', 'go', 'go', 'VB', 'modal_verb_form', 'goes'),
            ('did', 'do', 'VBD', 'go', 'go', 'VB', 'do_support_form', 'went'),
            ('has', 'have', 'VBZ', 'gone', 'go', 'VBN', 'perfect_participle', 'went')]:
            with self.subTest(rule=rule):
                self.check_edit(f'She {aux} {verb}.', [
                    'She she PRON PRP Case=Nom 3 nsubj',
                    f'{aux} {lemma} AUX {tag} Mood=Ind|VerbForm=Fin 3 aux',
                    f'{verb} {vlemma} VERB {vtag} VerbForm=Inf 0 root',
                    '. . PUNCT . _ 3 punct'], rule, f'She {aux} {bad}.')

    def test_negated_do_support(self):
        self.check_edit('She did not go.', [
            'She she PRON PRP Case=Nom 4 nsubj',
            'did do AUX VBD Mood=Ind|VerbForm=Fin 4 aux',
            'not not PART RB Polarity=Neg 4 advmod',
            'go go VERB VB VerbForm=Inf 0 root', '. . PUNCT . _ 4 punct'],
            'do_support_form', 'She did not went.')

    def test_no_was_were_irrealis_ambiguity(self):
        row = sentence('He was.', [
            'He he PRON PRP Case=Nom 2 nsubj',
            'was be AUX VBD Mood=Ind|VerbForm=Fin 0 root', '. . PUNCT . _ 2 punct'])
        self.assertNotIn('subject_verb_agreement', [e.corruption_type for e in candidates(row)])

    def test_abstains_on_typo_and_misalignment(self):
        row = sentence('He works.', HE_WORKS)
        row['words'][0]['feats']['Typo'] = 'Yes'
        self.assertEqual(candidates(row), [])
        self.assertEqual(candidates(sentence('He werkz.', HE_WORKS)), [])

    def test_no_collective_agreement(self):
        row = sentence('Staff work.', [
            'Staff staff NOUN NN Number=Sing 2 nsubj',
            'work work VERB VBP Mood=Ind|VerbForm=Fin 0 root',
            '. . PUNCT . _ 2 punct'])
        self.assertEqual(candidates(row), [])

    def test_no_coordinated_subject(self):
        row = sentence('He and I work.', [
            'He he PRON PRP Case=Nom 4 nsubj', 'and and CCONJ CC _ 3 cc',
            'I I PRON PRP Case=Nom 1 conj',
            'work work VERB VBP Mood=Ind|VerbForm=Fin 0 root', '. . PUNCT . _ 4 punct'])
        self.assertEqual(candidates(row), [])

    def test_subjunctive_and_comparative_abstention(self):
        row = sentence('He works.', HE_WORKS)
        row['words'][1]['feats']['Mood'] = 'Sub'
        self.assertEqual(candidates(row), [])
        row = sentence('He works.', HE_WORKS)
        row['words'].append(dict(id='4', form='than', lemma='than', upos='SCONJ', xpos='IN',
                                 feats={}, head='2', deprel='mark', misc={}, span=None))
        self.assertEqual(candidates(row), [])

    def test_participle_syncretism_not_changed(self):
        row = sentence('She has worked.', [
            'She she PRON PRP Case=Nom 3 nsubj',
            'has have AUX VBZ Mood=Ind|VerbForm=Fin 3 aux',
            'worked work VERB VBN VerbForm=Part 0 root', '. . PUNCT . _ 3 punct'])
        self.assertNotIn('perfect_participle', [e.corruption_type for e in candidates(row)])

    def test_parser_multiword_empty_nodes_and_eof(self):
        data = '''# sent_id = contraction
# text = I can't go.
1\tI\tI\tPRON\tPRP\t_\t4\tnsubj\t_\t_
2-3\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
2\tca\tcan\tAUX\tMD\t_\t4\taux\t_\t_
3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
3.1\tghost\tghost\tX\tX\t_\t_\t_\t_\t_
4\tgo\tgo\tVERB\tVB\tVerbForm=Inf\t0\troot\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_'''
        row = parse_conllu(data).iloc[0]
        self.assertTrue(row.aligned)
        self.assertEqual(len(row.words), 5)
        self.assertIsNone(row.words[1]['span'])
        self.assertEqual(row.words[3]['span'], (8, 10))
        self.assertEqual(candidates(row), [])

    def test_balanced_pairs_and_determinism(self):
        row = sentence('He works.', HE_WORKS)
        other = copy.deepcopy(row)
        other['aligned'] = False
        data, audit, report = prepare_english(pd.DataFrame([row, other]))
        self.assertEqual(data.label.value_counts().to_dict(), {'correct': 1, 'incorrect': 1})
        self.assertEqual(report['abstained'], 1)
        self.assertEqual(choose_edit(row), choose_edit(row))
        self.assertEqual(audit.original_text.tolist(), ['He works.'])

    def test_empty_input(self):
        data, audit, report = prepare_english(pd.DataFrame())
        self.assertTrue(data.empty)
        self.assertTrue(audit.empty)
        self.assertEqual(report['pairs'], 0)

    def test_unsupported_language(self):
        with self.assertRaises(ValueError):
            get_language('fr')

    def test_split_integrity_and_local_export(self):
        source = {}
        for split, pronoun in [('train', 'He'), ('val', 'She'), ('test', 'They')]:
            specs = [f'{pronoun} {pronoun.lower()} PRON PRP Case=Nom 2 nsubj',
                     'works work VERB VBZ Mood=Ind|VerbForm=Fin 0 root',
                     'at at ADP IN _ 4 case', 'home home NOUN NN Number=Sing 2 obl',
                     'every every DET DT _ 6 det', 'day day NOUN NN Number=Sing 2 obl',
                     '. . PUNCT . _ 2 punct']
            if pronoun == 'They':
                specs[1] = 'work work VERB VBP Mood=Ind|VerbForm=Fin 0 root'
            source[split] = pd.DataFrame([sentence(f'{pronoun} {"work" if pronoun == "They" else "works"} at home every day.', specs)])
        # Duplicate a held-out source into training; held-out data wins.
        source['train'] = pd.concat([source['train'], source['test']], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp, patch('dala.multilingual.load_annotated_ud', return_value=source), patch('builtins.print'):
            outputs = create_english(tmp)
            self.assertTrue((Path(tmp) / 'dala_en_report.json').exists())
            self.assertEqual([len(v) for v in outputs.values()], [2, 2, 2])
            self.assertFalse(set(outputs['train'].text) & set(outputs['test'].text))


if __name__ == '__main__':
    unittest.main()
