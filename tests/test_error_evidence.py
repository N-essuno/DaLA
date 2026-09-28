import tempfile
import unittest
from pathlib import Path

from dala.error_evidence import mine, read_m2


class EvidenceTests(unittest.TestCase):
    def test_multiple_errors_and_annotators_do_not_multiply_support(self):
        text = '''S A comercial book are here .
A 1 2|||R:SPELL|||commercial|||REQUIRED|||-NONE-|||0
A 3 4|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0
A 1 2|||R:SPELL|||commercial|||REQUIRED|||-NONE-|||1

S This comercial book is here .
A 1 2|||R:SPELL|||commercial|||REQUIRED|||-NONE-|||0
'''
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.m2', Path(directory) / 'b.m2'
            a.write_text(text)
            b.write_text(text)
            report = mine([a, b], min_support=2)
        self.assertEqual(len(report['patterns']), 1)
        p = report['patterns'][0]
        self.assertEqual(p['distinct_sentence_support'], 2)
        self.assertEqual(p['multi_error_sentence_support'], 1)
        self.assertTrue(p['single_word_spelling_candidate'])
        self.assertEqual((p['proposed_corruption_from'], p['proposed_corruption_to']), ('commercial', 'comercial'))

    def test_insertions_deletions_noops_and_eof(self):
        rows = list(read_m2('S I go .\nA 1 1|||M:ADV|||often|||REQUIRED|||-NONE-|||0\nA 2 3|||U:PUNCT||||||REQUIRED|||-NONE-|||0\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1'))
        self.assertEqual(len(rows[0][1]), 2)
        self.assertEqual(rows[0][1][1].replacement, '')

    def test_invalid_offsets(self):
        with self.assertRaises(ValueError):
            list(read_m2('S A book .\nA 8 9|||R:NOUN|||dog|||REQUIRED|||-NONE-|||0'))

    def test_identity_annotations_not_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.m2'
            path.write_text('S Hello .\nA 0 1|||R:SPELL|||Hello|||REQUIRED|||-NONE-|||0')
            self.assertEqual(mine([path], min_support=1)['patterns'], [])
