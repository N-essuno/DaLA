import hashlib
import unittest

from scripts.merge_dutch_extension import select_extension
from dala.pair_pipeline import order_documents


class ExtensionSelectionTests(unittest.TestCase):
    def test_weighted_source_order_keeps_every_document_and_is_deterministic(self):
        docs = [dict(document_id=f'{s}-{i}', source_name=s) for s in ('a', 'b') for i in range(6)]
        profile = dict(source_sampling='round_robin', source_sampling_weights=dict(a=2, b=1))
        ordered = order_documents(docs, 42, profile)
        self.assertEqual(ordered, order_documents(list(reversed(docs)), 42, profile))
        self.assertEqual([d['source_name'] for d in ordered[:9]], list('aabaabaab'))
        self.assertEqual({d['document_id'] for d in ordered}, {d['document_id'] for d in docs})
        self.assertEqual(len(ordered), len(docs))
        uniform = dict(source_sampling='round_robin')
        self.assertEqual(order_documents(docs,42,uniform), order_documents(docs,42,dict(**uniform,source_sampling_weights=dict(a=1,b=1))))

    def test_preserves_base_excludes_flags_and_deduplicates_globally(self):
        def pair(i, text, split='train'):
            return dict(pair_id=i, document_id=i, original=text, corrupted=text+' fout', split=split)
        base = [pair('a', 'De rechtbank heeft deze zaak zorgvuldig beoordeeld.'),
                pair('b', 'Deze bekende fout moet uit de oorspronkelijke set verdwijnen.')]
        additions = [pair('c', base[0]['original']),
                     pair('d', 'In het park bloeien verschillende bloemen langs het wandelpad.'),
                     pair('e', 'Onderzoekers verzamelen informatie over de uitstoot van schadelijke stoffen.', 'test')]
        exclude = {hashlib.sha256(base[1]['original'].encode()).hexdigest()}
        selected, receipt = select_extension(base, additions, exclude, dict(train=2, test=1))
        self.assertEqual({p['pair_id'] for p in selected}, {'a', 'd', 'e'})
        self.assertIs(selected[0], base[0])
        self.assertEqual(receipt['retained_base_pairs'], 1)
        self.assertEqual(receipt['rejections']['base_review_exclusion'], 1)

    def test_insufficient_extension_fails_instead_of_duplicating(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_extension([], [], set(), dict(train=1))
