import random
import unittest
import json
import gzip
import requests
from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from dala.batch_pipeline import verified_receipt, atomic_json
from dala.common_pile import sha256_file
from dala.common_pile import documents
from dala.language_check import LanguageCheck
from scripts.finalize_scale_dataset import attainable_quotas
from dala.curation import NearDuplicates
from dala.near_duplicates import IndexedNearDuplicates
from dala.pair_pipeline import split_for


class ScaleTests(unittest.TestCase):
    def test_checker_retries_transport_without_caching_failure(self):
        response=Mock();response.json.return_value={'software':{'version':'test'},'matches':[]}
        session=Mock();session.post.side_effect=[response,requests.ConnectionError('reset'),response]
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session), patch('dala.language_check.time.sleep'):
            with closing(LanguageCheck('http://localhost:12345',Path(directory)/'cache.sqlite')) as checker:
                self.assertEqual(checker.check('A sentence.','sv')['matches'],[])
                self.assertEqual(session.post.call_count,3)
                checker.check('A sentence.','sv')
                self.assertEqual(session.post.call_count,3)

    def test_checker_retry_exhaustion_remains_failure_and_uncached(self):
        response=Mock();response.json.return_value={'software':{'version':'test'},'matches':[]}
        session=Mock();session.post.side_effect=[response]+[requests.ConnectionError('reset') for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session), patch('dala.language_check.time.sleep'):
            with closing(LanguageCheck('http://localhost:12345',Path(directory)/'cache.sqlite')) as checker:
                with self.assertRaises(requests.ConnectionError):checker.check('A sentence.','sv')
                self.assertEqual(checker.db.execute('SELECT COUNT(*) FROM checks').fetchone()[0],0)
                self.assertEqual(session.post.call_count,5)

    def test_receipt_publication_preserves_previous_complete_version(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'receipt.json'
            p.write_text('{"version":1}')
            original_replace=Path.replace
            def before_publish(temporary,target):
                self.assertEqual(json.loads(p.read_text()),{'version':1})
                self.assertEqual(json.loads(temporary.read_text()),{'version':2})
                return original_replace(temporary,target)
            with patch.object(Path,'replace',before_publish):atomic_json(p,{'version':2})
            self.assertEqual(json.loads(p.read_text()),{'version':2})

    def test_checker_pool_rejects_remote_urls_and_mixed_versions(self):
        with self.assertRaisesRegex(ValueError, 'local'):
            LanguageCheck(['http://localhost:12345', 'https://example.org'])
        first=Mock();first.json.return_value={'software':{'version':'a'},'matches':[]}
        second=Mock();second.json.return_value={'software':{'version':'b'},'matches':[]}
        session=Mock();session.post.side_effect=[first,second]
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session):
            with self.assertRaisesRegex(ValueError,'versions differ'):
                LanguageCheck(['http://localhost:12345','http://localhost:23456'],Path(directory)/'cache.sqlite')

    def test_checker_pool_shares_versioned_cache(self):
        response=Mock();response.json.return_value={'software':{'version':'test'},'matches':[]}
        session=Mock();session.post.return_value=response
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session):
            path=Path(directory)/'cache.sqlite'
            with closing(LanguageCheck('http://localhost:12345',path)) as single:
                expected=single.check('A cached sentence.','nl')
            with closing(LanguageCheck(['http://localhost:23456','http://localhost:34567'],path)) as pool:
                session.post.reset_mock()
                self.assertEqual(pool.check('A cached sentence.','nl'),expected)
                session.post.assert_not_called()

    def test_attainable_quotas_do_not_oversample_scarce_splits(self):
        for available in ({'train':800,'validation':20,'test':150},
                          {'train':500,'validation':80,'test':31},
                          {'train':900,'validation':80,'test':190}):
            total,quota=attainable_quotas(available,1000)
            self.assertLessEqual(total,1000)
            self.assertEqual(sum(quota.values()),total)
            self.assertTrue(all(quota[s]<=available[s] for s in quota))
            self.assertLess(abs(quota['train']-.8*total),1)
            self.assertLess(abs(quota['validation']-.05*total),1)

    def test_expanded_source_filter_preserves_author_and_license_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.gz'
            source=dict(name='fixture',repo_id='fixture/data',revision='a'*40,
                        domains=['example.org'],source_values=['news'],license_contains='BY4',
                        min_document_chars=10,license_override='BY3',license_evidence_url='https://example.org/license')
            rows=[dict(id=i,source='news',text=text,metadata=dict(url=f'https://example.org/{i}',license='BY4',author='Writer'))
                  for i,text in enumerate(['short','An unchanged source paragraph.'])]
            with gzip.open(p,'wt') as writer:
                for row in rows:writer.write(json.dumps(row)+'\n')
            result=list(documents([(source,p,{'file':'source.gz'})]))
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['text'],rows[1]['text'])
            self.assertEqual(result[0]['author'],'Writer')
            self.assertEqual(result[0]['license'],'BY3')
            self.assertEqual(result[0]['captured_license'],'BY4')
            self.assertEqual(result[0]['license_evidence_url'],source['license_evidence_url'])

    def test_corrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'batch.jsonl';p.write_text('{}\n')
            self.assertIsNone(verified_receipt(p))
            receipt={'sha256':sha256_file(p),'pairs':1}
            p.with_suffix('.receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(verified_receipt(p),receipt)
            p.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                verified_receipt(p)

    def test_batched_cache_flushes_on_close_and_survives_reopen(self):
        response=Mock();response.json.return_value={'software':{'version':'test'},'matches':[]}
        session=Mock();session.post.return_value=response
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session):
            cache=Path(directory)/'cache.sqlite3'
            checker=LanguageCheck('http://localhost:12345',cache,commit_batch_size=128)
            checker.check('An uncached sentence.','en-US')
            with closing(sqlite3.connect(cache)) as reader:
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM checks').fetchone()[0],0)
            checker.close()
            checker=LanguageCheck('http://localhost:12345',cache,commit_batch_size=128)
            session.post.reset_mock()
            self.assertEqual(checker.check('An uncached sentence.','en-US')['matches'],[])
            session.post.assert_not_called()
            checker.close()

    def test_indexed_dedup_matches_original_on_adversarial_and_random_text(self):
        old,new=NearDuplicates(),IndexedNearDuplicates();rng=random.Random(41)
        texts=['a b c','a b c','a b c d','one two three four five six seven eight nine ten',
               'one two three four five six seven eight nine eleven',
               ' '.join(['a']*200+['x']), ' '.join(['y']+['a']*200)]
        vocab=[f'word{i}' for i in range(20)]
        for _ in range(500):
            words=rng.choices(vocab,k=rng.randint(6,48));texts.append(' '.join(words))
            for count in (1,2,5):
                changed=words.copy()
                for _ in range(count):
                    index=rng.randrange(len(changed));changed[index]=rng.choice(vocab)
                texts.append(' '.join(changed))
        for text in texts:self.assertEqual(old.add(text),new.add(text),text)

    def test_reference_split_proportions_keep_training_assignment(self):
        normal=[split_for(str(i)) for i in range(1000)]
        scaled=[split_for(str(i),proportions={'train_percent':80,'validation_percent':5}) for i in range(1000)]
        self.assertEqual([x=='train' for x in normal],[x=='train' for x in scaled])
        self.assertTrue(any(a=='validation' and b=='test' for a,b in zip(normal,scaled)))
        with self.assertRaises(ValueError):split_for('x',proportions={'train_percent':90,'validation_percent':20})
