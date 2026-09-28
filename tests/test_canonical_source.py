import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dala.canonical_source import snapshots, documents
from dala.pair_pipeline import order_documents


class CanonicalSourceTests(unittest.TestCase):
    def fixture(self, root, language='fr'):
        row=dict(language=language, source_document_id='article-1', text='Une phrase.', url='https://example.org/1')
        data=root/'data.jsonl';data.write_text(json.dumps(row)+'\n')
        source=dict(name='fixture',dataset='fixture:fr',language='fr',revision='v1',license='fixture',path=str(data),sha256=hashlib.sha256(data.read_bytes()).hexdigest())
        config=root/'sources.json';config.write_text(json.dumps(dict(sources=[source])))
        return config, data

    def test_identity_survives_snapshot_updates_and_text_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            config,data=self.fixture(Path(folder))
            first=list(documents(snapshots(config)))[0]
            settings=json.loads(config.read_text());settings['sources'][0]['revision']='v2';config.write_text(json.dumps(settings))
            second=list(documents(snapshots(config)))[0]
            self.assertEqual(first['document_id'],second['document_id'])
            self.assertEqual(first['text'],'Une phrase.')
            self.assertEqual(first['document_sha256'],hashlib.sha256(first['text'].encode()).hexdigest())

    def test_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            config,data=self.fixture(Path(folder));data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'): snapshots(config)

    def test_manifest_and_profile_language_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            config,data=self.fixture(Path(folder),'de')
            with self.assertRaisesRegex(ValueError,'language'):list(documents(snapshots(config)))
        with self.assertRaisesRegex(ValueError,'language'):
            order_documents([dict(document_id='x',language='de')],42,dict(language='fr'))

    def test_duplicate_documents_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            config,data=self.fixture(Path(folder));data.write_text(data.read_text()*2)
            source=json.loads(config.read_text())['sources'][0]
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                list(documents([(source,{'data':data},{})]))


if __name__=='__main__':unittest.main()
