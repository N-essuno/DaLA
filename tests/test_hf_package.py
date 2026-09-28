import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('hf_package_runtime', Path(__file__).parents[1] / 'scripts/hf_package_runtime.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class PackageTests(unittest.TestCase):
    def test_recreation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir(); (root / 'provenance').mkdir(); (root / 'data').mkdir()
            config = dict(seed=4242, shard_rows=1, sample_source_judgments={},
                          prompts=dict(acceptability='Correct?', correction='Correct it.'))
            (root / 'metadata/config.json').write_text(json.dumps(config))
            docs = []
            for split in runtime.SPLITS:
                doc = dict(document_id=split, url='https://example.org/' + split, license='CC0', author=None, source_revision='a'*40, document_sha256='b'*64)
                docs.append(doc)
                pair = dict(doc, pair_id=split, split=split, original='They work.', corrupted='They works.',
                            source_name='fixture', quality_status='test',
                            edits=[dict(original='work', replacement='works', start=5, end=9,
                                        corrupted_start=5, corrupted_end=10, corruption_type='agreement')])
                with runtime.compressed_writer(root / 'provenance' / f'{split}.pairs.jsonl.gz') as f:
                    f.write(runtime.encode(pair))
            with runtime.compressed_writer(root / 'provenance/documents.jsonl.gz') as f:
                for doc in docs:f.write(runtime.encode(doc))
            shards=[]
            for task in runtime.TASKS:
                original=runtime.build_data(root,root/'data'/task,task)
                with tempfile.TemporaryDirectory() as dest:
                    rebuilt=runtime.build_data(root,Path(dest)/task,task)
                    self.assertEqual(original,rebuilt)
                shards.extend(original)
            manifest=dict(shards=shards,splits={s:dict(pairs=1) for s in runtime.SPLITS},
                          files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=runtime.digest(p)) for p in root.rglob('*') if p.is_file()})
            (root/'metadata/manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(runtime.validate(root)['rows_per_task'],6)
            damaged=root/shards[0]['path']
            damaged.write_bytes(damaged.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'Checksum/size mismatch'):
                runtime.validate(root)


if __name__ == '__main__':unittest.main()
