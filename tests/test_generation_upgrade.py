import copy,json,tempfile,unittest
from pathlib import Path
from dala.common_pile import sha256_file
from scripts.upgrade_campaign_checkpoints import normalized,upgrade

class UpgradeTests(unittest.TestCase):
    def fixture(self,root):
        oldcode=root/'oldcode';newcode=root/'newcode';oldcode.mkdir();newcode.mkdir()
        (oldcode/'parsing.py').write_text('old');(newcode/'parsing.py').write_text('new')
        source=root/'old';source.mkdir()
        for name in ['rules','exclusions','sources']:(root/name).write_text('{}')
        profile=dict(schema_version=1,language='xx',mode='pairs',parser='stanza',sources={},selection={},checker={'workers':8},
            rulebook=str(root/'rules'),exclusions=str(root/'exclusions'),sources_config=str(root/'sources'),build={'checkpoint_dir':str(source),'parser_processes':1})
        oldpath=root/'old.json';oldpath.write_text(json.dumps(profile))
        new=copy.deepcopy(profile);new['build'].update(checkpoint_dir=str(root/'new'),backfill_reused_batches=True,parser_processes=2);new['parser_recover_mwt']=True
        newpath=root/'new.json';newpath.write_text(json.dumps(new))
        record=dict(profile=profile,profile_sha256=sha256_file(oldpath),code_sha256={'parsing.py':sha256_file(oldcode/'parsing.py')},rulebook_sha256=sha256_file(root/'rules'),source_exclusions_sha256=sha256_file(root/'exclusions'),sources_config_sha256=sha256_file(root/'sources'))
        (source/'run.json').write_text(json.dumps(record))
        for suffix in ['candidates','screened']:
            file=source/f'00000.{suffix}.jsonl';file.write_text('{}\n')
            file.with_suffix('.receipt.json').write_text(json.dumps({'sha256':sha256_file(file)}))
        return oldpath,newpath,oldcode,newcode,source

    def test_explicit_lineage_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,new,base,current,source=self.fixture(root)
            receipt=upgrade(old,new,base,current)
            self.assertTrue(receipt['generation_behavior_changed'])
            self.assertEqual(receipt['backfill_source_batches'],[0])
            self.assertEqual(receipt['reused_candidate_batches'],1)
            self.assertEqual((source/'00000.candidates.jsonl').read_bytes(),(root/'new/00000.candidates.jsonl').read_bytes())
            signature=json.loads((root/'new/run.json').read_text())
            self.assertEqual(signature['checkpoint_lineage_sha256'],sha256_file(root/'new/migration.json'))

    def test_rejects_changed_language_and_rule_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,base,current,source=self.fixture(Path(temp))
            profile=json.loads(new.read_text());profile['language']='different';new.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError,'Non-approved'):upgrade(old,new,base,current)

    def test_rejects_unreviewed_code(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,base,current,source=self.fixture(Path(temp));(current/'rule_compiler.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Unreviewed'):upgrade(old,new,base,current)

    def test_rejects_corrupt_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new,base,current,source=self.fixture(Path(temp));(source/'00000.candidates.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):upgrade(old,new,base,current)
