import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dala.common_pile import sha256_file
from scripts.migrate_execution_profile import migrate

class ExecutionResizeTests(unittest.TestCase):
    def test_resize_preserves_lineage_and_recovery_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'old';source.mkdir()
            old=dict(language='uk',build={'checkpoint_dir':str(source),'parser_processes':24},checker={'workers':8,'instances':1})
            for key in ['rulebook','exclusions','sources_config']:
                p=root/key;p.write_text('{}');old[key]=str(p)
            oldpath=root/'old.json';oldpath.write_text(json.dumps(old))
            new=copy.deepcopy(old);new['build'].update(checkpoint_dir=str(root/'new'),parser_processes=72);new['checker'].update(workers=24,instances=3)
            newpath=root/'new.json';newpath.write_text(json.dumps(new))
            signature=dict(profile=old,profile_sha256=sha256_file(oldpath),code_sha256={str(p.relative_to(Path('dala'))):sha256_file(p) for p in Path('dala').rglob('*.py')},rulebook_sha256=sha256_file(root/'rulebook'),source_exclusions_sha256=sha256_file(root/'exclusions'),sources_config_sha256=sha256_file(root/'sources_config'))
            parent=dict(mode='generation_upgrade',backfill_source_batches=[0],reused_batch_signatures={'0':'legacy-signature'})
            (source/'migration.json').write_text(json.dumps(parent));signature['checkpoint_lineage_sha256']=sha256_file(source/'migration.json')
            (source/'run.json').write_text(json.dumps(signature));(source/'progress.json').write_text('{"pairs":10}')
            for i in [0,1]:
                p=source/f'{i:05d}.candidates.jsonl';p.write_text('[]\n');p.with_suffix('.receipt.json').write_text(json.dumps({'sha256':sha256_file(p)}))
            profiles={str(oldpath):dict(old,_path=str(oldpath)),str(newpath):dict(new,_path=str(newpath))}
            with patch('scripts.migrate_execution_profile.load_profile',side_effect=lambda p:profiles[str(p)]):migrate(str(oldpath),str(newpath))
            receipt=json.loads((root/'new/migration.json').read_text());result=json.loads((root/'new/run.json').read_text())
            self.assertEqual(receipt['backfill_source_batches'],[0])
            self.assertEqual(receipt['reused_batch_signatures']['0'],'legacy-signature')
            self.assertEqual(receipt['reused_batch_signatures']['1'],sha256_file(source/'run.json'))
            self.assertEqual(result['checkpoint_lineage_sha256'],sha256_file(root/'new/migration.json'))
            self.assertEqual(receipt['prior_migration'],parent)
