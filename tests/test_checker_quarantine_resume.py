import copy,gzip,json,tempfile,unittest
from pathlib import Path
from dala.common_pile import sha256_file
from scripts.resume_checker_quarantine import migrate

class QuarantineResumeTests(unittest.TestCase):
    def fixture(self,root):
        oldcode=root/'oldcode';newcode=root/'newcode';oldcode.mkdir();newcode.mkdir()
        (oldcode/'morphology_check.py').write_text('old');(newcode/'morphology_check.py').write_text('new')
        source=root/'old';source.mkdir()
        for name in ['exclusions','sources']:(root/name).write_text('{}')
        with gzip.open(root/'rules.gz','wt') as f:json.dump({'rules':[]},f)
        profile=dict(language='uk',rulebook=str(root/'rules.gz'),exclusions=str(root/'exclusions'),sources_config=str(root/'sources'),checker={},build={'checkpoint_dir':str(source),'parser_processes':1})
        oldpath=root/'old.json';oldpath.write_text(json.dumps(profile))
        new=copy.deepcopy(profile);new['checker']['source_failure_exclusions']={'a'*64:'reproduction receipt'};new['build'].update(checkpoint_dir=str(root/'new'),parser_processes=2)
        newpath=root/'new.json';newpath.write_text(json.dumps(new))
        parent={'mode':'generation_upgrade','reused_batch_signatures':{'0':'original-generator'},'backfill_source_batches':[0]}
        (source/'migration.json').write_text(json.dumps(parent));(source/'progress.json').write_text('{}')
        old=dict(profile=profile,profile_sha256=sha256_file(oldpath),code_sha256={'morphology_check.py':sha256_file(oldcode/'morphology_check.py')},checkpoint_lineage_sha256=sha256_file(source/'migration.json'),rulebook_sha256=sha256_file(root/'rules.gz'),source_exclusions_sha256=sha256_file(root/'exclusions'),sources_config_sha256=sha256_file(root/'sources'))
        (source/'run.json').write_text(json.dumps(old))
        for suffix in ['candidates','screened']:
            p=source/f'00000.{suffix}.jsonl';p.write_text('');p.with_suffix('.receipt.json').write_text(json.dumps({'sha256':sha256_file(p)}))
        return oldpath,newpath,oldcode,newcode,source

    def test_preserves_original_generator_and_backfill_order(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));r=migrate(*args[:4])
            self.assertEqual(r['reused_batch_signatures'],{'0':'original-generator'})
            self.assertEqual(r['backfill_source_batches'],[0])
            self.assertEqual(r['parent_migration_sha256'],sha256_file(args[4]/'migration.json'))

    def test_failure_policy_upgrade_preserves_old_quarantine(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));old=json.loads(args[0].read_text());old['checker']['source_failure_exclusions']={'a'*64:'reproduction receipt'};args[0].write_text(json.dumps(old))
            signature=json.loads((args[4]/'run.json').read_text());signature['profile']=old;signature['profile_sha256']=sha256_file(args[0]);(args[4]/'run.json').write_text(json.dumps(signature))
            new=json.loads(args[1].read_text());new['checker'].update(instances=16,workers=128,failure_policy={'mode':'reject_sentence','log_path':'failures.jsonl'});args[1].write_text(json.dumps(new))
            result=migrate(*args[:4],policy_upgrade=True)
            self.assertTrue(result['checker_failure_policy_upgrade'])
            self.assertEqual(result['source_failure_exclusions'],old['checker']['source_failure_exclusions'])
            self.assertEqual(result['reused_batch_signatures']['0'],'original-generator')

    def test_rejects_changed_language(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));p=json.loads(args[1].read_text());p['language']='de';args[1].write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'profile change'):migrate(*args[:4])

    def test_rejects_corrupt_parent_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));(args[4]/'migration.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'lineage changed'):migrate(*args[:4])

if __name__=='__main__':unittest.main()
