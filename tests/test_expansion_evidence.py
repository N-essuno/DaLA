import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.mine_multilingual_evidence import prepare


class EvidenceBoundaryTests(unittest.TestCase):
    def test_declared_held_out_split_cannot_be_mined(self):
        with self.assertRaisesRegex(ValueError,'training'):
            prepare({'split':'test'},Path('/unused'))

    def test_held_out_path_cannot_be_disguised_as_training(self):
        with self.assertRaisesRegex(ValueError,'Held-out'):
            prepare({'split':'train','files':{'corpus/dev/sentence.m2':'x'}},Path('/unused'))

    def test_evidence_bytes_must_match_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.m2';path.write_text('S Example .\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                prepare({'split':'train','files':{str(path):'wrong'}},Path(directory)/'output')

    def test_checker_style_and_alternatives_stay_out(self):
        try:import lxml
        except ImportError:self.skipTest('Optional lxml preparation dependency unavailable')
        from scripts.extract_rule_examples import extract
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'grammar.xml'
            path.write_text('''<rules lang="xx"><category id="G" type="grammar"><rule id="ok"><example correction="runs">She <marker>run</marker>.</example><example correction="runs|ran">She <marker>run</marker>.</example></rule></category><category id="S" type="style"><rule id="style"><example correction="excellent">A <marker>good</marker> result.</example></rule></category></rules>''')
            rows,counts,deps=extract(path,'xx','xx',root)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['correct'],'She runs.')
            self.assertEqual(rows[0]['incorrect'],'She run.')
            self.assertEqual(rows[0]['rule_id'],'G/ok')
            with self.assertRaisesRegex(ValueError,'language mismatch'):extract(path,'xx','yy',root)



class ObservedExportTests(unittest.TestCase):
    def test_capitalized_observed_mapping_roundtrips_without_inflection_pairs(self):
        from dala.edits import Corruption
        from dala.pair_pipeline import export_dataset
        from dala.validate_dataset import validate
        edit=Corruption('de_observed','spelling',0,10,'Kenntnisse','Kentnisse',(0,)).record()
        edit.update(corrupted_start=0,corrupted_end=9,requires_lexical_screen=True)
        pair=dict(pair_id='pair',document_id='doc',split='train',source_name='fixture',source_dataset='fixture',source_revision='fixture',document_sha256='b'*64,url='https://example.org/fixture',license='fixture',language='de',original='Kenntnisse helfen.',corrupted='Kentnisse helfen.',edits=[edit],quality_status='candidate_rule_and_dictionary_screened',checker={'lexical_checks':[dict(rule_id='de_observed',original_recognized=True,replacement_recognized=False,corrupted_start=0,corrupted_end=9)]})
        book=dict(language='de',edit_validator='dala.language_packs.morphology:permits',rules=[dict(id='de_observed',operator='observed_nonword',family='spelling',correct='Kenntnisse',incorrect='Kentnisse',mappings={'kenntnisse':['kentnisse']},requires_lexical_screen=True)])
        manifest=dict(language='de',seed=42,quality_status=pair['quality_status'],profile={'export_used_rule_mappings':True,'checker':{'mode':'morphology'}})
        document={k:pair[k] for k in ['document_id','source_dataset','source_revision','document_sha256','url','license']}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'dataset';export_dataset([pair],root,manifest,book,[document])
            self.assertTrue(validate(root)['correction_roundtrip'])
            self.assertEqual(json.loads((root/'rules.json').read_text())['rules'][0]['mappings'],{'kenntnisse':['kentnisse']})

if __name__=='__main__':unittest.main()
