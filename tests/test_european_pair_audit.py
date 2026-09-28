"""Infrastructure checks; these do not validate linguistic judgment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_european_pairs import Database,FIELDS,request,validate

class PairAuditTests(unittest.TestCase):
    def test_uncertain_never_passes(self):
        for field in FIELDS:
            row={k:'yes' for k in FIELDS};row.update(reason='reason');row[field]='uncertain'
            self.assertEqual(validate(row)['decision'],'review')
        with self.assertRaises(ValueError):validate(dict(reason='missing labels'))

    def test_language_and_no_generator_anchoring(self):
        row=dict(language='pt-PT',original='a',corrupted='b',edits=[{'rule_id':'claim'}])
        data=json.loads(request(row)['messages'][1]['content'])
        self.assertIn('Portugal',data['language'])
        self.assertNotIn('edits',data)

    def test_resume_retry_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'jobs.sqlite'
            source=dict(component='de:test',sha256='abc')
            db=Database(path);self.assertEqual(db.register(source),(-1,0))
            db.put(source,[(0,{'id':'pair0'},[])])
            endpoint,key,raw,attempt,owner=db.claim(1,['test'],0)[0]
            db.finish(key,owner,attempt,None,'temporary')
            self.assertEqual(db.pending(),1)
            endpoint,key,raw,attempt,owner=db.claim(1,['test'],0)[0]
            self.assertEqual(attempt,2)
            db.finish(key,owner,attempt,{'decision':'review'},None)
            db.close();db=Database(path)
            self.assertEqual(db.register(source),(0,0))
            self.assertEqual(db.unfinished(),0)
            with self.assertRaises(ValueError):db.register(dict(source,sha256='changed'))
            db.close()

if __name__=='__main__':unittest.main()
