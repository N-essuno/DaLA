import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from contextlib import closing
import requests
from dala.language_check import LanguageCheck,SentenceCheckFailure,CheckerUnavailable
from dala.morphology_check import MorphologyCheck

SOFTWARE={'version':'test'}
def response(status=200,incomplete=False):
    r=Mock();r.status_code=status;r.text='checker failure'
    r.json.return_value={'software':SOFTWARE,'matches':[],'warnings':{'incompleteResults':incomplete}}
    if status>=400:r.raise_for_status.side_effect=requests.HTTPError('HTTP error',response=r)
    return r

class ResilienceTests(unittest.TestCase):
    def run_client(self,folder,sequence,**limits):
        session=Mock();session.post.side_effect=[response()]+sequence
        sessions=patch('dala.language_check.requests.Session',return_value=session);sessions.start();self.addCleanup(sessions.stop)
        sleep=patch('dala.language_check.time.sleep');sleep.start();self.addCleanup(sleep.stop)
        checker=LanguageCheck('http://localhost:1234',Path(folder)/'cache.db',failure_policy=dict(mode='reject_sentence',log_path=str(Path(folder)/'failures.jsonl'),**limits))
        self.addCleanup(checker.close)
        return checker,session
    def test_transient_error_retries_without_caching_failure(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[response(500),response()]);self.assertEqual(c.check('sentence','uk')['matches'],[])
            self.assertEqual(c.db.execute('SELECT COUNT(*) FROM checks').fetchone()[0],1)
            self.assertFalse((Path(d)/'failures.jsonl').exists())
    def test_persistent_sentence_failure_rejected_logged_and_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[response(500)]*3+[response(),response()])
            with self.assertRaises(SentenceCheckFailure):c.check('bad sentence','uk')
            self.assertEqual(c.db.execute('SELECT COUNT(*) FROM checks').fetchone()[0],0)
            row=json.loads((Path(d)/'failures.jsonl').read_text());self.assertEqual(row['original'],'bad sentence');self.assertEqual(row['disposition'],'sentence_rejected')
            self.assertEqual(c.check('good sentence','uk')['matches'],[])
    def test_outage_stops_instead_of_discarding_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[requests.ConnectionError('offline')]*4)
            with self.assertRaises(CheckerUnavailable):c.check('sentence','uk')
            calls=s.post.call_count
            with self.assertRaises(CheckerUnavailable):c.check('another','uk')
            self.assertEqual(calls,s.post.call_count)
    def test_incomplete_results_are_never_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[response(incomplete=True)]*3+[response()])
            with self.assertRaises(SentenceCheckFailure):c.check('sentence','uk')
            self.assertEqual(c.db.execute('SELECT COUNT(*) FROM checks').fetchone()[0],0)
    def test_failure_budget_stops_repeated_sentence_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[response(500),response(),response(500),response()],attempts=1,max_consecutive=2)
            with self.assertRaises(SentenceCheckFailure):c.check('bad1','uk')
            with self.assertRaises(CheckerUnavailable):c.check('bad2','uk')
    def test_configuration_errors_still_raise(self):
        with tempfile.TemporaryDirectory() as d:
            c,s=self.run_client(d,[response(400)])
            with self.assertRaises(requests.HTTPError):c.check('sentence','uk')
    def test_screening_rejects_only_typed_sentence_failure(self):
        c=MorphologyCheck.__new__(MorphologyCheck);c.adapter=Mock(language='uk');c.source_checker=Mock();c.source_checker.check.side_effect=SentenceCheckFailure()
        self.assertEqual(c.screen('original','corrupted',[]),(None,'source_checker_sentence_failure'))
        c.source_checker.check.side_effect=CheckerUnavailable()
        with self.assertRaises(CheckerUnavailable):c.screen('original','corrupted',[])
