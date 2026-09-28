from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from dala.language_check import LanguageCheck
from scripts.screening_pool import PersistentScreeningPool, close_screening_pools


class ScreeningPoolTests(unittest.TestCase):
    def tearDown(self):
        close_screening_pools()

    def test_checker_sessions_bounded_across_many_batches(self):
        response=Mock();response.json.return_value={'software':{'version':'test'},'matches':[]}
        session=Mock();session.post.return_value=response
        with tempfile.TemporaryDirectory() as directory, patch('dala.language_check.requests.Session',return_value=session):
            with closing(LanguageCheck('http://localhost:12345',Path(directory)/'cache.sqlite')) as checker:
                for batch in range(100):
                    with PersistentScreeningPool(max_workers=4) as pool:
                        results=list(pool.map(lambda text:checker.check(text,'pl'),[f'{batch}: sentence {i}' for i in range(12)]))
                        self.assertTrue(all(r['matches']==[] for r in results))
                self.assertLessEqual(len(checker.sessions),5)
                self.assertEqual(checker.db.execute('SELECT COUNT(*) FROM checks').fetchone()[0],1200)

    def test_error_propagates_and_pool_can_screen_next_batch(self):
        def check(number):
            if number==2:raise ValueError('failed diagnostic')
            return number
        with self.assertRaisesRegex(ValueError,'failed diagnostic'):
            with PersistentScreeningPool(max_workers=3) as pool:list(pool.map(check,range(6)))
        with PersistentScreeningPool(max_workers=3) as pool:
            self.assertEqual(list(pool.map(check,[4,3,1])),[4,3,1])
