import unittest
from contextlib import nullcontext
from unittest.mock import patch, MagicMock
from dala.morphology_check import MorphologyCheck

class CheckerPoolTests(unittest.TestCase):
    def test_opt_in_pool_reuses_existing_client(self):
        with patch('dala.morphology_check.load_adapter'), patch('dala.language_check.local_servers') as servers, patch('dala.language_check.LanguageCheck') as client:
            urls=['http://localhost:1234','http://localhost:1235']
            servers.return_value=nullcontext(urls)
            profile={'language':'fr','checker':{'source_languagetool':True,'instances':2,'workers':16,'cache_path':'/tmp/diagnostic.sqlite3'}}
            checker=MorphologyCheck(profile)
            self.assertEqual(client.call_args.args[0],urls)
            self.assertEqual(str(client.call_args.args[1]),'/tmp/diagnostic.sqlite3')
            self.assertEqual(servers.call_args.kwargs['instances'],2)
            checker.close();client.return_value.close.assert_called_once()

    def test_default_single_server_behavior_preserved(self):
        with patch('dala.morphology_check.load_adapter'), patch('dala.language_check.local_server') as server, patch('dala.language_check.local_servers') as servers, patch('dala.language_check.LanguageCheck') as client:
            server.return_value=nullcontext('http://localhost:1234')
            checker=MorphologyCheck({'language':'fr','checker':{'source_languagetool':True}})
            self.assertEqual(client.call_args.args[0],'http://localhost:1234')
            servers.assert_not_called();checker.close()
