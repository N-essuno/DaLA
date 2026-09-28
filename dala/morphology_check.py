"""Independent lexical screening; explicitly not an independent grammar checker."""
from importlib.metadata import version
from contextlib import ExitStack
from pathlib import Path
import hashlib
from .pair_pipeline import load_adapter


class MorphologyCheck:
    quality_status = 'candidate_rule_and_dictionary_screened'

    def __init__(self, profile):
        self.adapter = load_adapter(profile)
        # Only explicitly reproduced, documented sentence failures may be quarantined.
        self.source_failure_exclusions = profile['checker'].get('source_failure_exclusions', {})
        if any(len(k) != 64 or not v for k, v in self.source_failure_exclusions.items()):
            raise ValueError('Checker failure exclusions require SHA256 and evidence')
        self.software = {'name':'Hunspell dictionary via spylls', 'version':version('spylls'),
                         'language':profile['language'], 'independent_grammar_checker':False}
        if profile.get('lexical_backend', {}).get('backend') == 'voikko':
            self.software.update(name='Voikko', version='resource-pinned', language=profile['language'])
        self.stack = ExitStack()
        self.source_checker = None
        if profile['checker'].get('source_languagetool'):
            from .language_check import LanguageCheck, local_server, local_servers
            instances = profile['checker'].get('instances', 1)
            if not isinstance(instances, int) or instances < 1:
                raise ValueError('Positive integer checker instances required')
            if instances == 1:
                servers = local_server(directory=profile['checker'].get('tools_dir','la_output/tools'),
                                       instance='six-'+profile['language'])
            else:
                servers = local_servers(directory=profile['checker'].get('tools_dir','la_output/tools'),
                                        instances=instances, threads=profile['checker'].get('workers', 8))
            url = self.stack.enter_context(servers)
            self.source_checker = LanguageCheck(url, Path(profile['checker'].get('cache_path', Path('la_output/cache') / ('source-'+profile['language']+'.sqlite3'))),
                                               dialects=[profile['language']], commit_batch_size=100,
                                               failure_policy=profile['checker'].get('failure_policy'))
            self.stack.callback(self.source_checker.close)
            self.software['original_sentence_checker'] = self.source_checker.software
            self.software['corrupted_grammar_independently_verified'] = False

    def screen(self, original, corrupted, edits, named_spans=()):
        if hashlib.sha256(original.encode()).hexdigest() in getattr(self, 'source_failure_exclusions', {}):
            return None, 'documented_source_checker_failure'
        if self.source_checker:
            from .language_check import SentenceCheckFailure
            try:
                response=self.source_checker.check(original,self.adapter.language)
            except SentenceCheckFailure:
                return None, 'source_checker_sentence_failure'
            matches = self.source_checker._hard_matches(response,original)
            matches = [m for m in matches if not (m['issue_type']=='misspelling'
                       and any(a<=m['start'] and m['end']<=b for a,b in named_spans))]
            if matches: return None, 'independent_source_checker_error'
        evidence = []
        for edit in edits:
            clean = self.adapter.recognized(edit['original'].lower())
            bad = self.adapter.recognized(edit['replacement'].lower())
            if not clean: return None, 'dictionary_unrecognized_original'
            if edit['corruption_type'] == 'spelling' and bad: return None, 'dictionary_recognizes_replacement'
            evidence.append(dict(rule_id=edit['rule_id'],original_recognized=clean,replacement_recognized=bad,
                                 corrupted_start=edit['corrupted_start'],corrupted_end=edit['corrupted_end']))
        return dict(language=self.adapter.language, lexical_checks=evidence,
                    grammatical_validity='generator_constraint_only_requires_audit'), None

    def close(self): self.stack.close()
