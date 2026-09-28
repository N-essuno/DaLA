"""Independent lexical screening; explicitly not an independent grammar checker."""
from importlib.metadata import version
from contextlib import ExitStack
from pathlib import Path
from .pair_pipeline import load_adapter


class MorphologyCheck:
    quality_status = 'candidate_rule_and_dictionary_screened'

    def __init__(self, profile):
        self.adapter = load_adapter(profile)
        self.software = {'name':'Hunspell dictionary via spylls', 'version':version('spylls'),
                         'language':profile['language'], 'independent_grammar_checker':False}
        self.stack = ExitStack()
        self.source_checker = None
        if profile['checker'].get('source_languagetool'):
            from .language_check import LanguageCheck, local_server
            url = self.stack.enter_context(local_server(directory=profile['checker'].get('tools_dir','la_output/tools'),
                                                       instance='six-'+profile['language']))
            self.source_checker = LanguageCheck(url, Path('la_output/cache') / ('source-'+profile['language']+'.sqlite3'),
                                               dialects=[profile['language']], commit_batch_size=100)
            self.stack.callback(self.source_checker.close)
            self.software['original_sentence_checker'] = self.source_checker.software
            self.software['corrupted_grammar_independently_verified'] = False

    def screen(self, original, corrupted, edits, named_spans=()):
        if self.source_checker:
            matches = self.source_checker._hard_matches(self.source_checker.check(original,self.adapter.language),original)
            matches = [m for m in matches if not (m['issue_type']=='misspelling'
                       and any(a<=m['start'] and m['end']<=b for a,b in named_spans))]
            if matches: return None, 'independent_source_checker_error'
        evidence = []
        for edit in edits:
            clean = self.adapter.recognized(edit['original'].casefold())
            bad = self.adapter.recognized(edit['replacement'].casefold())
            if not clean: return None, 'dictionary_unrecognized_original'
            if edit['corruption_type'] == 'spelling' and bad: return None, 'dictionary_recognizes_replacement'
            evidence.append(dict(rule_id=edit['rule_id'],original_recognized=clean,replacement_recognized=bad,
                                 corrupted_start=edit['corrupted_start'],corrupted_end=edit['corrupted_end']))
        return dict(language=self.adapter.language, lexical_checks=evidence,
                    grammatical_validity='generator_constraint_only_requires_audit'), None

    def close(self): self.stack.close()
