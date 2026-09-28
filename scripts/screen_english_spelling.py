"""Recheck active lexical spelling pairs against local US/GB LanguageTool dictionaries.

Run: python -m scripts.screen_english_spelling
This screening is not a linguistic precision estimate.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from dala.language_check import local_server, LanguageCheck


def main():
    path = Path('config/english_productive_rules.json')
    book = json.loads(path.read_text())
    rules = [r for r in book['rules'] if r['family'] == 'spelling' and 'correct' in r]
    with local_server() as url:
        checker = LanguageCheck(url)
        def check(rule):
            checks = {}
            for dialect in ('en-US', 'en-GB'):
                flags = []
                for word in (rule['correct'], rule['incorrect']):
                    text = 'The word is ' + word + '.'
                    hits = checker.hard_matches(checker.check(text, dialect), text)
                    flags.append(any(h['issue_type'] == 'misspelling' and h['start'] == 12
                                     and h['end'] == 12 + len(word) for h in hits))
                checks[dialect] = dict(correct_flagged=flags[0], incorrect_flagged=flags[1])
            return dict(id=rule['id'], checks=checks,
                        passed=all(not c['correct_flagged'] and c['incorrect_flagged'] for c in checks.values()))
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(check, rules))
            report = dict(at=datetime.now(timezone.utc).isoformat(), software=checker.software,
                          rulebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          method='Exact token misspelling diagnostic in The word is WORD.; both en-US and en-GB; automatic evidence, not human precision',
                          rows=rows)
        finally:
            checker.close()
    Path('wiki/artifacts/english-spelling-screening.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Passed {sum(r['passed'] for r in rows)} of {len(rows)} mappings")
    if not all(r['passed'] for r in rows):
        raise SystemExit('Spelling screening failed; inspect receipt')


if __name__ == '__main__':
    main()
