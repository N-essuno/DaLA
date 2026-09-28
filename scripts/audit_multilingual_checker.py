"""CPU-only independent checker diagnostics; silence is not proof of correctness."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

import requests
from dala.language_check import LanguageCheck, local_server
from dala.common_pile import sha256_file


def audit(root,output,checker,supported,sample_size):
    language=json.loads((root/'manifest.json').read_text())['language']
    if language not in supported:
        return dict(language=language,status='unsupported_by_local_checker',sampled=0)
    pairs=[json.loads(line) for path in sorted(root.glob('*/pairs.jsonl')) for line in path.open()]
    pairs=sorted(pairs,key=lambda r:r['pair_id'])
    selected=random.Random(92642).sample(pairs,min(sample_size,len(pairs))) if sample_size else pairs
    rows=[]
    def check(pair):
        original=checker.hard_matches(checker.check(pair['original'],language),pair['original'])
        corrupted=checker.hard_matches(checker.check(pair['corrupted'],language),pair['corrupted'])
        near=[m for m in corrupted if any(m['start']<e['corrupted_end'] and m['end']>e['corrupted_start'] for e in pair['edits'])]
        return dict(pair_id=pair['pair_id'],language=language,original=pair['original'],corrupted=pair['corrupted'],edits=pair['edits'],
                    source_diagnostics=original,corrupted_diagnostics=corrupted,diagnostics_overlapping_edit=near)
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(check,selected))
    path=output/(language+'-diagnostics.jsonl')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return dict(language=language,status='automatic_diagnostic_not_linguistic_validation',pairs=len(pairs),sampled=len(rows),
                source_flagged=sum(bool(r['source_diagnostics']) for r in rows),
                corruption_flagged_at_edit=sum(bool(r['diagnostics_overlapping_edit']) for r in rows),
                source_rule_counts=dict(Counter(m['rule_id'] for r in rows for m in r['source_diagnostics'])),
                manifest_sha256=sha256_file(root/'manifest.json'),diagnostics_sha256=sha256_file(path))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--sample-size',type=int,default=50);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with local_server(heap='8g',threads=12,instance='european-audit') as url:
        response=requests.get(url+'/v2/languages',timeout=30);response.raise_for_status()
        languages=response.json();supported={x['longCode'] for x in languages}|{x['code'] for x in languages}
        checker=LanguageCheck(url,cache='la_output/cache/european-audit.sqlite3',dialects=['fr'],commit_batch_size=100)
        report=dict(software=checker.software,available_languages=languages,sample_size=a.sample_size,
                    qualification='Hard checker diagnostics are triage signals; absence is not precision, names can cause false alarms. No native review.',results={})
        try:
            for root in sorted(a.root.iterdir()):
                if not (root/'manifest.json').exists():continue
                row=audit(root,a.output,checker,supported,a.sample_size)
                report['results'][row['language']]=row
                (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(row['language'],row['status'],row.get('sampled'),row.get('source_flagged'),flush=True)
        finally:checker.close()


if __name__=='__main__':main()
