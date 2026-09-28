"""Validate selected independent pilots and record honest coverage/split totals."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from dala.common_pile import sha256_file
from dala.validate_dataset import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--coverage',required=True,type=Path);a=p.parse_args()
    coverage=json.loads((a.coverage/'coverage.json').read_text());results={};validations={}
    for root in a.directories:
        manifest=json.loads((root/'manifest.json').read_text());language=manifest['language']
        if language in results:raise ValueError('Only one selected dataset per language')
        if manifest.get('inputs_changed_during_build') or manifest.get('code_changed_during_build'):raise ValueError('Unstable build provenance')
        validations[language]=validate(root)
        observed=coverage[language]
        if Path(observed['dataset']).resolve()!=root.resolve():raise ValueError('Coverage points to a different dataset')
        book=json.loads((root/'rules.json').read_text())
        configured={r['family'] for r in book['rules'] if r['family']!='spelling'}
        selected=set(observed['active_grammar_families'])
        results[language]=dict(dataset=str(root.resolve()),profile=manifest['profile'],
            manifest_sha256=sha256_file(root/'manifest.json'),pairs=manifest['verification']['pairs'],
            grammar=observed['grammatical_pairs'],spelling=observed['spelling_pairs'],
            splits={split:record['pairs'] for split,record in manifest['splits'].items()},
            selected_grammar_families=sorted(selected),configured_grammar_families=sorted(configured),
            missing_grammar_families=sorted(configured-selected),source_documents=manifest['counts']['documents'],
            independent_original_checker=manifest['profile']['checker'].get('source_languagetool',False),
            observed_spelling_rules_selected=[rid for rid in observed['selected_rules'] if '_observed_spelling_' in rid],
            quality_status=manifest['quality_status'],native_validation=False)
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),compute='CPU only; Stanza use_gpu=False',
                total_pairs=sum(r['pairs'] for r in results.values()),
                total_grammar=sum(r['grammar'] for r in results.values()),
                total_spelling=sum(r['spelling'] for r in results.values()),
                total_splits={s:sum(r['splits'][s] for r in results.values()) for s in ['train','validation','test']},
                all_splits_nonempty=all(all(r['splits'].values()) for r in results.values()),languages=results,
                limitation='Candidate pilots; automated validation, checker diagnostics and agent inspection do not establish native linguistic precision. Per-family coverage gaps remain.')
    a.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('assessment.json',report),('validation.json',validations),('current.json',dict(datasets={l:r['dataset'] for l,r in results.items()},assessment='assessment.json'))]:
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='languages'},indent=2))


if __name__=='__main__':main()
