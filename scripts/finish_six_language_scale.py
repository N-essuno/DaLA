"""After candidate builds finish, validate, isolate and report; never upload.

This is an automatic artifact/coverage audit, not human linguistic validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from dala.validate_dataset import validate
from dala.profiles import load_profile,resource
from dala.common_pile import sha256_file

ORDER=['nn','fo','is','pl','sv','nb']
MINIMUM={'nn':7,'fo':10,'is':10,'pl':8,'sv':7,'nb':7}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='scale_v1')
    parser.add_argument('--language-runs',nargs='*',default=[],help='Per-language replacement runs, e.g. pl=scale_v2')
    parser.add_argument('--review-exclusions',type=Path)
    parser.add_argument('--pair-caps',nargs='*',default=[])
    a=parser.parse_args()
    runs={l:a.run_id for l in ORDER}
    for item in a.language_runs:
        language,run=item.split('=',1)
        if language not in runs:raise ValueError(language)
        runs[language]=run
    status_paths={l:Path('wiki/artifacts/six-language-expansion')/run/f'{l}-status.json' for l,run in runs.items()}
    root=Path('wiki/artifacts/six-language-expansion')/a.run_id
    status=root/'finalization.json'
    def write(state,**fields):status.write_text(json.dumps(dict(status=state,run_id=a.run_id,language_runs=runs,pair_caps=a.pair_caps,human_linguistic_validation=False,uploaded=False,**fields),indent=2)+'\n')
    write('waiting_for_builds')
    try:
        while True:
            records={l:json.loads(p.read_text()) for l,p in status_paths.items() if p.exists()}
            failed={l:r for l,r in records.items() if r['exit_code']}
            if failed:raise RuntimeError('Build failed: '+json.dumps(failed))
            if len(records)==len(ORDER):break
            time.sleep(30)
        write('validating_builds');checks={}
        current_code={str(p.relative_to(Path('dala'))):sha256_file(p) for p in sorted(Path('dala').rglob('*.py'))}
        for l in ORDER:
            directory=Path(records[l]['output']);manifest=json.loads((directory/'manifest.json').read_text())
            if manifest['code_sha256']!=current_code:raise ValueError(f'{l}: code changed since generation')
            worker_pool_hash=manifest.get('profile',{}).get('build',{}).get('worker_pool_sha256')
            if worker_pool_hash and sha256_file('scripts/worker_pool.py')!=worker_pool_hash:
                raise ValueError(f'{l}: pinned worker-pool implementation changed')
            screening_hash=manifest.get('profile',{}).get('build',{}).get('screening_pool_sha256')
            if screening_hash and sha256_file('scripts/screening_pool.py')!=screening_hash:
                raise ValueError(f'{l}: pinned screening-pool implementation changed')
            profile=load_profile(f'{l}_{runs[l]}')
            for key,field in [('rulebook','rulebook_sha256'),('exclusions','source_exclusions_sha256'),('sources_config','sources_config_sha256')]:
                if sha256_file(resource(profile,key))!=manifest[field]:raise ValueError(f'{l}: input changed: {key}')
            checks[l]=validate(directory)
            (root/f'{l}-automatic-validation.json').write_text(json.dumps(checks[l],indent=2)+'\n')
        destination=Path('la_output')/('six_language_candidates_'+a.run_id)
        write('isolating_datasets',input_checks=checks)
        isolation=[sys.executable,'-m','scripts.isolate_multilingual_datasets',*[records[l]['output'] for l in ORDER],
                   '--output-root',str(destination)]
        if a.review_exclusions:isolation.extend(['--review-exclusions',str(a.review_exclusions)])
        if a.pair_caps:isolation.extend(['--pair-caps',*a.pair_caps])
        subprocess.run(isolation,check=True)
        outputs=[str(destination/l) for l in ORDER]
        subprocess.run([sys.executable,'-m','scripts.report_six_language_coverage',*outputs,'--output',str(root/'final-audit')],check=True)
        coverage=json.loads((root/'final-audit/coverage.json').read_text())
        for l in ORDER:
            validate(destination/l)
            r=coverage[l]
            if len(r['active_grammar_families'])<MINIMUM[l] or r['spelling_pairs']==0:
                raise ValueError(f'{l}: insufficient realized grammar/spelling breadth; inspect final-audit/coverage.json')
        write('candidate_datasets_complete_require_linguistic_review',outputs={l:str(destination/l) for l in ORDER},
              pairs={l:coverage[l]['pairs'] for l in ORDER},coverage=str(root/'final-audit/coverage.json'),
              review_samples=str(root/'final-audit'),cross_dataset_isolation=str(destination/'isolation.json'))
    except Exception as error:
        write('failed_requires_investigation',error=str(error));raise


if __name__=='__main__':main()
