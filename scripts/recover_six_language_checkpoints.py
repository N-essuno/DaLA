"""Audit and reuse immutable checkpoints across the 2026-09-24 I/O-only fix.

Original directories/signatures remain untouched. Every copied batch has its
checksum verified. The replacement signature and exported migration receipt
explicitly retain the original code/input identity; no linguistic change is
permitted by this migration.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from dala.batch_pipeline import atomic_json, verified_receipt
from dala.common_pile import sha256_file
from dala.profiles import load_profile, resource

ARCHIVE=Path('wiki/artifacts/six-language-expansion/recovery-20260924/previous-code/dala')
RUNS={'pl':'scale_v2','sv':'scale_v1','nb':'scale_v1','nn':'scale_v1','fo':'scale_v1','is':'scale_v2'}


def recover(language, previous):
    oldroot=Path(f'la_output/{language}_{previous}_checkpoints')
    destination=Path(f'la_output/{language}_recovery_v1_checkpoints')
    if destination.exists():raise FileExistsError(destination)
    old=json.loads((oldroot/'run.json').read_text())
    archived={str(p.relative_to(ARCHIVE)):sha256_file(p) for p in ARCHIVE.rglob('*.py')}
    current={str(p.relative_to(Path('dala'))):sha256_file(p) for p in Path('dala').rglob('*.py')}
    if old['code_sha256']!=archived:raise ValueError(f'{language}: original code differs from archived code')
    changed={k for k in set(archived)|set(current) if archived.get(k)!=current.get(k)}
    if changed!={'batch_pipeline.py','language_check.py'}:raise ValueError(f'Unexpected code changes: {changed}')
    oldprofile=load_profile(f'{language}_{previous}')
    for key,field in [('rulebook','rulebook_sha256'),('exclusions','source_exclusions_sha256'),('sources_config','sources_config_sha256')]:
        if sha256_file(resource(oldprofile,key))!=old[field]:raise ValueError(f'{language}: changed linguistic input {key}')
    profile=json.loads(Path(oldprofile['_path']).read_text())
    profile['build']['checkpoint_dir']=str(destination)
    profilepath=Path(f'config/languages/{language}_recovery_v1.json')
    if profilepath.exists():raise FileExistsError(profilepath)
    atomic_json(profilepath,profile)
    loaded=load_profile(profilepath)
    replacement={**old,'profile':{k:v for k,v in loaded.items() if k!='_path'},
                 'profile_sha256':sha256_file(profilepath),'code_sha256':current}
    # Only execution location is allowed to differ in the profile.
    comparison=json.loads(json.dumps(replacement['profile']))
    comparison['build']['checkpoint_dir']=old['profile']['build']['checkpoint_dir']
    if comparison!=old['profile']:raise ValueError('Unexpected profile change')
    destination.mkdir()
    reused=[];uncommitted=[]
    for path in sorted(oldroot.glob('*.candidates.jsonl')):
        try:receipt=verified_receipt(path)
        except json.JSONDecodeError:receipt=None
        if receipt is None:
            uncommitted.append(path.name);continue
        for candidate in [path,path.with_name(path.name.replace('.candidates.','.screened.'))]:
            try:record=verified_receipt(candidate)
            except json.JSONDecodeError:record=None
            if record is None:
                if candidate.exists():uncommitted.append(candidate.name)
                continue
            for source in [candidate,candidate.with_suffix('.receipt.json')]:
                os.link(source,destination/source.name)
            reused.append({'file':candidate.name,'sha256':record['sha256'],
                           'receipt_sha256':sha256_file(candidate.with_suffix('.receipt.json'))})
    atomic_json(destination/'reused-batches.json',reused)
    shutil.copy2(oldroot/'run.json',destination/'migration-original-run.json')
    atomic_json(destination/'run.json',replacement)
    migration=dict(at=datetime.now(timezone.utc).isoformat(),language=language,
        reason='Atomic receipt publication and future synchronization; bounded retry of identical checker HTTP requests',
        linguistic_inputs_changed=False,source_checkpoint=str(oldroot),
        original_run_sha256=sha256_file(oldroot/'run.json'),original_code_sha256=archived,
        current_code_sha256=current,changed_code=sorted(changed),archived_code=str(ARCHIVE),
        reused_candidate_batches=sum('.candidates.' in r['file'] for r in reused),
        reused_screened_batches=sum('.screened.' in r['file'] for r in reused),
        reused_inventory_sha256=sha256_file(destination/'reused-batches.json'),uncommitted_to_regenerate=uncommitted)
    atomic_json(destination/'migration.json',migration)
    print(language,migration['reused_candidate_batches'],migration['reused_screened_batches'],'verified batches reused',flush=True)


if __name__=='__main__':
    for language,run in RUNS.items():recover(language,run)
