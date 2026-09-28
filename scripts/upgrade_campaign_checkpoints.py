"""Preserve verified legacy batches under an explicit generation-version lineage.

This is not an equivalence migration: old batches retain their original generator
signature and new batches use the upgraded generator. The same language, models,
rules, source bytes, ordering and batch boundaries are mandatory.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from dala.batch_pipeline import atomic_json, verified_receipt
from dala.common_pile import sha256_file
from dala.profiles import load_profile, resource

ALLOWED_CODE = {'parsing.py','surface_syntax.py','source_screen.py','language_packs/morphology.py','morphology_check.py','batch_pipeline.py'}


def normalized(profile):
    result=copy.deepcopy(profile)
    result.pop('_path',None)
    for key in ['parser_recover_mwt','parser_prefilter_text','source_dictionary_diagnostics']:
        result.pop(key,None)
    for key in ['checkpoint_dir','parser_processes','parser_prefetch_batches_per_worker','backfill_reused_batches','review_exclusions']:
        result['build'].pop(key,None)
    for key in ['workers','instances','cache_path']:
        result['checker'].pop(key,None)
    return result


def upgrade(old_name,new_name,baseline_code,code_root=Path('dala')):
    old_profile=load_profile(old_name);new_profile=load_profile(new_name)
    source=Path(old_profile['build']['checkpoint_dir']);target=Path(new_profile['build']['checkpoint_dir'])
    if target.exists():raise FileExistsError(target)
    old=json.loads((source/'run.json').read_text())
    if (source/'migration.json').exists():raise ValueError('Chained generation upgrades require a separately reviewed lineage plan')
    if sha256_file(old_profile['_path'])!=old['profile_sha256']:raise ValueError('Old profile changed')
    if normalized(old_profile)!=normalized(new_profile):raise ValueError('Non-approved generation inputs changed')
    baseline={str(p.relative_to(baseline_code)):sha256_file(p) for p in baseline_code.rglob('*.py')}
    if baseline!=old['code_sha256']:raise ValueError('Baseline source does not match original checkpoint')
    current={str(p.relative_to(code_root)):sha256_file(p) for p in code_root.rglob('*.py')}
    changed={k for k in baseline.keys()|current.keys() if baseline.get(k)!=current.get(k)}
    if not changed<=ALLOWED_CODE:raise ValueError(f'Unreviewed code changes: {changed-ALLOWED_CODE}')
    for key,field in [('rulebook','rulebook_sha256'),('exclusions','source_exclusions_sha256'),('sources_config','sources_config_sha256')]:
        if sha256_file(resource(new_profile,key))!=old[field]:raise ValueError(f'Changed input: {key}')
    target.mkdir(parents=True);inventory=[];batch_signatures={};uncommitted=[]
    original_hash=sha256_file(source/'run.json')
    for path in sorted(source.glob('*.candidates.jsonl')):
        try:record=verified_receipt(path)
        except json.JSONDecodeError:record=None
        if record is None:uncommitted.append(path.name);continue
        number=int(path.name.split('.')[0]);batch_signatures[str(number)]=original_hash
        for candidate in [path,path.with_name(path.name.replace('.candidates.','.screened.'))]:
            try:receipt=verified_receipt(candidate)
            except json.JSONDecodeError:receipt=None
            if receipt is None:
                if candidate.exists():uncommitted.append(candidate.name)
                continue
            for file in [candidate,candidate.with_suffix('.receipt.json')]:
                os.link(file,target/file.name)
            inventory.append(dict(file=candidate.name,sha256=receipt['sha256'],receipt_sha256=sha256_file(candidate.with_suffix('.receipt.json'))))
    atomic_json(target/'reused-batches.json',inventory)
    receipt=dict(mode='generation_upgrade',at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint=str(source),original_run_sha256=original_hash,original_signature=old,
        previous_progress=json.loads((source/'progress.json').read_text()) if (source/'progress.json').exists() else {},
        changed_code=sorted(changed),new_code_sha256=current,
        source_rules_models_order_unchanged=True,generation_behavior_changed=True,
        reused_batch_signatures=batch_signatures,
        reused_candidate_batches=len(batch_signatures),reused_screened_batches=sum('.screened.' in r['file'] for r in inventory),
        reused_inventory_sha256=sha256_file(target/'reused-batches.json'),
        backfill_source_batches=sorted(map(int,batch_signatures)) if new_profile['build'].get('backfill_reused_batches') else [],
        uncommitted_to_regenerate=uncommitted)
    atomic_json(target/'migration.json',receipt)
    signature={**old,'profile':{k:v for k,v in new_profile.items() if k!='_path'},
        'profile_sha256':sha256_file(new_profile['_path']),'code_sha256':current,
        'checkpoint_lineage_sha256':sha256_file(target/'migration.json')}
    if new_profile['build'].get('review_exclusions'):
        signature['review_exclusions_sha256']=sha256_file(new_profile['build']['review_exclusions'])
    atomic_json(target/'run.json',signature)
    print(new_profile['language'],len(batch_signatures),'candidate batches preserved;',receipt['reused_screened_batches'],'screened',flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('old_profile',type=Path);p.add_argument('new_profile',type=Path);p.add_argument('--baseline-code',type=Path,required=True);a=p.parse_args()
    upgrade(a.old_profile,a.new_profile,a.baseline_code)
