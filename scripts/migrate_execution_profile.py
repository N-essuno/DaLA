"""Reuse verified batches when only worker count, target or checkpoint path changes.

Never overwrite or delete the source run. Parser models, linguistic resources,
batch composition, seed and all other generation options must remain identical.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from dala.batch_pipeline import atomic_json, verified_receipt
from dala.common_pile import sha256_file
from dala.profiles import load_profile, resource


def migrate(old_name,new_name):
    old_profile=load_profile(old_name);new_profile=load_profile(new_name)
    source=Path(old_profile['build']['checkpoint_dir']);target=Path(new_profile['build']['checkpoint_dir'])
    if target.exists():raise FileExistsError(target)
    old=json.loads((source/'run.json').read_text())
    if sha256_file(old_profile['_path'])!=old['profile_sha256']:raise ValueError('Old profile changed')
    current_code={str(p.relative_to(Path('dala'))):sha256_file(p) for p in Path('dala').rglob('*.py')}
    if current_code!=old['code_sha256']:raise ValueError('Generation code changed')
    profile={k:v for k,v in new_profile.items() if k!='_path'}
    def generation_options(value):
        value=copy.deepcopy(value)
        for key in ['checkpoint_dir','parser_processes','target_pairs','split_targets','prestart_parser_workers','worker_pool_sha256','persistent_screening_workers','screening_pool_sha256']:value['build'].pop(key,None)
        value.get('release_policy',{}).pop('target',None)
        for key in ['workers','instances']:value.get('checker',{}).pop(key,None)
        return value
    if generation_options(old['profile'])!=generation_options(profile):raise ValueError('Non-execution profile options changed')
    for key,field in [('rulebook','rulebook_sha256'),('exclusions','source_exclusions_sha256'),('sources_config','sources_config_sha256')]:
        if sha256_file(resource(new_profile,key))!=old[field]:raise ValueError(f'Changed input: {key}')
    target.mkdir();inventory=[];uncommitted=[]
    for path in sorted(source.glob('*.candidates.jsonl')):
        try:record=verified_receipt(path)
        except json.JSONDecodeError:record=None
        if record is None:uncommitted.append(path.name);continue
        for candidate in [path,path.with_name(path.name.replace('.candidates.','.screened.'))]:
            try:receipt=verified_receipt(candidate)
            except json.JSONDecodeError:receipt=None
            if receipt is None:
                if candidate.exists():uncommitted.append(candidate.name)
                continue
            for file in [candidate,candidate.with_suffix('.receipt.json')]:os.link(file,target/file.name)
            inventory.append({'file':candidate.name,'sha256':receipt['sha256'],
                              'receipt_sha256':sha256_file(candidate.with_suffix('.receipt.json'))})
    atomic_json(target/'reused-batches.json',inventory)
    shutil.copy2(source/'run.json',target/'migration-original-run.json')
    replacement={**old,'profile':profile,'profile_sha256':sha256_file(new_profile['_path'])}
    atomic_json(target/'run.json',replacement)
    prior=source/'migration.json'
    receipt={'at':datetime.now(timezone.utc).isoformat(),'source_checkpoint':str(source),
             'original_run_sha256':sha256_file(source/'run.json'),'generation_code_unchanged':True,
             'linguistic_inputs_changed':False,'old_build':old['profile']['build'],'new_build':profile['build'],
             'reused_candidate_batches':sum('.candidates.' in r['file'] for r in inventory),
             'reused_screened_batches':sum('.screened.' in r['file'] for r in inventory),
             'reused_inventory_sha256':sha256_file(target/'reused-batches.json'),
             'uncommitted_to_regenerate':uncommitted,
             'prior_migration':json.loads(prior.read_text()) if prior.exists() else None,
             'prior_migration_sha256':sha256_file(prior) if prior.exists() else None}
    # Preserve generation lineage and recovery ordering through execution-only resizes.
    parent=receipt['prior_migration']
    if parent and parent.get('mode')=='generation_upgrade':
        receipt.update(mode='generation_upgrade',
            original_signature=old,
            previous_progress=json.loads((source/'progress.json').read_text()) if (source/'progress.json').exists() else {},
            backfill_source_batches=parent.get('backfill_source_batches',[]),
            reused_batch_signatures={str(int(r['file'].split('.')[0])):parent.get('reused_batch_signatures',{}).get(str(int(r['file'].split('.')[0])),sha256_file(source/'run.json')) for r in inventory if '.candidates.' in r['file']})
    atomic_json(target/'migration.json',receipt)
    if receipt.get('mode')=='generation_upgrade':
        replacement['checkpoint_lineage_sha256']=sha256_file(target/'migration.json')
        atomic_json(target/'run.json',replacement)
    print(new_name,receipt['reused_candidate_batches'],receipt['reused_screened_batches'],'verified batches preserved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('old_profile');p.add_argument('new_profile');a=p.parse_args()
    migrate(a.old_profile,a.new_profile)
