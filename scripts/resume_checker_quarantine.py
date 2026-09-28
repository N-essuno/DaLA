"""Continue a failed checker run with exact-source quarantine and preserved lineage."""
import copy,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
from dala.batch_pipeline import atomic_json,verified_receipt
from dala.common_pile import sha256_file
from dala.language_packs.morphology import permits
from types import SimpleNamespace
import gzip


def migrate(old_path,new_path,baseline_code,new_code,policy_upgrade=False):
    old_profile=json.loads(old_path.read_text());new_profile=json.loads(new_path.read_text())
    source=Path(old_profile['build']['checkpoint_dir']);target=Path(new_profile['build']['checkpoint_dir'])
    old=json.loads((source/'run.json').read_text())
    if target.exists():raise FileExistsError(target)
    if sha256_file(old_path)!=old['profile_sha256']:raise ValueError('Changed parent profile')
    normalized=copy.deepcopy(new_profile)
    normalized['build']['checkpoint_dir']=old_profile['build']['checkpoint_dir']
    normalized['build']['parser_processes']=old_profile['build']['parser_processes']
    if policy_upgrade:
        policy=normalized['checker'].get('failure_policy',{})
        if policy.get('mode')!='reject_sentence' or not policy.get('log_path'):
            raise ValueError('Explicit fail-closed checker policy required')
        exclusions=normalized['checker'].get('source_failure_exclusions',{})
        for key in ['failure_policy','workers','instances']:
            if key in old_profile['checker']:normalized['checker'][key]=old_profile['checker'][key]
            else:normalized['checker'].pop(key,None)
    else:
        exclusions=normalized['checker'].pop('source_failure_exclusions')
    if normalized!=old_profile:raise ValueError('Unapproved profile change')
    if (not exclusions and not policy_upgrade) or any(len(k)!=64 or not v for k,v in exclusions.items()):raise ValueError('Invalid exact quarantine')
    for key,field in [('rulebook','rulebook_sha256'),('exclusions','source_exclusions_sha256'),('sources_config','sources_config_sha256')]:
        if sha256_file(Path(new_profile[key]))!=old[field]:raise ValueError('Changed '+key)
    baseline={str(p.relative_to(baseline_code)):sha256_file(p) for p in baseline_code.rglob('*.py')}
    current={str(p.relative_to(new_code)):sha256_file(p) for p in new_code.rglob('*.py')}
    if baseline!=old['code_sha256']:raise ValueError('Parent code mismatch')
    changed={k for k in baseline.keys()|current.keys() if baseline.get(k)!=current.get(k)}
    allowed={'language_check.py','morphology_check.py'} if policy_upgrade else {'language_packs/morphology.py','morphology_check.py','pair_pipeline.py'}
    if not changed<=allowed:raise ValueError('Unreviewed code change')
    parent=json.loads((source/'migration.json').read_text())
    if sha256_file(source/'migration.json')!=old['checkpoint_lineage_sha256']:raise ValueError('Parent lineage changed')
    with gzip.open(old_profile['rulebook'],'rt') as f:book=json.load(f)
    rules={r['id']:r for r in book['rules']}
    target.mkdir(parents=True);inventory=[];signatures={};checked=0
    old_hash=sha256_file(source/'run.json')
    for path in sorted(source.glob('*.candidates.jsonl')):
        if verified_receipt(path) is None:continue
        number=int(path.name.split('.')[0]);signatures[str(number)]=parent.get('reused_batch_signatures',{}).get(str(number),old_hash)
        for candidate in [path,path.with_name(path.name.replace('.candidates.','.screened.'))]:
            receipt=verified_receipt(candidate)
            if receipt is None:continue
            if '.screened.' in candidate.name:
                with candidate.open() as stream:
                    rows = [json.loads(line) for line in stream]
                for row in rows:
                    if any(not permits(rules[e['rule_id']],SimpleNamespace(**e)) for e in row['edits']):
                        raise ValueError('Invalid legacy edit under repaired validator')
                    if hashlib.sha256(row['original'].encode()).hexdigest() in exclusions:
                        raise ValueError('Quarantined source already screened; explicit reselection needed')
                    checked+=1
            for file in [candidate,candidate.with_suffix('.receipt.json')]:os.link(file,target/file.name)
            inventory.append(dict(file=candidate.name,sha256=receipt['sha256'],receipt_sha256=sha256_file(candidate.with_suffix('.receipt.json'))))
    atomic_json(target/'reused-batches.json',inventory)
    receipt=dict(mode='generation_upgrade',at=datetime.now(timezone.utc).isoformat(),source_checkpoint=str(source),
        original_run_sha256=old_hash,original_signature=old,parent_migration=parent,parent_migration_sha256=sha256_file(source/'migration.json'),
        previous_progress=json.loads((source/'progress.json').read_text()),changed_code=sorted(changed),new_code_sha256=current,
        source_rules_models_order_unchanged=True,generation_behavior_changed=True,source_failure_exclusions=exclusions,
        reused_batch_signatures=signatures,reused_candidate_batches=len(signatures),reused_screened_batches=sum('.screened.' in r['file'] for r in inventory),
        reused_inventory_sha256=sha256_file(target/'reused-batches.json'),backfill_source_batches=parent.get('backfill_source_batches',[]),
        screened_rows_revalidated=checked,checker_failure_policy_upgrade=policy_upgrade)
    atomic_json(target/'migration.json',receipt)
    signature={**old,'profile':new_profile,'profile_sha256':sha256_file(new_path),'code_sha256':current,'checkpoint_lineage_sha256':sha256_file(target/'migration.json')}
    atomic_json(target/'run.json',signature)
    print('Preserved',checked,'screened rows;',len(signatures),'candidate batches',flush=True)
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('--baseline-code',type=Path,required=True);p.add_argument('--policy-upgrade',action='store_true');a=p.parse_args()
    migrate(a.old,a.new,a.baseline_code,Path('dala'),policy_upgrade=a.policy_upgrade)
