#!/usr/bin/env python3
"""Prepare passed-only standalone DaLA packages; never upload or alter producer data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import yaml
import hf_package_runtime as runtime

NAMES={'ca':'Catalan','cs':'Czech','de':'German','el':'Greek','es':'Spanish','et':'Estonian','fi':'Finnish','fr':'French','it':'Italian','pt-PT':'European Portuguese','ro':'Romanian','uk':'Ukrainian'}
FIELDS=('original_correct','corrupted_incorrect','edit_is_grammar_or_spelling','meaning_preserved')

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def eligible(status,result,excluded=False):
    return not excluded and status=='done' and isinstance(result,dict) and result.get('decision')=='pass' and all(result.get(k)=='yes' for k in FIELDS)

def freeze(audit,output):
    marker=output/'audit-snapshot.json'
    if marker.exists():return json.loads(marker.read_text())
    c=sqlite3.connect(f'file:{audit}/jobs.sqlite?mode=ro',uri=True)
    c.execute('BEGIN')
    if c.execute("select count(*) from jobs where status in ('pending','running')").fetchone()[0]:raise ValueError('Audit not terminal')
    config=json.loads((audit/'configuration.json').read_text());paths={}
    for lang in NAMES:
        p=output/'selection'/f'{lang}.jsonl'
        p.parent.mkdir(parents=True,exist_ok=True);paths[lang]=str(p.resolve())
    handles={lang:open(p,'w') for lang,p in paths.items()};counts=Counter()
    try:
        for key,component,raw,status,result,error,excluded in c.execute('SELECT j.id,j.component,j.record,j.status,j.result,j.error,x.id IS NOT NULL FROM jobs j LEFT JOIN release_exclusions x ON x.id=j.id ORDER BY j.rowid'):
            r=json.loads(raw);v=json.loads(result) if result else None
            keep=eligible(status,v,excluded)
            row=dict(job_id=key,component=component,pair_id=r['pair_id'],ordinal=r['ordinal'],original=r['original'],corrupted=r['corrupted'],audit=v,status=status,excluded=bool(excluded),keep=keep,error=error)
            handles[r['language']].write(json.dumps(row,ensure_ascii=False)+'\n');counts['kept' if keep else 'omitted']+=1
    finally:
        for f in handles.values():f.close()
    c.commit();c.close()
    result=dict(policy='automated_pair_audit_pass_only',counts=dict(counts),sources=config['sources'],model=config['model'],prompt=config['prompt'],schema=config['schema'],selection={lang:dict(path=p,sha256=runtime.digest(p)) for lang,p in paths.items()},audit_configuration_sha256=runtime.digest(audit/'configuration.json'),human_validation=False)
    write(marker,result);return result

def build(job):
    lang,source,selection,output,snapshot=job
    dest=Path(output)/('dala-'+lang.lower()+'-audited');building=dest.with_name('.building-'+dest.name)
    if dest.exists():
        proof=json.loads((dest/'metadata/validation.json').read_text())
        if proof.get('status')!='passed':raise ValueError('Unvalidated existing package')
        return dict(language=lang,path=str(dest.resolve()),repo_id='schneiderkamplab/'+dest.name,**proof)
    if building.exists():raise FileExistsError('Partial build retained: '+str(building))
    for sub in ('metadata','provenance','data'):(building/sub).mkdir(parents=True,exist_ok=True)
    parent=Path(source);generation=json.loads((parent/'manifest.json').read_text());selected={};audit_counts=Counter()
    for name in ('documents.jsonl','rules.json'):
        if runtime.digest(parent/name) != generation['artifacts'][name]['sha256']:
            raise ValueError('Producer provenance resource changed: '+str(parent/name))
    with runtime.compressed_writer(building/'metadata/audit-dispositions.jsonl.gz') as f:
        for line in open(selection['path']):
            r=json.loads(line);audit_counts['kept' if r['keep'] else 'omitted']+=1
            if r['keep']:selected[(r['component'].split(':')[1],r['pair_id'])]=r
            f.write(runtime.encode({k:v for k,v in r.items() if k not in ('original','corrupted')}))
    assert runtime.digest(selection['path'])==selection['sha256']
    split_counts={};documents=set();licenses=Counter();families=Counter();matched=0
    for split in runtime.SPLITS:
        path=parent/split/'pairs.jsonl';expected=next(x for x in snapshot['sources'] if x['language']==lang and x['component']==lang+':'+split)
        assert runtime.digest(path)==expected['sha256'] and runtime.digest(parent/'manifest.json')==expected['receipt_sha256']
        count=0
        with runtime.compressed_writer(building/'provenance'/f'{split}.pairs.jsonl.gz') as f:
            for ordinal,line in enumerate(open(path)):
                pair=json.loads(line);audit=selected.get((split,pair['pair_id']))
                if audit is None:continue
                assert pair['language']==lang and pair['split']==split and audit['ordinal']==ordinal
                assert (pair['original'],pair['corrupted'])==(audit['original'],audit['corrupted'])
                pair.update(quality_status='automated_pair_audit_pass',audit=dict(model=snapshot['model'],result=audit['audit'],job_id=audit['job_id'],human_validated=False))
                f.write(runtime.encode(pair));count+=1;matched+=1;documents.add(pair['document_id']);licenses[pair['license']]+=1
                families.update(e['corruption_type'] for e in pair['edits'])
        split_counts[split]=dict(pairs=count,rows_per_task=2*count)
    assert matched==len(selected)
    with runtime.compressed_writer(building/'provenance/documents.jsonl.gz') as f:
        for line in open(parent/'documents.jsonl'):
            doc=json.loads(line)
            if doc['document_id'] in documents:f.write(runtime.encode(doc))
    for name in ('manifest.json','rules.json'):shutil.copyfile(parent/name,building/'metadata'/('generation-'+name))
    write(building/'metadata/config.json',dict(tasks=list(runtime.TASKS),prompts=generation['prompts'],seed=generation['seed'],shard_rows=100000,sample_source_judgments={}))
    write(building/'metadata/audit-policy.json',dict(policy=snapshot['policy'],model=snapshot['model'],prompt=snapshot['prompt'],schema=snapshot['schema'],selection_sha256=selection['sha256'],counts=dict(audit_counts),human_validation=False))
    shutil.copyfile(Path(__file__).with_name('hf_package_runtime.py'),building/'recreate_dataset.py')
    front=dict(language=["pt" if lang=="pt-PT" else lang],license='other',license_name='source-specific',license_link='LICENSE.md',task_categories=['text-classification','text-generation'],configs=[dict(config_name=t,data_files=[dict(split=s,path=f'data/{t}/{s}-*.jsonl.gz') for s in runtime.SPLITS]) for t in runtime.TASKS])
    if lang=='pt-PT':front['language_bcp47']=['pt-PT']
    table='\n'.join(f"| {s} | {v['pairs']:,} | {v['rows_per_task']:,} |" for s,v in split_counts.items())
    (building/'README.md').write_text('---\n'+yaml.safe_dump(front,sort_keys=False,allow_unicode=True)+'---\n'+f'''# DaLA {NAMES[lang]} — audited grammar and spelling

{matched:,} synthetic original/corrupted pairs, selected by an automated Gemma audit.
Not native-speaker validated or a gold benchmark. No paraphrasing, simplification
or style-transfer task. Valid variants should not be labelled errors; model
judgments remain fallible and systematic errors can survive filtering.

| Split | Pairs | Rows per task |
| --- | ---: | ---: |
{table}

Two configurations: acceptability (yes/no) and correction (return the original).
Each pair yields a clean control and a corrupted input per task: four chat rows,
not four independent pairs. Use only `messages` for training; metadata and
provenance contain labels. Preserve document splits and use train only in DFM12.
Prompts explicitly name {NAMES[lang]}. Source-language standards are not mixed.

All retained pairs passed all four audit criteria: source correctness, injected
error validity, grammar/spelling scope, and meaning preservation. Flagged,
uncertain, failed and explicitly excluded pairs are omitted. Failure is not a
linguistic rejection. Audit decisions and the rubric are in metadata; exact
sentences, edits, source revisions, URLs, licenses and document attribution are
in provenance. Some source errors or valid-variant corruptions may remain.

The supplied generation manifest and rule inventory document pinned resources.
Sources are excerpts; corrupted variants are deliberately modified. No source
publisher endorsement is implied. See LICENSE.md and per-document attribution.

Run `python recreate_dataset.py --root .` to independently verify checksums,
exact chat reconstruction, clean controls, edit round trips and document splits.
This validates package mechanics, not linguistic precision. Publication has not
yet been performed by this preparation step.
''')
    (building/'LICENSE.md').write_text('# Source-specific licensing and attribution\n\nNo blanket replacement license is asserted. Retain per-record license, source\nrevision, URL, title and document attribution from provenance. DaLA selects\nsentences, adds task instructions and deliberately changes recorded grammar or\nspelling spans; clean controls and targets reproduce source excerpts.\n\nLicense labels and retained pair counts:\n'+''.join(f'- {k}: {v:,}\n' for k,v in sorted(licenses.items())))
    (building/'.gitattributes').write_text('*.gz filter=lfs diff=lfs merge=lfs -text\n')
    shards=[]
    for task in runtime.TASKS:shards.extend(runtime.build_data(building,building/'data'/task,task))
    files={str(p.relative_to(building)):dict(bytes=p.stat().st_size,sha256=runtime.digest(p)) for p in sorted(building.rglob('*')) if p.is_file()}
    write(building/'metadata/manifest.json',dict(schema_version=1,repo_id='schneiderkamplab/'+dest.name,quality_status='automated_pair_audit_pass_only',source_manifest_sha256=runtime.digest(parent/'manifest.json'),splits=split_counts,shards=shards,files=files,corruption_families=dict(families),upload_performed=False))
    proof=runtime.validate(building);write(building/'metadata/validation.json',proof)
    building.rename(dest)
    print('HF_READY',lang,matched,flush=True)
    return dict(language=lang,path=str(dest.resolve()),repo_id='schneiderkamplab/'+dest.name,**proof)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--audit',type=Path,default=Path('la_output/european_audit/full_20260927'));p.add_argument('--output',type=Path,default=Path('export-upload/european-audited-20260928'));p.add_argument('--workers',type=int,default=12);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    snapshot=freeze(a.audit.resolve(),a.output.resolve());jobs=[]
    for lang in NAMES:
        source=Path(next(s['path'] for s in snapshot['sources'] if s['language']==lang)).parent.parent
        jobs.append((lang,str(source),snapshot['selection'][lang],str(a.output.resolve()),snapshot))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(build,jobs))
    write(a.output/'packages.json',dict(status='validated_local_upload_preparation',upload_performed=False,packages=results))

if __name__=='__main__':main()
