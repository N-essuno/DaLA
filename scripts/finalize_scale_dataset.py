"""Finalize fully screened scale checkpoints with attainable balanced quotas.

The requested size is approximate. This command never duplicates sentences to
fill a short source pool and does not alter candidate generation or screening.
It verifies immutable inputs and receipts, then repeats global deduplication and
applies the recorded source-review exclusions before exporting task views.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from dala.batch_pipeline import verified_receipt
from dala.common_pile import documents, snapshots, sha256_file
from dala.near_duplicates import IndexedNearDuplicates
from dala.pair_pipeline import export_dataset
from dala.profiles import load_profile, resource


def attainable_quotas(available, requested):
    total=min(requested,available['train']*100//80,
              available['validation']*100//5,available['test']*100//15)
    while True:
        targets={'train':total*80//100,'validation':total*5//100}
        targets['test']=total-sum(targets.values())
        if all(targets[s]<=available[s] for s in targets):
            return total,targets
        total-=1


def finalize(destination):
    if Path(destination).exists():
        raise FileExistsError(destination)
    profile=load_profile('en_scale');settings=profile['build']
    root=Path(settings['checkpoint_dir']);signature=json.loads((root/'run.json').read_text())
    if signature['profile_sha256']!=sha256_file(profile['_path']):
        # Display-name changes do not alter candidate generation or screening.
        original={k:v for k,v in signature['profile'].items() if k!='name'}
        current={k:v for k,v in profile.items() if k not in {'name','_path'}}
        if original!=current:
            raise ValueError('Profile changed since checkpoint creation')
    for key,path in [('sources_config_sha256',resource(profile,'sources_config')),
                     ('rulebook_sha256',resource(profile,'rulebook')),
                     ('source_exclusions_sha256',resource(profile,'exclusions'))]:
        if signature[key]!=sha256_file(path):
            raise ValueError(f'Changed checkpoint input: {key}')
    current_code={str(p.relative_to('dala')):sha256_file(p) for p in sorted(Path('dala').rglob('*.py'))}
    changed={name for name in current_code.keys() | signature['code_sha256'].keys()
             if current_code.get(name)!=signature['code_sha256'].get(name)}
    if changed:
        # Only these exact final-deduplication/edit-validation fixes may reuse
        # parser/checker receipts. Reverse each literal patch and verify the
        # complete prior file: candidate generation and screening stay identical.
        reversions={
            'near_duplicates.py': [('SequenceMatcher(None,tokens,other,autojunk=False)',
                                    'SequenceMatcher(None,tokens,other)')],
            'language_packs/english.py': [
                (r"match = re.fullmatch(r'(\w+)([^\S\n]+)(\w+)', edit.original)",
                 r"match = re.fullmatch(r'([A-Za-z]+)([ \t]+)([A-Za-z]+)', edit.original)"),
                ('        if not left.isalpha() or not right.isalpha():\n            return False\n','')],
        }
        reader_receipt=json.loads(Path('wiki/artifacts/english-scale-jsonl-reader-fix.json').read_text())
        for name in changed:
            if name in {'dataset_review.py','validate_dataset.py','multilingual.py'}:
                fix=reader_receipt['changes'][name]
                if signature['code_sha256'][name]!=fix['before_sha256'] or current_code[name]!=fix['after_sha256']:
                    raise ValueError('Unexpected JSONL reader change')
                continue
            if name not in reversions:raise ValueError('Generation code changed since checkpoint creation')
            before=(Path('dala')/name).read_text()
            for after,old in reversions[name]:
                if before.count(after)!=1:raise ValueError('Unexpected compatibility patch')
                before=before.replace(after,old)
            if hashlib.sha256(before.encode()).hexdigest()!=signature['code_sha256'][name]:
                raise ValueError('Generation code changed since checkpoint creation')
    cached=snapshots(resource(profile,'sources_config'),offline=True)
    if [receipt for _,_,receipt in cached]!=signature['sources']:
        raise ValueError('Changed source snapshots')
    docs=list(documents(cached))
    expected=(len(docs)+settings['document_batch_size']-1)//settings['document_batch_size']
    review_path=Path(settings['review_exclusions'])
    excluded=set(json.loads(review_path.read_text()))
    pairs=[];seen=set();near=IndexedNearDuplicates();counts=Counter();rejected=Counter();eligible=Counter()
    receipts=[]
    for number in range(expected):
        candidate=root/f'{number:05d}.candidates.jsonl'
        screened=root/f'{number:05d}.screened.jsonl'
        cr,sr=verified_receipt(candidate),verified_receipt(screened)
        if cr is None or sr is None:
            raise ValueError(f'Incomplete screened batch: {number}')
        counts.update(cr['counts']);rejected.update(cr['rejections']);rejected.update(sr['rejections']);eligible.update(cr['eligible'])
        receipts.append(dict(batch=number,candidates_sha256=cr['sha256'],screened_sha256=sr['sha256']))
        for line in screened.open():
            p=json.loads(line)
            if hashlib.sha256(p['original'].encode()).hexdigest() in excluded:
                rejected['review_exclusion']+=1;continue
            if p['original'] in seen or p['corrupted'] in seen:
                rejected['text_collision']+=1;continue
            if not near.add(p['original']):
                rejected['near_duplicate']+=1;continue
            pairs.append(p);seen.update((p['original'],p['corrupted']))
        print(f'Final selection batch {number+1}/{expected}: {len(pairs):,} unique reviewed-filter pairs',flush=True)
    available=Counter(p['split'] for p in pairs)
    # Rounded 80/5/15 proportions, capped by the requested reference size.
    total,targets=attainable_quotas(available,settings['target_pairs'])
    kept=[];splits=Counter();selected=Counter()
    for p in pairs:
        if splits[p['split']]>=targets[p['split']]:
            rejected['split_quota']+=1;continue
        kept.append(p);splits[p['split']]+=1
        selected.update(e['corruption_type'] for e in p['edits'])
    review_files=[Path('wiki/artifacts')/name for name in
                  ('english-scale-first-batch-review.json','english-scale-random-batch-review.json')]
    reviews=[json.loads(p.read_text()) for p in review_files]
    description=profile['description'].replace(
        'Original authors, source URLs and source-specific licenses accompany each document.',
        'Source URLs and licenses accompany each document; supplied author metadata is preserved, with missing values left null.')
    description+=' Expanded sources have known checker-undetected errors. Agent review of 130 pre-curation pairs accepted all injected edits, but flagged 18 originals for grammar/spelling uncertainty or errors and four as boilerplate. Documents with grammar/spelling flags, boilerplate and ambiguous embedded line separators are excluded; remaining source correctness is not established by this review. This is a provisional checker-screened corpus, not a human-validated benchmark.'
    manifest=dict(schema_version='2',language=profile['language'],name=profile['name'],prompts=profile['prompts'],
        source_description=description,profile={k:v for k,v in profile.items() if k!='_path'},
        profile_sha256=sha256_file(profile['_path']),generation_profile_sha256=signature['profile_sha256'],quality_status='checker_screened',
        language_checker=signature['checker'],checker_runtime=json.loads(Path('la_output/tools/runtime.json').read_text()),
        seed=signature['seed'],max_errors=signature['max_errors'],max_documents=None,
        source_snapshots=signature['sources'],sources_config_sha256=signature['sources_config_sha256'],
        rulebook_sha256=signature['rulebook_sha256'],source_exclusions_sha256=signature['source_exclusions_sha256'],
        parser=signature['parser'],runtime=dict(python=platform.python_version(),**{n:version(n) for n in profile['dependencies']}),
        code_sha256=current_code,generation_code_sha256=signature['code_sha256'],
        jsonl_reader_fix_receipt_sha256=sha256_file('wiki/artifacts/english-scale-jsonl-reader-fix.json'),
        finalization_code_changes=sorted(changed),counts=dict(counts),rejections=dict(rejected),
        eligible_sentences_by_type=dict(eligible),selected_edits_by_type=dict(selected),
        checkpoint_receipt_sha256=sha256_file(root/'run.json'),screened_receipts=receipts,
        review_exclusions_sha256=sha256_file(review_path),requested_target_pairs=settings['target_pairs'],
        final_target_pairs=total,available_after_curation=dict(available),final_split_targets=targets,
        finalizer_sha256=sha256_file(__file__),finalization='All completed checkpoints; reviewed exclusions; global deduplication; attainable 80/5/15 quotas',
        quality_review=dict(reviews=[dict(path=str(p),sha256=sha256_file(p),summary=r['summary']) for p,r in zip(review_files,reviews)],human_precision=None))
    result=export_dataset(kept,destination,manifest,json.loads(resource(profile,'rulebook').read_text()),
                          [{k:v for k,v in d.items() if k!='text'} for d in docs],signature['seed'])
    print(json.dumps(dict(pairs=total,rows_per_task=total*2,splits=result['splits']),indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='la_output/english_common_pile_scaled')
    args=parser.parse_args();finalize(args.output_dir)
