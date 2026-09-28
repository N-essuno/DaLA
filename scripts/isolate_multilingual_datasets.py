"""Write separate candidate datasets with shared exact/near-original isolation.

Supply smaller languages first to preserve their scarce clean material. Drops
cross-language duplicates rather than moving documents across existing splits.
Never overwrites inputs or certifies linguistic precision.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from dala.near_duplicates import IndexedNearDuplicates
from dala.pair_pipeline import export_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets',nargs='+',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--review-exclusions',type=Path,
                   help='Late review flags by language and original SHA256; leave generation inputs immutable')
    p.add_argument('--pair-caps',nargs='*',default=[],help='Non-destructive output caps, e.g. nn=478930; 80/10/10 splits')
    a=p.parse_args()
    caps={}
    for item in a.pair_caps:
        language,total=item.split('=',1);total=int(total)
        if total<10:raise ValueError('Pair cap must be at least 10')
        caps[language]={'train':total*8//10,'validation':total//10,'test':total-total*8//10-total//10}
    review=json.loads(a.review_exclusions.read_text()) if a.review_exclusions else {}
    near=IndexedNearDuplicates();seen=set();report={}
    for root in a.datasets:
        manifest=json.loads((root/'manifest.json').read_text());language=manifest['language']
        if language in report:raise ValueError('One input per written standard required')
        book=json.loads((root/'rules.json').read_text());rows=[];rejected=Counter()
        # Stable priority within each language, independent of export row shuffle.
        candidates=[json.loads(s) for f in sorted(root.glob('*/pairs.jsonl')) for s in f.open()]
        excluded={r['original_sha256'] for r in review.get(language,[])}
        quotas=caps.get(language);split_counts=Counter()
        for row in sorted(candidates,key=lambda r:r['pair_id']):
            if hashlib.sha256(row['original'].encode()).hexdigest() in excluded:
                rejected['late_agent_review_exclusion']+=1;continue
            if quotas and split_counts[row['split']]>=quotas[row['split']]:
                rejected['output_cap_retained_in_full_input']+=1;continue
            if row['original'] in seen or row['corrupted'] in seen:
                rejected['cross_dataset_text_collision']+=1;continue
            if not near.add(row['original']):
                rejected['cross_dataset_near_original']+=1;continue
            rows.append(row);split_counts[row['split']]+=1;seen.update((row['original'],row['corrupted']))
        manifest=dict(manifest)
        if quotas:
            manifest['target_pairs']=sum(quotas.values())
            manifest['target_shortfall']={s:max(0,n-split_counts[s]) for s,n in quotas.items()}
            manifest['output_cap']={'split_targets':quotas,'method':'stable pair_id order within original splits, after review exclusions and deduplication',
                                    'full_input_preserved':str(root),'full_input_pairs':len(candidates),
                                    'unselected_pairs_deleted':False}
        if a.review_exclusions:
            manifest['late_agent_review']={'exclusions_file':str(a.review_exclusions),
                'sha256':hashlib.sha256(a.review_exclusions.read_bytes()).hexdigest(),
                'excluded_originals':len(excluded),'human_linguistic_validation':False}
        manifest['cross_dataset_isolation']={'method':'exact both views; normalized original trigrams plus SequenceMatcher >=0.9',
             'input':str(root),'input_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
             'earlier_languages':list(report),'rejections':dict(rejected),'linguistic_precision_claim':False}
        selected=Counter(e['corruption_type'] for row in rows for e in row['edits'])
        manifest['selected_edits_by_type']=dict(selected)
        manifest.pop('selected_by_type',None)
        docs=[json.loads(s) for s in (root/'documents.jsonl').open()]
        destination=a.output_root/language
        export_dataset(rows,destination,manifest,book,docs,manifest['seed'])
        report[language]={'input_pairs':len(candidates),'retained_pairs':len(rows),'rejections':dict(rejected),'output':str(destination)}
        (a.output_root/'isolation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(language,report[language],flush=True)


if __name__=='__main__':main()
