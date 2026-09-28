"""Measure retained amount, lexical/error diversity and screening coverage.

These are corpus and mechanical metrics, not human precision estimates.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from dala.dataset_review import load_pairs
from dala.common_pile import sha256_file


def metrics(path):
    root=Path(path);pairs=load_pairs(root);manifest=json.loads((root/'manifest.json').read_text())
    families=defaultdict(Counter);all_edits=Counter();sources=Counter();errorcounts=Counter()
    for p in pairs:
        sources[p['source_name']]+=1;errorcounts[len(p['edits'])]+=1
        for e in p['edits']:
            pair=(e['original'].lower(),e['replacement'].lower())
            families[e['corruption_type']][pair]+=1;all_edits[pair]+=1
    def describe(c):
        n=sum(c.values());h=-sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0
        return dict(edits=n,distinct_surface_substitutions=len(c),distinct_original_forms=len({k[0] for k in c}),top_substitution_share=max(c.values())/n if n else 0,entropy_bits=h,effective_substitutions=2**h if n else 0,top_substitutions=[dict(original=k[0],replacement=k[1],count=v) for k,v in c.most_common(10)])
    train={(e['corruption_type'],e['original'].lower(),e['replacement'].lower()) for p in pairs if p['split']=='train' for e in p['edits']}
    test=[(e['corruption_type'],e['original'].lower(),e['replacement'].lower()) for p in pairs if p['split']=='test' for e in p['edits']]
    return dict(path=str(root),manifest_sha256=sha256_file(root/'manifest.json'),pairs=len(pairs),rows_per_task=2*len(pairs),source_documents=len({p['document_id'] for p in pairs}),splits=manifest['splits'],pairs_by_source=dict(sources),pairs_by_error_count=dict(errorcounts),all_edits=describe(all_edits),families={f:describe(c) for f,c in sorted(families.items())},held_out_edit_instances=len(test),held_out_edits_unseen_surface_pair_in_train=sum(e not in train for e in test),quality_status=manifest['quality_status'],rejections=manifest.get('rejections',{}),human_precision=None)


def assess(baseline,current,output):
    before,after=metrics(baseline),metrics(current)
    old=load_pairs(baseline);new=load_pairs(current)
    oldtexts={p['original'] for p in old};newtexts={p['original'] for p in new}
    oldedits={(e['corruption_type'],e['original'].lower(),e['replacement'].lower()) for p in old for e in p['edits']}
    newedits=[(e['corruption_type'],e['original'].lower(),e['replacement'].lower()) for p in new for e in p['edits']]
    report=dict(baseline=before,current=after,pair_increase=after['pairs']-before['pairs'],pair_growth_percent=100*(after['pairs']/before['pairs']-1),new_original_sentences=len(newtexts-oldtexts),previous_originals_not_retained=len(oldtexts-newtexts),edit_instances_with_new_surface_substitution=sum(e not in oldedits for e in newedits),distinct_new_surface_substitutions=len(set(newedits)-oldedits),limitations=['Checker agreement is not human precision.','Family and lexical frequencies reflect source and rule selection.','The manual queue is family-stratified; unweighted results are not corpus-wide estimates.'])
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def review_sample(dataset,output,per_family=10,seed=20260921):
    pairs=load_pairs(dataset);families=sorted({e['corruption_type'] for p in pairs for e in p['edits']});selected=set();rows=[]
    for family in families:
        eligible=sorted([p for p in pairs if p['pair_id'] not in selected and any(e['corruption_type']==family for e in p['edits'])],key=lambda p:p['pair_id'])
        for p in random.Random(f'{seed}:{family}').sample(eligible,min(per_family,len(eligible))):
            selected.add(p['pair_id']);rows.append(dict(pair_id=p['pair_id'],stratum=family,split=p['split'],source_url=p['url'],original=p['original'],corrupted=p['corrupted'],edits=[dict(type=e['corruption_type'],original=e['original'],replacement=e['replacement']) for e in p['edits']],source_judgment=None,edit_judgment=None,notes=None))
    Path(output).write_text(json.dumps(dict(method='Agent linguistic inspection queue, not human validation',seed=seed,per_family=per_family,rows=rows),ensure_ascii=False,indent=2)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);p.add_argument('--review-output')
    a=p.parse_args();r=assess(a.baseline,a.current,a.output)
    if a.review_output:review_sample(a.current,a.review_output)
    print(json.dumps({k:v for k,v in r.items() if k not in ['baseline','current']},indent=2))
