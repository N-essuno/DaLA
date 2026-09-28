"""Reproducible fresh English/Danish comparison samples, before agent judgments."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import pyarrow.parquet as pq

N=100
SEED='dala-quality-comparison-20260921-v1'
OUT=Path('wiki/artifacts/language-quality-comparison')


def sample(rows,language):
    rng=random.Random(SEED+':'+language);selected=[];n=0
    for n,row in enumerate(rows,1):
        if n<=N:selected.append(row)
        else:
            i=rng.randrange(n)
            if i<N:selected[i]=row
    return selected,n


def english():
    for split in ('train','validation','test'):
        path=Path('la_output/english_common_pile_scaled')/split/'pairs.jsonl'
        for line in path.open():
            p=json.loads(line)
            yield dict(pair_id=p['pair_id'],split=split,original=p['original'],corrupted=p['corrupted'],
                       source=p['source_name'],url=p['url'],types=[e['corruption_type'] for e in p['edits']],edits=p['edits'])


def parquet_rows(root):
    for split in ('train','val','test'):
        path=Path(root)/'data'/f'{split}-00000-of-00001.parquet'
        offset=0
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                yield split,offset,row
                offset+=1


def danish():
    for split,index,p in parquet_rows('la_output/cache/danish_tv2r_raw_comparison'):
        if p['label']=='incorrect':
            yield dict(pair_id=f'{split}:{index}',split=split,corrupted=p['text'],types=[p['corruption_type']],source='tv2r')


def words(s):return re.findall(r'\w+|[^\w\s]',s.casefold())
def grams(s):
    w=words(s)
    return set(zip(w,w[1:],w[2:]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    en,en_n=sample(english(),'en');da,da_n=sample(danish(),'da')
    requested={r['corrupted'] for r in da};matches=defaultdict(set)
    for split,index,p in parquet_rows('la_output/cache/danish_tv2r_comparison'):
        if p['corrupted'] in requested and p['original']!=p['corrupted']:
            matches[p['corrupted']].add(p['original'])
    missing=[r for r in da if not matches[r['corrupted']]]
    target_index=defaultdict(list)
    for i,r in enumerate(missing):
        for gram in grams(r['corrupted']):target_index[gram].append(i)
    possibilities=[Counter() for _ in missing]
    for split,index,p in parquet_rows('la_output/cache/danish_tv2r_raw_comparison'):
        if p['label']!='correct':continue
        counts=Counter(i for gram in grams(p['text']) for i in target_index.get(gram,()))
        for i,count in counts.items():possibilities[i][p['text']]=count
    recovery=[]
    for i,r in enumerate(missing):
        candidates=sorted(possibilities[i],key=lambda s:possibilities[i][s],reverse=True)[:20]
        ranked=sorted([(SequenceMatcher(None,words(r['corrupted']),words(s),autojunk=False).ratio(),s) for s in candidates],reverse=True)
        recovery.append(dict(pair_id=r['pair_id'],corrupted=r['corrupted'],type=r['types'],candidates=ranked[:3]))
        if ranked:
            r['original']=ranked[0][1];r['pair_recovery']='nearest published clean sentence; pending explicit agent alignment check'
        else:r['original']=None;r['pair_recovery']='unresolved'
    for r in da:
        if matches[r['corrupted']]:
            values=sorted(matches[r['corrupted']]);r['original']=values[0]
            r['pair_recovery']='exact corrupted-text match in pinned paired export'
            if len(values)!=1:raise ValueError('Ambiguous exact counterpart')
    for language,rows,n in [('en',en,en_n),('da',da,da_n)]:
        for i,r in enumerate(rows):
            r['sample_id']=f'{language}-{i:03d}'
            r.update(source_judgment=None,corruption_judgment=None,intended_edits_judgment=None,formatting_damage=None,notes=None)
        (OUT/f'{language}-sample.json').write_text(json.dumps(dict(language=language,seed=SEED,population=n,n=N,
            method='Uniform reservoir sample of corrupted pairs/incorrect rows across all published splits; before judgments',rows=rows),ensure_ascii=False,indent=2)+'\n')
    (OUT/'danish-pair-recovery.json').write_text(json.dumps(recovery,ensure_ascii=False,indent=2)+'\n')
    receipts={}
    for root in ['la_output/cache/danish_tv2r_comparison','la_output/cache/danish_tv2r_raw_comparison']:
        for path in Path(root).rglob('*.parquet'):
            receipts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'sampling.json').write_text(json.dumps(dict(seed=SEED,n_per_language=N,english_population=en_n,danish_population=da_n,
        english_manifest_sha256=hashlib.sha256(Path('la_output/english_common_pile_scaled/manifest.json').read_bytes()).hexdigest(),
        danish_raw=dict(repo='giannor/dala_tv2r',revision='b2deeb25200996ccedf61858df9d4620ecb3a039'),
        danish_pairs=dict(repo='giannor/dala_gen_tv2r',revision='75a61fd90fd929a5859dd5caccde02300e5b5036'),
        danish_exact_paired_matches=N-len(missing),danish_needing_alignment_review=len(missing),file_sha256=receipts),indent=2)+'\n')
    print('English',en_n,'Danish',da_n,'Danish counterpart review',len(missing),flush=True)
    print(json.dumps(recovery,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
