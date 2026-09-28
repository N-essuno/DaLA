"""Combine compatible candidate pilots through the existing validated exporter.

Rare families take priority when parents offer different edits of one original. Conflicting
source-document contents or rule inventories fail rather than hiding provenance.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from dala.common_pile import sha256_file
from dala.near_duplicates import IndexedNearDuplicates
from dala.pair_pipeline import export_dataset
from dala.validate_dataset import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parents',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    manifests=[];documents={};pairs=[];pool=[];seen=set();near=IndexedNearDuplicates();rejections=Counter()
    for root in a.parents:
        validate(root);m=json.loads((root/'manifest.json').read_text())
        if m['quality_status']!='candidate_rule_and_dictionary_screened':raise ValueError('Candidate parents required')
        if manifests and any(m[k]!=manifests[0][k] for k in ['language','seed','rulebook_sha256','prompts']):raise ValueError('Incompatible parent language, split or rules')
        manifests.append(m)
        for line in (root/'documents.jsonl').open():
            d=json.loads(line);previous=documents.setdefault(d['document_id'],d)
            for key in ['document_sha256','source_dataset','source_revision','license','url']:
                if d[key]!=previous[key]:raise ValueError('Conflicting document provenance')
        for file in sorted(root.glob('*/pairs.jsonl')):
            for line in file.open():
                pool.append(json.loads(line))
    support=Counter(e['corruption_type'] for row in pool for e in row['edits'])
    pool.sort(key=lambda r:(support[r['edits'][0]['corruption_type']],r['pair_id']))
    for row in pool:
        if row['original'] in seen or row['corrupted'] in seen:rejections['text_collision']+=1;continue
        if not near.add(row['original']):rejections['near_duplicate']+=1;continue
        seen.update([row['original'],row['corrupted']]);pairs.append(row)
    manifest=dict(manifests[0]);path=Path(manifest['profile']['rulebook'])
    if sha256_file(path)!=manifest['rulebook_sha256']:raise ValueError('Full rulebook changed')
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as f:book=json.load(f)
    manifest['counts']={'documents':len(documents)}
    manifest['selected_by_type']=dict(Counter(e['corruption_type'] for row in pairs for e in row['edits']))
    manifest['rejections']=dict(rejections)
    # Parent diagnostics count overlapping processing and must not be presented
    # as unique-corpus aggregate eligibility or rejection counts.
    for key in ['eligible_by_type','eligible_sentences_by_type','grammar_context_diagnostics','selected_edits_by_type','processed_batches','checkpoint_receipt_sha256','source_snapshots']:
        manifest.pop(key,None)
    manifest['postprocessing']=dict(operation='merge_compatible_candidate_pilots',parents=[dict(path=str(r.resolve()),manifest_sha256=sha256_file(r/'manifest.json'),source_snapshots=m.get('source_snapshots',[]),parent_postprocessing=m.get('postprocessing'),code_sha256=m['code_sha256']) for r,m in zip(a.parents,manifests)],script_sha256=sha256_file(__file__),native_validation=False)
    export_dataset(pairs,a.output,manifest,book,list(documents.values()),manifest['seed']);validate(a.output)
    print(manifest['language'],len(pairs),dict(rejections),flush=True)

if __name__=='__main__':main()
