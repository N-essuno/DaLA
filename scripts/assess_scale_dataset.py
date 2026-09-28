"""Assess the final scale corpus and independently verify source spans/review flags."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from dala.common_pile import documents, snapshots, sha256_file
from dala.dataset_review import load_pairs
from dala.curation import normalized_tokens
from scripts.assess_dataset import assess


def main():
    root=Path('la_output/english_common_pile_scaled');out=Path('wiki/artifacts/english-scale-comparison.json')
    report=assess('la_output/english_generators',root,out)
    pairs=load_pairs(root)
    docs={d['document_id']:d for d in documents(snapshots('config/english_sources_scale.json',offline=True))}
    exclusions=set(json.loads(Path('config/english_scale_review_exclusions.json').read_text()))
    reviewed={r['pair_id']:r for name in ['english-scale-first-batch-review.json','english-scale-random-batch-review.json']
              for r in json.loads((Path('wiki/artifacts')/name).read_text())['rows']}
    matched=[];operators=Counter();surfaces=defaultdict(set);authors=Counter();max_tokens=0
    for p in pairs:
        doc=docs[p['document_id']]
        if doc['text'][p['sentence_start']:p['sentence_end']]!=p['original']:
            raise ValueError('Original differs from pinned source span')
        for key in ['document_sha256','author','license','captured_license','license_evidence_url']:
            if p.get(key)!=doc.get(key):raise ValueError(f'Provenance mismatch: {key}')
        if hashlib.sha256(p['original'].encode()).hexdigest() in exclusions:
            raise ValueError('Excluded original retained')
        max_tokens=max(max_tokens,len(normalized_tokens(p['original'])))
        authors['with_author' if p.get('author') else 'without_author']+=1
        if p['pair_id'] in reviewed:
            r=reviewed[p['pair_id']]
            if r['source_judgment']!='accept' or r['edit_judgment']!='accept':
                raise ValueError('Flagged review retained')
            if r['original']!=p['original'] or r['corrupted']!=p['corrupted'] or r['edits']!=p['edits']:
                raise ValueError('Reviewed pair changed')
            matched.append(p['pair_id'])
        for e in p['edits']:
            op=e['rule_id'].split(':')[0] if not e['rule_id'].startswith(('fallback:','spelling_pattern:')) else e['rule_id']
            operators[op]+=1;surfaces[op].add((e['original'].lower(),e['replacement'].lower()))
    report.update(operators={k:dict(edits=v,distinct_surface_substitutions=len(surfaces[k])) for k,v in sorted(operators.items())},
                  original_source_spans_verified=True,source_attribution_verified=True,
                  author_coverage=dict(authors),all_review_exclusions_absent=True,
                  retained_reviewed_pairs_exactly_unchanged=len(matched),maximum_normalized_sentence_tokens=max_tokens)
    out.write_text(json.dumps(report,indent=2)+'\n')
    (Path('wiki/artifacts')/'english-scale-final-review.json').write_text(json.dumps(dict(
        method='Transfer only unchanged accepted agent-reviewed pairs; not a fresh post-curation sample or human validation',
        retained_unchanged_pair_ids=matched,all_flagged_originals_absent=True,human_precision=None),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','current','operators')},indent=2))


if __name__=='__main__':main()
