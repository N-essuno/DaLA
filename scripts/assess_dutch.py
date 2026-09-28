"""Measure a frozen Dutch build and create a reproducible, initially unjudged sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from dala.dataset_review import load_pairs
from dala.dynaword import documents, snapshots
from dala.profiles import load_profile, resource
from dala.validate_dataset import validate


def assess(dataset, output, seed='dala-dutch-dynaword-audit-v1', profile_name='nl', sample_size=100, reviewed_samples=()):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    profile = load_profile(profile_name)
    if hashlib.sha256(Path(profile['_path']).read_bytes()).hexdigest() != manifest['profile_sha256']:
        raise ValueError('Assessment requires the frozen build profile')
    if hashlib.sha256(resource(profile, 'lexicon').read_bytes()).hexdigest() != profile['lexicon_sha256']:
        raise ValueError('Assessment lexicon differs from pinned OpenTaal')
    pairs = load_pairs(dataset)
    checks = validate(dataset)
    # Independently recover original texts from the pinned Parquet snapshot.
    source_docs = {d['document_id']: d for d in documents(snapshots(resource(profile, 'sources_config'), offline=True))}
    words = set(resource(profile, 'lexicon').read_text().casefold().splitlines())
    families, rules, substitutions, documents_count = Counter(), Counter(), Counter(), Counter()
    spelling = 0
    for p in pairs:
        d = source_docs[p['document_id']]
        assert d['text'][p['sentence_start']:p['sentence_end']] == p['original']
        assert d['document_sha256'] == p['document_sha256']
        documents_count[p['document_id']] += 1
        for e in p['edits']:
            families[e['corruption_type']] += 1
            rules[e['rule_id']] += 1
            substitutions[(e['original'].casefold(), e['replacement'].casefold())] += 1
            if e['corruption_type'] == 'spelling':
                assert e['original'].casefold() in words
                assert e['replacement'].casefold() not in words
                spelling += 1
    reviewed_originals = set()
    reviewed_receipts = []
    for path in reviewed_samples:
        path = Path(path)
        previous = json.loads(path.read_text())
        reviewed_originals.update(r['original'] for r in previous['rows'])
        reviewed_receipts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    sample_pool = [p for p in pairs if p['original'] not in reviewed_originals]
    sample = random.Random(seed).sample(sorted(sample_pool, key=lambda p:p['pair_id']), min(sample_size,len(sample_pool)))
    fields = ('pair_id','split','document_id','source_name','original','corrupted','edits')
    rows = [dict(sample_id=f'nl-{i:03d}', **{k:p[k] for k in fields},
                 source_judgment=None, intended_edits_judgment=None, notes=None) for i,p in enumerate(sample)]
    (output/'sample.json').write_text(json.dumps(dict(seed=seed,method='Uniform sample without replacement of frozen build pairs excluding previously reviewed originals; judgments added separately',sample_population=len(sample_pool),excluded_previously_reviewed=len(pairs)-len(sample_pool),reviewed_sample_receipts=reviewed_receipts,rows=rows),ensure_ascii=False,indent=2)+'\n')
    covered = Counter(e['corruption_type'] for p in sample for e in p['edits'])
    selected = {p['pair_id'] for p in sample}
    supplementary = []
    for family in sorted(families):
        wanted = max(0, 5-covered[family])
        available = [p for p in sample_pool if p['pair_id'] not in selected and any(e['corruption_type']==family for e in p['edits'])]
        for p in random.Random(seed+':'+family).sample(sorted(available,key=lambda p:p['pair_id']), min(wanted,len(available))):
            selected.add(p['pair_id'])
            supplementary.append(dict(sample_id=f'nl-extra-{len(supplementary):03d}', **{k:p[k] for k in fields},
                                      source_judgment=None, intended_edits_judgment=None, notes=None))
    (output/'family-supplement.json').write_text(json.dumps(dict(method='Separate coverage supplement: up to five reviewed examples per family including the uniform sample; do not pool for precision estimates',rows=supplementary),ensure_ascii=False,indent=2)+'\n')
    summary = dict(dataset=str(dataset),manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
                   pairs=len(pairs), task_rows_each=2*len(pairs),contributing_documents=len(documents_count),
                   max_pairs_per_document=max(documents_count.values()),
                   sources=dict(Counter(p['source_name'] for p in pairs)),
                   families=dict(families), active_rules=len(rules),configured_rules=len(json.loads((dataset/'rules.json').read_text())['rules']),
                   rule_counts=dict(rules), distinct_substitutions=len(substitutions),
                   top_substitutions=[dict(original=a,replacement=b,count=n) for (a,b),n in substitutions.most_common(20)],
                   validation=checks, source_text_roundtrips=len(pairs),opentaal_spelling_checks=spelling,
                   splits=manifest['splits'],rejections=manifest['rejections'])
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',default='dala-dutch-dynaword-audit-v1')
    p.add_argument('--profile',default='nl')
    p.add_argument('--sample-size',default=100,type=int)
    p.add_argument('--exclude-reviewed-sample',action='append',default=[],type=Path)
    a=p.parse_args();assess(a.dataset,a.output,a.seed,a.profile,a.sample_size,a.exclude_reviewed_sample)
