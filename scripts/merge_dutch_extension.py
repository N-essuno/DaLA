"""Preserve a frozen Dutch base and add screened, globally deduplicated pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from dala.common_pile import sha256_file
from dala.dataset_review import load_pairs, load_jsonl
from dala.near_duplicates import IndexedNearDuplicates
from dala.pair_pipeline import export_dataset
from dala.profiles import load_profile, resource
from dala.validate_dataset import validate


def select_extension(base, additions, excluded, targets):
    near = IndexedNearDuplicates()
    seen, ids, documents = set(), set(), {}
    selected, counts, rejected = [], Counter(), Counter()

    def excluded_source(pair):
        return hashlib.sha256(pair['original'].encode()).hexdigest() in excluded

    for pair in base:
        if excluded_source(pair):
            rejected['base_review_exclusion'] += 1
            continue
        if not near.add(pair['original']):
            raise ValueError('Base must be near-duplicate-free in supplied order')
        if pair['original'] in seen or pair['corrupted'] in seen or pair['pair_id'] in ids:
            raise ValueError('Base has duplicate texts or pair IDs')
        if documents.setdefault(pair['document_id'], pair['split']) != pair['split']:
            raise ValueError('Base document crosses splits')
        selected.append(pair)
        counts[pair['split']] += 1
        seen.update((pair['original'], pair['corrupted']))
        ids.add(pair['pair_id'])
    retained_base = len(selected)
    base_documents = set(documents)
    if any(counts[s] > targets[s] for s in targets):
        raise ValueError('Target would discard base pairs')
    for pair in sorted(additions, key=lambda p: hashlib.sha256(p['pair_id'].encode()).digest()):
        if all(counts[s] == targets[s] for s in targets):
            break
        if excluded_source(pair):
            rejected['extension_review_exclusion'] += 1
            continue
        if pair['document_id'] in base_documents:
            rejected['base_document_overlap'] += 1
            continue
        if counts[pair['split']] >= targets[pair['split']]:
            rejected['split_target'] += 1
            continue
        if documents.get(pair['document_id'], pair['split']) != pair['split']:
            raise ValueError('Extension changes a base document split')
        if pair['pair_id'] in ids or pair['original'] in seen or pair['corrupted'] in seen:
            rejected['text_collision'] += 1
            continue
        if not near.add(pair['original']):
            rejected['near_duplicate'] += 1
            continue
        selected.append(pair)
        counts[pair['split']] += 1
        seen.update((pair['original'], pair['corrupted']))
        ids.add(pair['pair_id'])
        documents[pair['document_id']] = pair['split']
    if dict(counts) != targets:
        raise ValueError(f'Insufficient unique extension pairs; retained {dict(counts)}, target {targets}')
    return selected, dict(retained_base_pairs=retained_base,
                          retained_extension_pairs=len(selected)-retained_base,
                          rejections=dict(rejected), targets=targets)


def merge(base, addition, output, profile_name, total=478930):
    if output.exists():
        raise FileExistsError(output)
    profile = load_profile(profile_name)
    parents = [json.loads((p/'manifest.json').read_text()) for p in (base, addition)]
    book = json.loads(resource(profile, 'rulebook').read_text())
    for root, manifest in zip((base, addition), parents):
        validate(root)
        if (manifest['language'] != profile['language'] or manifest['prompts'] != profile['prompts']
                or json.loads((root/'rules.json').read_text()) != book
                or any(manifest[k] != parents[0][k] for k in ('seed', 'parser', 'max_errors', 'language_checker'))):
            raise ValueError('Incompatible parent language, prompts, rules or split seed')
    excluded = {r['sentence_sha256'] for r in json.loads(resource(profile, 'exclusions').read_text())}
    targets = dict(train=total*80//100, validation=total*10//100)
    targets['test'] = total-sum(targets.values())
    pairs, selection = select_extension(load_pairs(base), load_pairs(addition), excluded, targets)
    documents = {}
    snapshots = {}
    for root, manifest in zip((base, addition), parents):
        used_documents = {p['document_id'] for p in pairs}
        for d in load_jsonl(root/'documents.jsonl'):
            if d['document_id'] not in used_documents:
                continue
            if d['document_id'] in documents and documents[d['document_id']] != d:
                # All additions from base documents were rejected above.
                continue
            documents[d['document_id']] = d
        for snap in manifest['source_snapshots']:
            snapshots[(snap['repo_id'], snap['revision'], snap['file'])] = snap
    manifest = dict(parents[1])
    for key in ('artifacts', 'splits', 'verification', 'processed_batches', 'checkpoint_receipt_sha256', 'counts', 'rejections'):
        manifest.pop(key, None)
    manifest.update(name=profile['name'], profile={k:v for k,v in profile.items() if k!='_path'},
                    profile_sha256=sha256_file(profile['_path']),
                    source_description=profile['description'], source_snapshots=list(snapshots.values()),
                    sources_config_sha256=sha256_file(resource(profile, 'sources_config')),
                    source_exclusions_sha256=sha256_file(resource(profile, 'exclusions')),
                    max_documents=None, target_pairs=total, split_targets=targets,
                    counts=dict(pairs=len(pairs)), rejections=selection['rejections'],
                    selected_by_type=dict(Counter(e['corruption_type'] for p in pairs for e in p['edits'])),
                    extension=dict(selection=selection, merger_sha256=sha256_file(__file__), parents=[dict(path=str(p),manifest_sha256=sha256_file(p/'manifest.json')) for p in (base, addition)],
                                   method='Unchanged base pairs except attributed source exclusions; unchanged new pairs; global text/near-duplicate filtering; 80/10/10 document-preserving targets; no family caps.'))
    result = export_dataset(pairs, output, manifest, book, list(documents.values()), parents[0]['seed'])
    validate(output)
    print(json.dumps(dict(output=str(output), selection=selection, splits=result['splits']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('addition', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--profile', required=True)
    p.add_argument('--target', type=int, default=478930)
    a = p.parse_args()
    merge(a.base, a.addition, a.output, a.profile, a.target)
