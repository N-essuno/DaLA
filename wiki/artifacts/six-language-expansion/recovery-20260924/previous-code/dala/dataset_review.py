"""Export a complete review sheet or a dataset containing explicitly accepted pairs."""
import argparse
import csv
import json
from pathlib import Path

from .build_english import SPLITS, export_dataset
from .common_pile import sha256_file


FIELDS = ['pair_id', 'split', 'source_url', 'original', 'corrupted', 'edits',
          'original_correct', 'corrupted_incorrect', 'edits_valid', 'reviewer', 'notes']


def load_jsonl(path):
    """Split physical JSONL records, preserving Unicode separators inside strings."""
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def load_pairs(dataset):
    return [row for split in SPLITS
            for row in load_jsonl(Path(dataset) / split / 'pairs.jsonl')]


def review_sheet(dataset, output):
    if Path(output).exists():
        raise FileExistsError(output)
    with Path(output).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for p in load_pairs(dataset):
            writer.writerow(dict(pair_id=p['pair_id'], split=p['split'], source_url=p['url'],
                original=p['original'], corrupted=p['corrupted'], edits=json.dumps(p['edits'], ensure_ascii=False)))


def reviewed_export(dataset, decisions, output):
    dataset = Path(dataset)
    pairs = load_pairs(dataset)
    originals = {p['pair_id']: p for p in pairs}
    accepted, seen, reviewers = [], set(), set()
    with Path(decisions).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            pair_id = row['pair_id']
            if pair_id in seen:
                raise ValueError(f'Duplicate decision for {pair_id}')
            seen.add(pair_id)
            if pair_id not in originals:
                raise ValueError(f'Unknown pair ID: {pair_id}')
            p = originals[pair_id]
            if row['original'] != p['original'] or row['corrupted'] != p['corrupted']:
                raise ValueError('Review text differs from dataset; cannot transfer the verdict')
            if all(row.get(field, '').strip().lower() in {'yes', 'true', '1'}
                   for field in ('original_correct', 'corrupted_incorrect', 'edits_valid')):
                if not row.get('reviewer', '').strip():
                    raise ValueError('Accepted pairs must identify the reviewer')
                copy = dict(p, quality_status='review_accepted', review=dict(reviewer=row['reviewer'], notes=row.get('notes', '')))
                accepted.append(copy)
                reviewers.add(row['reviewer'])
    if not accepted:
        raise ValueError('No explicitly accepted pairs; blank or missing judgments are not approval')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    manifest.pop('artifacts', None)
    manifest['quality_status'] = 'review_accepted'
    manifest['review'] = dict(parent_manifest_sha256=sha256_file(dataset / 'manifest.json'),
        decisions_sha256=sha256_file(decisions), parent_pairs=len(pairs),
        accepted_pairs=len(accepted), omitted_pairs=len(pairs) - len(accepted), reviewers=sorted(reviewers))
    book = json.loads((dataset / 'rules.json').read_text())
    documents = load_jsonl(dataset / 'documents.jsonl')
    return export_dataset(accepted, output, manifest, book, documents, manifest['seed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all', action='store_true', help='Write a review CSV for every pair')
    mode.add_argument('--decisions', type=Path, help='Export only pairs with three explicit positive judgments')
    a = p.parse_args()
    if a.all:
        review_sheet(a.dataset, a.output)
    else:
        result = reviewed_export(a.dataset, a.decisions, a.output)
        print(json.dumps(result['review'], indent=2))
