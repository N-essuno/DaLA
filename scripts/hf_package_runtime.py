"""Validate or recreate a DaLA chat export from its bundled canonical pairs.

Standard library only. This recreates the task export, not upstream corruption
selection or linguistic validation. No downloads, credentials or repository code.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random

SPLITS = ('train', 'validation', 'test')
TASKS = ('acceptability', 'correction')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def compressed_writer(path):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as handle:
            yield handle


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def encode(row):
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def chat_row(pair, clean, config):
    task = config['task']
    text = pair['original'] if clean else pair['corrupted']
    answer = ('yes' if clean else 'no') if task == 'acceptability' else pair['original']
    return dict(messages=[dict(role='user', content=config['prompts'][task] + '\n\n' + text),
                          dict(role='assistant', content=answer)],
                metadata=dict(row_id=f"{task}:{pair['pair_id']}:{'clean' if clean else 'corrupted'}",
                    task=task, pair_id=pair['pair_id'], document_id=pair['document_id'], split=pair['split'],
                    variant='clean' if clean else 'corrupted', source_label='correct' if clean else 'incorrect',
                    error_count=0 if clean else len(pair['edits']),
                    corruption_types=[] if clean else sorted({e['corruption_type'] for e in pair['edits']}),
                    source_name=pair['source_name'], source_url=pair['url'], source_license=pair['license'],
                    source_author=pair.get('author'), quality_status=pair['quality_status'],
                    agent_source_judgment=config['sample_source_judgments'].get(pair['pair_id'], 'not_reviewed')))


def expected_rows(root, split, config):
    pairs = list(read_gzip(root / 'provenance' / f'{split}.pairs.jsonl.gz'))
    order = list(range(2 * len(pairs)))
    random.Random(config['seed']).shuffle(order)
    for index in order:
        yield chat_row(pairs[index // 2], index % 2 == 0, config)


def build_data(root, destination, task):
    config = dict(json.loads((root / 'metadata/config.json').read_text()), task=task)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for split in SPLITS:
        rows = iter(expected_rows(root, split, config))
        shard = 0
        while True:
            first = next(rows, None)
            if first is None:
                break
            path = destination / f'{split}-{shard:05d}.jsonl.gz'
            count = 0
            plain_hash = hashlib.sha256()
            with compressed_writer(path) as handle:
                for row in itertools.chain([first], itertools.islice(rows, config['shard_rows'] - 1)):
                    line = encode(row); handle.write(line); plain_hash.update(line); count += 1
            records.append(dict(path=f'data/{task}/' + path.name, task=task, split=split, rows=count,
                                bytes=path.stat().st_size, sha256=digest(path),
                                uncompressed_sha256=plain_hash.hexdigest()))
            shard += 1
        print(f'{config["task"]}: wrote {split}', flush=True)
    return records


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'metadata/manifest.json').read_text())
    config = json.loads((root / 'metadata/config.json').read_text())
    expected_files = set(manifest['files']) | {'metadata/manifest.json', 'metadata/validation.json', '.gitattributes'}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink in package: {path}')
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise ValueError(f'Hard-linked file: {path}')
            if str(path.relative_to(root)) not in expected_files:
                raise ValueError(f'Unexpected upload file: {path}')
    for name, record in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Unsafe/missing artifact: {name}')
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Checksum/size mismatch: {name}')
    doc_rows = list(read_gzip(root / 'provenance/documents.jsonl.gz'))
    docs = {d['document_id']: d for d in doc_rows}
    if len(docs) != len(doc_rows):
        raise ValueError('Duplicate document IDs')
    pair_ids = set(); doc_splits = {}; results = {}
    for split in SPLITS:
        pairs = 0
        for p in read_gzip(root / 'provenance' / f'{split}.pairs.jsonl.gz'):
            if p['pair_id'] in pair_ids or p['split'] != split:
                raise ValueError('Repeated pair or incorrect split')
            pair_ids.add(p['pair_id']); pairs += 1
            if doc_splits.setdefault(p['document_id'], split) != split:
                raise ValueError('Document crosses splits')
            doc = docs[p['document_id']]
            for key in ('url', 'license', 'author', 'source_revision', 'document_sha256'):
                if p.get(key) != doc.get(key):
                    raise ValueError(f'Provenance mismatch: {key}')
            text = p['original']
            for e in sorted(p['edits'], key=lambda e: e['start'], reverse=True):
                if text[e['start']:e['end']] != e['original']:
                    raise ValueError('Invalid source edit span')
                text = text[:e['start']] + e['replacement'] + text[e['end']:]
            if text != p['corrupted']:
                raise ValueError('Corruption reconstruction failed')
            for e in sorted(p['edits'], key=lambda e: e['corrupted_start'], reverse=True):
                if text[e['corrupted_start']:e['corrupted_end']] != e['replacement']:
                    raise ValueError('Invalid corrupted edit span')
                text = text[:e['corrupted_start']] + e['original'] + text[e['corrupted_end']:]
            if text != p['original']:
                raise ValueError('Correction reconstruction failed')
        task_results = {}
        for task in TASKS:
            task_config = dict(config, task=task)
            shards = [s for s in manifest['shards'] if s['split'] == split and s['task'] == task]
            actual = itertools.chain.from_iterable(read_gzip(root / s['path']) for s in shards)
            counts = Counter()
            for row, expected in itertools.zip_longest(actual, expected_rows(root, split, task_config)):
                if row != expected:
                    raise ValueError(f'Chat row differs from frozen pair: {task}/{split}/{sum(counts.values())}')
                counts[row['metadata']['variant']] += 1
            if counts != {'clean': pairs, 'corrupted': pairs} or pairs != manifest['splits'][split]['pairs']:
                raise ValueError('Missing controls, variants or pairs')
            task_results[task] = dict(rows=sum(counts.values()), variants=dict(counts))
            print(f'Validated {task}/{split}: {sum(counts.values()):,} rows', flush=True)
        results[split] = dict(pairs=pairs, tasks=task_results)

    return dict(status='passed', splits=results, rows_per_task=sum(r['tasks']['acceptability']['rows'] for r in results.values()),
                pairs=len(pair_ids), document_split_isolation=True, artifact_checksums=True,
                exact_chat_recreation=True, edits_roundtrip=True, source_provenance=True,
                no_symlinks_or_hardlinks=True, human_precision=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--recreate-data', type=Path, help='Write identical chat shards to a NEW directory')
    parser.add_argument('--report', type=Path, help='Write a validation receipt')
    args = parser.parse_args()
    report = validate(args.root)
    if args.recreate_data:
        args.recreate_data.mkdir(parents=True, exist_ok=False)
        rebuilt = []
        for task in TASKS:
            rebuilt.extend(build_data(args.root, args.recreate_data / task, task))
        manifest = json.loads((args.root / 'metadata/manifest.json').read_text())
        if rebuilt != manifest['shards']:
            raise ValueError('Recreated shard hashes differ (use the recorded Python/zlib versions)')
        report['recreated_shards_identical'] = True
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
