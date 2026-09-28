"""Independently check exported artifacts, task views, labels and provenance."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .build_english import SPLITS, task_rows, validate_pairs
from .common_pile import sha256_file
from .dataset_review import load_pairs, load_jsonl


def validate(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, artifact in manifest['artifacts'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Artifact path escapes dataset')
        if not path.is_file() or sha256_file(path) != artifact['sha256'] or path.stat().st_size != artifact['bytes']:
            raise ValueError(f'Artifact checksum/size mismatch: {name}')
    pairs = load_pairs(root)
    rules = json.loads((root / 'rules.json').read_text())
    rules_by_id = {r['id']: r for r in rules['rules']}
    result = validate_pairs(pairs, rules)
    if result != manifest['verification']:
        raise ValueError('Recorded verification differs from current result')
    docs = {d['document_id']: d for d in load_jsonl(root / 'documents.jsonl')}
    for pair in pairs:
        doc = docs[pair['document_id']]
        for field in ('source_dataset', 'source_revision', 'document_sha256', 'url', 'license'):
            if doc[field] != pair[field]:
                raise ValueError(f'Source provenance mismatch: {field}')
        if pair['quality_status'] == 'checker_screened':
            evidence = pair['checker']
            if not evidence or evidence['source_hard_diagnostics'] != 0 or len(evidence['edits']) != len(pair['edits']):
                raise ValueError('Missing checker evidence')
            if any(not e['diagnostics'] for e in evidence['edits']):
                raise ValueError('Missing edit diagnostics')
        for edit in pair['edits']:
            if rules_by_id[edit['rule_id']].get('requires_lexical_screen'):
                checks = [c for c in (pair.get('checker') or {}).get('lexical_checks', [])
                          if c['rule_id'] == edit['rule_id']
                          and c.get('corrupted_start') == edit['corrupted_start']
                          and c.get('corrupted_end') == edit['corrupted_end']]
                if pair['quality_status'] == 'candidate_rule_and_dictionary_screened':
                    if (manifest['profile']['checker']['mode'] != 'morphology'
                            or not edit.get('requires_lexical_screen') or not checks
                            or any(c.get('original_recognized') is not True
                                   or c.get('replacement_recognized') is not False for c in checks)):
                        raise ValueError('Missing candidate dictionary evidence')
                    continue
                if (not edit.get('requires_lexical_screen') or not checks
                        or any(not set(manifest['profile']['checker']['dialects']).issubset(c['dialects']) or
                               not all(d['source_recognized'] and d['replacement_nonword']
                                       for d in c['dialects'].values()) for c in checks)):
                    raise ValueError('Missing productive spelling lexical evidence')
    for split in SPLITS:
        group = [p for p in pairs if p['split'] == split]
        views = ([], [], [])
        for pair in group:
            for rows in task_rows(pair, manifest.get('prompts')):
                for expected, row in zip(views, rows):
                    expected.append(row)
        for name, expected in zip(('acceptability', 'acceptability_it', 'correction_it'), views):
            actual = load_jsonl(root / split / f'{name}.jsonl')
            encode = lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
            if Counter(map(encode, actual)) != Counter(map(encode, expected)):
                raise ValueError(f'Task rows differ from canonical pairs: {split}/{name}')
        counts = Counter(r['label'] for r in views[0])
        if counts.get('correct', 0) != len(group) or counts.get('incorrect', 0) != len(group):
            raise ValueError('Unbalanced acceptability labels')
        if manifest['splits'][split]['pairs'] != len(group):
            raise ValueError('Manifest split counts differ')
    result.update(artifact_checksums=True, task_views_match_pairs=True,
                  balanced_labels=True, source_provenance_consistent=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))
