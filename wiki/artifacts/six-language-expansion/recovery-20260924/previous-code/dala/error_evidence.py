"""Mine observed M2 corrections as evidence, never as automatically safe rules.

Example:
    python -m dala.error_evidence --output la_output/evidence/patterns.json FILE.m2 ...

All annotated edits are considered, including edits in multi-error sentences.
Annotators and duplicated source sentences do not multiply pattern support.
Keep learner/native files separate in reports; do not mine held-out test sets.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Correction:
    start: int
    end: int
    error_type: str
    replacement: str
    annotator: str


def read_m2(text):
    """Yield tokenized sentences and annotations with validated token offsets."""
    for block in text.replace('\r\n', '\n').strip().split('\n\n'):
        tokens, edits = None, []
        for line in block.splitlines():
            if line.startswith('S '):
                if tokens is not None:
                    raise ValueError('Multiple source sentences in one M2 block')
                tokens = line[2:].split()
            elif line.startswith('A '):
                fields = line[2:].split('|||')
                if len(fields) != 6:
                    raise ValueError('Expected six M2 annotation fields')
                start, end = map(int, fields[0].split())
                if fields[1] in {'noop', 'UNK', 'Um'}:
                    continue
                edits.append(Correction(start, end, fields[1], fields[2], fields[5]))
            elif line and not line.startswith('#'):
                raise ValueError(f'Unknown M2 line: {line[:40]}')
        if tokens is None:
            if edits:
                raise ValueError('Annotation without source sentence')
            continue
        for edit in edits:
            if not 0 <= edit.start <= edit.end <= len(tokens):
                raise ValueError(f'Invalid M2 offsets: {edit.start}, {edit.end}')
        yield tokens, edits


def nearby(a, b):
    """Flag touching/overlapping edits; even separated edits may interact."""
    return a.start <= b.end and b.start <= a.end


def mine(files, min_support=3):
    if min_support < 1:
        raise ValueError('min_support must be positive')
    patterns = {}
    file_reports = []
    for filename in files:
        path = Path(filename)
        raw = path.read_bytes()
        counts = Counter()
        sentence_count = 0
        for tokens, edits in read_m2(raw.decode('utf-8')):
            sentence_count += 1
            sid = hashlib.sha256(' '.join(tokens).encode()).hexdigest()
            counted = set()
            for edit in edits:
                wrong = ' '.join(tokens[edit.start:edit.end])
                correct = edit.replacement
                if wrong == correct:
                    continue
                key = (edit.error_type, wrong, correct)
                # Count a pattern once per sentence per file, not per annotator.
                if key not in counted:
                    counts[edit.error_type] += 1
                    counted.add(key)
                record = patterns.setdefault(key, dict(sentences=set(), files=defaultdict(set),
                    multi_error=set(), nearby_edits=set(), occurrences=[]))
                record['sentences'].add(sid)
                record['files'][str(path)].add(sid)
                others = [e for e in edits if e != edit and e.annotator == edit.annotator]
                if others:
                    record['multi_error'].add(sid)
                if any(nearby(edit, e) for e in others):
                    record['nearby_edits'].add(sid)
                if len(record['occurrences']) < 3 and not any(x['sentence_sha256'] == sid for x in record['occurrences']):
                    record['occurrences'].append(dict(file=str(path), sentence_number=sentence_count,
                        sentence_sha256=sid, start=edit.start, end=edit.end, annotator=edit.annotator))
        file_reports.append(dict(file=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                 sentences=sentence_count, pattern_instances_by_type=dict(counts)))
    output = []
    for (error_type, wrong, correct), rec in patterns.items():
        if len(rec['sentences']) < min_support:
            continue
        spelling = error_type == 'R:SPELL' and wrong.isalpha() and correct.isalpha() and wrong.lower() != correct.lower()
        output.append(dict(error_type=error_type, observed_error=wrong, observed_correction=correct,
            proposed_corruption_from=correct, proposed_corruption_to=wrong,
            distinct_sentence_support=len(rec['sentences']),
            support_by_file={k: len(v) for k, v in rec['files'].items()},
            multi_error_sentence_support=len(rec['multi_error']),
            touching_or_overlapping_edit_support=len(rec['nearby_edits']),
            single_word_spelling_candidate=spelling,
            status='unreviewed_evidence_not_a_rule', evidence_locations=rec['occurrences']))
    output.sort(key=lambda p: (-p['distinct_sentence_support'], p['error_type'], p['observed_error'], p['observed_correction']))
    return dict(min_support=min_support, support_unit='distinct tokenized source sentence; not writers or documents',
        limitations='ERRANT categories are automatic. Reversing a correction is not necessarily an error in a new context. Disjoint edits can interact.',
        files=file_reports, patterns=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--min-support', type=int, default=3)
    args = parser.parse_args()
    report = mine(args.files, args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Wrote {len(report['patterns'])} unreviewed patterns to {args.output}")


if __name__ == '__main__':
    main()
