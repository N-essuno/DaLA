"""Convert explicitly configured lexical tags without silently dropping unknowns.

UniMorph supplies descriptive lexical evidence, not a normative dictionary. Only
UD training lemmas are expanded. All retained ambiguous/incomplete analyses remain
available to the compiler's syncretism veto, even when not usable for generation.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from dala.common_pile import sha256_file


def convert(lemma, form, tags, recipe):
    tags = tags.split(';')
    features, positions, unknown = {}, set(), []
    for tag in tags:
        values = recipe['tags'].get(tag)
        if values is None:
            unknown.append(tag)
            continue
        for key, value in values.items():
            if key == 'pos': positions.add(value)
            elif key in features and features[key] != value:
                features[key] = ','.join(sorted(set(features[key].split(',')) | set(value.split(','))))
            else: features[key] = value
    if len(positions) != 1: return None
    pos = next(iter(positions))
    for condition in recipe.get('conditions', []):
        if set(condition.get('all', [])) <= set(tags) and not set(condition.get('none', [])) & set(tags):
            for key in condition.get('remove', []): features.pop(key, None)
            features.update(condition.get('set', {}))
    for key, value in recipe.get('defaults', {}).get(pos, {}).items():
        features.setdefault(key, value)
    return dict(lemma=lemma.casefold(), pos=pos, features=features, forms=[form.lower()],
                evidence_kind='unimorph_descriptive_paradigm', generation_eligible=not unknown,
                unmapped_tags=sorted(set(unknown)), raw_tags=';'.join(tags))


def prepare(recipe, base):
    language = recipe['language']
    old = base.parent / 'european' / language
    original = json.loads((old / 'morphology.json').read_text())
    lemmas = {a['lemma'] for a in original}
    folder = base / 'unimorph' / recipe['resource']
    receipt = json.loads((folder / 'receipt.json').read_text())
    rows, stats, unknown = [], Counter(), Counter()
    for record in receipt['files']:
        if record['file'] not in recipe['files']: continue
        path = folder / record['file']
        if sha256_file(path) != record['sha256']: raise ValueError('UniMorph checksum mismatch')
        for line in path.open():
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 3: stats['blank_or_nonrecord_lines'] += 1; continue
            lemma, form, tags = fields
            stats['rows'] += 1
            if lemma.casefold() not in lemmas: stats['outside_training_lemma_inventory'] += 1; continue
            if not lemma.isalpha() or not form.isalpha(): stats['not_single_alphabetic_word'] += 1; continue
            entry = convert(lemma, form, tags, recipe)
            if entry is None: stats['ambiguous_or_unknown_pos'] += 1; continue
            rows.append(entry);unknown.update(entry['unmapped_tags'])
            stats['generation_eligible' if entry['generation_eligible'] else 'veto_only_unknown_tags'] += 1
    # Deduplicate exact analyses; raw source tags remain in the lexical evidence.
    output = base / 'converted' / language
    output.mkdir(parents=True, exist_ok=True)
    (output / 'lexical-analyses.json').write_text(json.dumps(rows, ensure_ascii=False) + '\n')
    report = dict(language=language, surface_normalization='unicode_lower', recipe=recipe, upstream=receipt, counts=dict(stats),
                  unmapped_tags=dict(unknown), distinct_lemmas=len({a['lemma'] for a in rows}),
                  sha256=sha256_file(output / 'lexical-analyses.json'),
                  status='descriptive lexical evidence; not normative or human validated')
    (output / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(language, report['distinct_lemmas'], dict(stats), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('la_output/resources/european-expansion'))
    parser.add_argument('--recipes',type=Path,default=Path('config/european-expansion/morphology'))
    args=parser.parse_args()
    for path in sorted(args.recipes.glob('*.json')):
        recipe=json.loads(path.read_text())
        if path.stem != recipe['language']: raise ValueError('Lexical recipe language mismatch')
        prepare(recipe,args.base)


if __name__=='__main__':main()
