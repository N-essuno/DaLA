"""Create an auditable, deterministic subset with per-split family-group caps."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from dala.common_pile import sha256_file
from dala.dataset_review import load_pairs, load_jsonl
from dala.pair_pipeline import export_dataset, SPLITS
from dala.validate_dataset import validate


def balanced_subset(pairs, config, seed=4242):
    groups = config['groups']
    family_group = {}
    limits = {}
    family_priority = {}
    for group in groups:
        name = group['name']
        if name in limits or name == 'other':
            raise ValueError('Group names must be unique and cannot be other')
        limits[name] = Fraction(str(group['max_fraction']))
        if not 0 < limits[name] <= 1:
            raise ValueError('Group caps must be in (0, 1]')
        for family in group['families']:
            if family in family_group:
                raise ValueError('Family appears in multiple groups')
            family_group[family] = name
        priority = group.get('priority_families', [])
        if not set(priority) <= set(group['families']):
            raise ValueError('Priority family is outside its group')
        family_priority[name] = {family: i for i, family in enumerate(priority)}
    limits['other'] = Fraction(1)
    selected, reports = [], {}
    for split in SPLITS:
        buckets = defaultdict(list)
        for pair in pairs:
            if pair['split'] != split:
                continue
            if len(pair['edits']) != 1:
                raise ValueError('Balancing requires single-edit pairs')
            family = pair['edits'][0]['corruption_type']
            buckets[family_group.get(family, 'other')].append(pair)
        counts = {name: len(buckets[name]) for name in limits}
        if not sum(counts.values()):
            reports[split] = dict(before=counts, after=counts, pairs=0)
            continue
        # Exact integer feasibility; a descending scan avoids assuming that
        # rounding preserves monotonic feasibility at very small sample sizes.
        for target in range(sum(counts.values()), 0, -1):
            quotas = {name: min(n, int(limits[name] * target)) for name, n in counts.items()}
            if sum(quotas.values()) >= target:
                break
        else:
            raise ValueError(f'No nonempty subset satisfies caps in {split}')
        # Preserve uncapped, scarce families first. Any excess consists only of
        # capped groups; truncate deterministically without changing any pair.
        remaining = target
        taken = {}
        for name in ['other', *[g['name'] for g in groups]]:
            priority = family_priority.get(name, {})
            members = sorted(buckets[name], key=lambda p: (
                priority.get(p['edits'][0]['corruption_type'], len(priority)),
                hashlib.sha256(f"{seed}\0{p['pair_id']}".encode()).digest()))
            n = min(quotas[name], remaining)
            selected.extend(members[:n]); taken[name] = n; remaining -= n
        assert remaining == 0
        assert all(Fraction(taken[g], target) <= limits[g] for g in taken)
        reports[split] = dict(before=counts, after=taken, pairs=target)
    return selected, reports


def build(source, destination, config_path, seed=4242, exclusions_path=None):
    source, destination, config_path = map(Path, (source, destination, config_path))
    parent_validation = validate(source)
    pairs = load_pairs(source)
    excluded = set()
    if exclusions_path:
        excluded = {r['sentence_sha256'] for r in json.loads(Path(exclusions_path).read_text())}
    eligible = [p for p in pairs if hashlib.sha256(p['original'].encode()).hexdigest() not in excluded]
    config = json.loads(config_path.read_text())
    selected, reports = balanced_subset(eligible, config, seed)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['name'] = config['name']
    manifest['quality_status'] = 'checker_screened_balanced_subset_not_human_gold'
    manifest['parent_build_counts'] = manifest.pop('counts', {})
    manifest['parent_build_rejections'] = manifest.pop('rejections', {})
    manifest['counts'] = dict(parent_pairs=len(pairs), eligible_pairs=len(eligible), selected_pairs=len(selected))
    manifest['rejections'] = dict(review_exclusion=len(pairs)-len(eligible), balancing=len(eligible)-len(selected))
    manifest['subset_selection'] = dict(method='Maximum-size single-edit subset under per-split family-group caps; explicit family priorities then hash-ranked within groups',
        seed=seed, config=config, config_sha256=sha256_file(config_path),
        implementation_sha256=sha256_file(__file__), parent_manifest_sha256=sha256_file(source/'manifest.json'),
        parent_validation=parent_validation, split_selection=reports,
        review_exclusions_sha256=sha256_file(exclusions_path) if exclusions_path else None)
    manifest.pop('artifacts', None)
    for key in ('selected_by_type', 'eligible_by_type', 'target_pairs'):
        if key in manifest:
            manifest['parent_'+key] = manifest.pop(key)
    result = export_dataset(selected, destination, manifest,
                            json.loads((source/'rules.json').read_text()),
                            load_jsonl(source/'documents.jsonl'), seed)
    verification = validate(destination)
    print(json.dumps(dict(pairs=len(selected), families=dict(Counter(e['corruption_type'] for p in selected for e in p['edits'])),
                         selection=reports, validation=verification), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', default=4242, type=int)
    parser.add_argument('--exclusions', type=Path)
    args = parser.parse_args()
    build(args.source, args.destination, args.config, args.seed, args.exclusions)
