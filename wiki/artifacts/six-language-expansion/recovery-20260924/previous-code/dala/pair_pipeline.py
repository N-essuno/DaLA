"""Shared provenance-preserving pair pipeline, configured by a language pack."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import tempfile
import time

from .common_pile import DEFAULT_SOURCES, documents, sha256_file, snapshots
from .curation import NearDuplicates
from .edits import apply_edits
from .profiles import load_profile, resource
import importlib

def load_adapter(profile):
    module, name = profile['adapter'].split(':')
    return getattr(importlib.import_module(module), name)(profile)


SPLITS = ('train', 'validation', 'test')


def split_for(document_id, seed=4242, proportions=None):
    bucket = int.from_bytes(hashlib.sha256(f'{seed}\0{document_id}'.encode()).digest()[:8], 'big') % 100
    proportions = proportions or {'train_percent': 80, 'validation_percent': 10}
    train = proportions['train_percent']
    validation = proportions['validation_percent']
    if not 0 < train < train + validation < 100:
        raise ValueError('Invalid document split proportions')
    return 'train' if bucket < train else 'validation' if bucket < train + validation else 'test'


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def task_rows(pair, prompts=None):
    """Two acceptability and two correction rows, including identity correction."""
    prompts = prompts or load_profile(pair.get('language', 'en'))['prompts']
    families = sorted({e['corruption_type'] for e in pair['edits']})
    for clean in (True, False):
        edits = [] if clean else pair['edits']
        types = [] if clean else families
        corruption_type = None if clean else types[0] if len(types) == 1 else 'multiple'
        text = pair['original'] if clean else pair['corrupted']
        metadata = dict(pair_id=pair['pair_id'], document_id=pair['document_id'], language=pair.get('language', 'en'),
                        error_count=len(edits), corruption_types=types)
        raw = dict(text=text, corruption_type=corruption_type,
                   label='correct' if clean else 'incorrect', **metadata)
        sample = dict(content=text, corruption_type=corruption_type, corruption_types=types,
                      error_count=len(edits))
        accept = dict(direction=prompts['acceptability'],
                      samples=dict(sample, response='yes' if clean else 'no'), **metadata)
        correction = dict(direction=prompts['correction'], samples=dict(sample,
            response=pair['original'], affected_token_1=edits[0]['original'] if len(edits) == 1 else None,
            affected_token_2=edits[0]['replacement'] if len(edits) == 1 else None,
            edits=edits), **metadata)
        yield raw, accept, correction


def validate_pairs(pairs, rulebook):
    """Mechanical validation only; no human grammatical-precision claim."""
    from .edits import Corruption
    rules = {r['id']: r for r in rulebook['rules']}
    documents_by_split, seen, pair_ids = {}, set(), set()
    for pair in pairs:
        if pair['pair_id'] in pair_ids:
            raise ValueError('Duplicate pair ID')
        pair_ids.add(pair['pair_id'])
        if pair['split'] not in SPLITS:
            raise ValueError('Unknown split')
        if documents_by_split.setdefault(pair['document_id'], pair['split']) != pair['split']:
            raise ValueError('Document crosses splits')
        if not pair['edits'] or pair['original'] == pair['corrupted']:
            raise ValueError('Empty or identity corruption')
        edits = []
        grammar = 0
        protected = set()
        for record in pair['edits']:
            fields = {k: record[k] for k in Corruption.__dataclass_fields__}
            edit = Corruption(**fields)
            r = rules[edit.rule_id]
            if r.get('operator') in {'dictionary_inflection', 'dictionary_mapping', 'character_edit', 'guarded_token'}:
                module, function = rulebook['edit_validator'].split(':')
                valid = getattr(importlib.import_module(module), function)(r, edit)
            else:
                valid = r['correct'] == edit.original.lower() and r['incorrect'] == edit.replacement.lower()
            if r['family'] != edit.corruption_type or not valid:
                raise ValueError('Edit differs from licensed rule')
            grammar += edit.corruption_type != 'spelling'
            if protected.intersection(edit.protected_tokens):
                raise ValueError('Edits interact through a protected token')
            protected.update(edit.protected_tokens)
            edits.append(edit)
        if grammar > 1:
            raise ValueError('Multiple grammar edits are not licensed')
        if apply_edits(pair['original'], edits) != pair['corrupted']:
            raise ValueError('Corruption reconstruction failed')
        restored = pair['corrupted']
        for e in sorted(pair['edits'], key=lambda e: e['corrupted_start'], reverse=True):
            if restored[e['corrupted_start']:e['corrupted_end']] != e['replacement']:
                raise ValueError('Corrupted offsets do not match')
            restored = restored[:e['corrupted_start']] + e['original'] + restored[e['corrupted_end']:]
        if restored != pair['original']:
            raise ValueError('Correction roundtrip failed')
        for text in (pair['original'], pair['corrupted']):
            if text in seen:
                raise ValueError('Repeated exported sentence within or across splits')
            seen.add(text)
    return dict(pairs=len(pairs), unique_documents=len(documents_by_split),
                exact_edit_reconstruction=True, correction_roundtrip=True,
                document_split_isolation=True, exported_text_unique=True,
                at_most_one_grammar_edit=True)


def export_dataset(pairs, destination, manifest, rulebook, document_records, seed=4242):
    """Create a complete, validated dataset atomically; never mix with old output."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}; choose a new output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest['verification'] = validate_pairs(pairs, rulebook)
    manifest['splits'] = {}
    if not pairs:
        raise ValueError('No eligible pairs; inspect source filters and rule coverage')
    with tempfile.TemporaryDirectory(prefix=destination.name + '.', dir=destination.parent) as temporary:
        root = Path(temporary)
        reviews = []
        for split in SPLITS:
            group = [p for p in pairs if p['split'] == split]
            random.Random(seed).shuffle(group)
            folder = root / split
            folder.mkdir()
            write_jsonl(folder / 'pairs.jsonl', group)
            raw, acceptability, correction = [], [], []
            for pair in group:
                for a, b, c in task_rows(pair, manifest.get('prompts')):
                    raw.append(a); acceptability.append(b); correction.append(c)
            for name, rows in [('acceptability', raw), ('acceptability_it', acceptability), ('correction_it', correction)]:
                random.Random(seed).shuffle(rows)
                write_jsonl(folder / f'{name}.jsonl', rows)
            # Readable raw CSV alongside instruction-shaped JSONL files.
            with (folder / 'acceptability.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['text', 'corruption_type', 'label', 'pair_id', 'document_id', 'language', 'error_count', 'corruption_types'])
                writer.writeheader(); writer.writerows(raw)
            family_counts = Counter(e['corruption_type'] for p in group for e in p['edits'])
            manifest['splits'][split] = dict(pairs=len(group), acceptability_rows=len(raw),
                correction_rows=len(correction), edits_by_type=dict(sorted(family_counts.items())),
                pairs_by_error_count=dict(sorted(Counter(len(p['edits']) for p in group).items())),
                pairs_by_source=dict(sorted(Counter(p['source_name'] for p in group).items())))
            strata = {}
            for pair in group:
                for family in {e['corruption_type'] for e in pair['edits']}:
                    strata.setdefault((family, len(pair['edits'])), []).append(pair)
            chosen = {}
            for members in strata.values():
                for p in random.Random(seed).sample(members, min(30, len(members))):
                    chosen[p['pair_id']] = p
            for p in chosen.values():
                reviews.append(dict(pair_id=p['pair_id'], split=split, source_url=p['url'],
                    original=p['original'], corrupted=p['corrupted'],
                    edits=json.dumps(p['edits'], ensure_ascii=False),
                    original_correct='', corrupted_incorrect='', edits_valid='', reviewer='', notes=''))
        with (root / 'review.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['pair_id', 'split', 'source_url', 'original', 'corrupted', 'edits',
                                                       'original_correct', 'corrupted_incorrect', 'edits_valid', 'reviewer', 'notes'])
            writer.writeheader(); writer.writerows(reviews)
        used_ids = {p['document_id'] for p in pairs}
        write_jsonl(root / 'documents.jsonl', (d for d in document_records if d['document_id'] in used_ids))
        exported_book = rulebook
        if manifest.get('profile', {}).get('export_used_rule_mappings'):
            used = {}
            for pair in pairs:
                for edit in pair['edits']:
                    used.setdefault(edit['rule_id'], set()).add((edit['original'].casefold(), edit['replacement'].casefold()))
            exported_book = dict(rulebook, rules=[])
            for rule in rulebook['rules']:
                subset = dict(rule)
                if 'mappings' in rule:
                    selected = used.get(rule['id'], set())
                    subset['mappings'] = {word:[v for v in values if (word,v) in selected]
                                          for word,values in rule['mappings'].items() if any((word,v) in selected for v in values)}
                    subset['pairs'] = {word:[p for p in rule['pairs'][word] if (word,p['replacement']) in selected]
                                       for word in subset['mappings']}
                    subset['export_scope'] = 'selected_surface_substitutions_only; full inventory pinned by manifest rulebook_sha256'
                exported_book['rules'].append(subset)
        serialized = (json.dumps(exported_book, ensure_ascii=False, separators=(',',':'))
                      if manifest.get('profile', {}).get('export_used_rule_mappings') else json.dumps(exported_book, indent=2))
        (root / 'rules.json').write_text(serialized + '\n')
        (root / 'README.md').write_text(dataset_card(manifest), encoding='utf-8')
        manifest['artifacts'] = {str(p.relative_to(root)): dict(sha256=sha256_file(p), bytes=p.stat().st_size)
                                 for p in sorted(root.rglob('*')) if p.is_file()}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        root.rename(destination)
    return manifest


def dataset_card(manifest):
    n = manifest['verification']['pairs']
    screening = ('Dictionary and generator-constraint checks only. There is no independent\n'
                 'grammar-checker verdict; these candidates require a linguistic audit.'
                 if manifest['quality_status'] == 'candidate_rule_and_dictionary_screened' else
                 'Checker-screened builds require no relevant source diagnostics and a diagnostic at every injected edit.')
    return f"""# {manifest.get('name', 'DaLA')} — grammar and spelling

{n:,} original/corrupted pairs. Status: **{manifest['quality_status']}**.
No human linguistic precision measurement is implied. {screening}

{manifest.get('source_description', 'See document metadata for source licenses and attribution.')}

Each split provides canonical pairs, raw acceptability CSV/JSONL and instruction
acceptability/correction JSONL. Both tasks include clean controls. Source URLs,
licenses, revisions, offsets and error evidence accompany the data. Rule and
word frequencies are not natural error-frequency estimates.

Documents receive seeded train/validation/test assignments before selection.
Exact and heuristic near-duplicate filters reduce leakage. Up to
{manifest.get('max_errors', 2)} independent edits are permitted, at most one grammar
edit. Correct input maps to itself in correction. No paraphrasing or style task
is included. `rules.json` records the provenance and generation method of each rule.

`review.csv` samples by split/family/error count. The review command requires
explicit positive judgments for the source, corruption and intended edits.
See `manifest.json` for resources, counts and checksums. Automatic screening can
miss errors; it is not human validation.
"""


def prepare_sentence(sent, original_doc, paragraph_start, adapter, book, profile, seed, max_errors,
                     misspellings, rules_by_id):
    """Shared candidate construction for ordinary and checkpointed builds."""
    reason = adapter.sentence_rejection(sent, misspellings)
    if reason:
        return None, reason, set()
    edits = adapter.candidates(sent, book)
    eligible = {e.corruption_type for e in edits}
    if not edits:
        return None, 'no_licensed_corruption', eligible
    chosen = adapter.select_edits(sent.text, edits, seed, max_errors)
    corrupted = apply_edits(sent.text, chosen)
    start = paragraph_start + sent.start_char
    end = paragraph_start + sent.end_char
    if original_doc['text'][start:end] != sent.text:
        raise ValueError('Sentence offsets do not match original document')
    edit_records, delta = [], 0
    for e in chosen:
        record = asdict(e)
        if rules_by_id[e.rule_id].get('requires_lexical_screen'):
            record['requires_lexical_screen'] = True
        record.update(corrupted_start=e.start + delta, corrupted_end=e.start + delta + len(e.replacement))
        delta += len(e.replacement) - len(e.original)
        edit_records.append(record)
    named = [(max(ent.start_char, sent.start_char) - sent.start_char,
              min(ent.end_char, sent.end_char) - sent.start_char)
             for ent in sent.doc.ents if ent.start < sent.end and ent.end > sent.start]
    pair = {k: v for k, v in original_doc.items() if k != 'text' and not k.startswith('_')}
    pair.update(pair_id=hashlib.sha256(f"{original_doc['document_id']}\0{start}\0{corrupted}".encode()).hexdigest()[:24],
        split=split_for(original_doc['document_id'], seed, profile.get('splits')), language=profile['language'], sentence_start=start, sentence_end=end,
        original=sent.text, corrupted=corrupted, edits=edit_records,
        quality_status='pending_screening', checker=None)
    return (pair, named), None, eligible


def source_functions(profile):
    if profile['sources']['adapter'] == 'common_pile':
        return snapshots, documents
    if profile['sources']['adapter'] == 'dynaword':
        from .dynaword import snapshots as snap, documents as docs
        return snap, docs
    raise ValueError('Unsupported source adapter')


def order_documents(docs, seed, profile):
    ordered = sorted(docs, key=lambda d: hashlib.sha256(f"{seed}\0{d['document_id']}".encode()).digest())
    if profile.get('source_sampling') != 'round_robin':
        return ordered
    from itertools import zip_longest
    groups = {}
    for d in ordered:
        groups.setdefault(d['source_name'], []).append(d)
    weights = profile.get('source_sampling_weights')
    if weights:
        if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in weights.values()):
            raise ValueError('Source sampling weights must be positive integers')
        result = []
        rounds = max(((len(g) + weights.get(k, 1) - 1) // weights.get(k, 1)
                      for k, g in groups.items()), default=0)
        for i in range(rounds):
            for k in sorted(groups):
                w = weights.get(k, 1)
                result.extend(groups[k][i*w:(i+1)*w])
        return result
    return [d for row in zip_longest(*(groups[k] for k in sorted(groups))) for d in row if d is not None]


def _build(output_dir='la_output/english_common_pile', seed=4242, max_errors=2,
          sources_path=None, rulebook_path=None,
          cache_dir='la_output/cache', model=None, max_documents=None, offline=False, checker=None, profile=None):
    import spacy
    code_dir = Path(__file__).parent
    code_at_start = {str(p.relative_to(code_dir)): sha256_file(p) for p in sorted(code_dir.rglob('*.py'))}
    profile = load_profile('en') if profile is None else profile
    source_snapshots, source_documents = source_functions(profile)
    adapter = load_adapter(profile)
    model = model or profile['parser']
    if sources_path is None:
        sources_path = resource(profile, 'sources_config')
    if rulebook_path is None:
        rulebook_path = resource(profile, 'rulebook')
    input_paths={'profile_sha256':Path(profile['_path']),'sources_config_sha256':Path(sources_path),
                 'rulebook_sha256':Path(rulebook_path),'source_exclusions_sha256':resource(profile,'exclusions')}
    input_hashes={k:sha256_file(p) for k,p in input_paths.items()}
    if not 1 <= max_errors <= 3:
        raise ValueError('max_errors must be 1, 2 or 3')
    if max_documents is not None and max_documents < 1:
        raise ValueError('max_documents must be positive')
    if Path(output_dir).exists():
        raise FileExistsError(f'Output already exists: {output_dir}')
    from .parsing import load_parser
    nlp = load_parser(profile, model)
    book = adapter.load_rulebook(rulebook_path)
    if checker is None and any(r.get('requires_lexical_screen') for r in book['rules']):
        raise ValueError('Productive spelling requires the local lexical/context checker')
    cached = source_snapshots(sources_path, cache_dir, offline)
    source_docs = list(source_documents(cached))
    # Deterministic interleaving keeps bounded smoke runs representative of both sources.
    source_docs = order_documents(source_docs, seed, profile)
    if max_documents is not None:
        source_docs = source_docs[:max_documents]
    metadata = [{k: v for k, v in d.items() if k != 'text'} for d in source_docs]
    by_id = {d['document_id']: d for d in source_docs}
    rejected, eligible, selected = Counter(), Counter(), Counter()
    counts = Counter(documents=len(source_docs))
    seen, near, pairs, pending = set(), NearDuplicates(), [], []
    misspellings = {r['incorrect'] for r in book['rules'] if r['family'] == 'spelling' and 'incorrect' in r}
    rules_by_id = {r['id']: r for r in book['rules']}

    def inputs():
        for source_doc in source_docs:
            for text, offset in adapter.paragraphs(source_doc):
                counts['paragraphs'] += 1
                yield text, (source_doc['document_id'], offset)

    last_progress = time.monotonic()
    for parsed, (doc_id, paragraph_start) in nlp.pipe(inputs(), as_tuples=True, batch_size=64):
        original_doc = by_id[doc_id]
        for sent in parsed.sents:
            counts['sentences'] += 1
            item, reason, families = prepare_sentence(sent, original_doc, paragraph_start, adapter,
                                                       book, profile, seed, max_errors, misspellings, rules_by_id)
            eligible.update(families)
            if reason:
                rejected[reason] += 1
                continue
            item[0]['quality_status'] = getattr(checker, 'quality_status', 'checker_screened') if checker else 'automatic_screening_only'
            pending.append(item)
        if time.monotonic() - last_progress >= 30:
            print(f"Parsed {counts['sentences']:,} sentences; prepared {len(pending):,} candidate pairs", flush=True)
            last_progress = time.monotonic()
    print(f"Screening {len(pending):,} candidate pairs from {counts['sentences']:,} parsed sentences", flush=True)
    rejected.update(getattr(nlp, 'rejections', {}))

    def screen_pair(item):
        pair, named = item
        if checker is None:
            return pair, None
        check, reason = checker.screen(pair['original'], pair['corrupted'], pair['edits'], named)
        pair['checker'] = check
        return pair, reason

    last_progress = time.monotonic()
    with ThreadPoolExecutor(max_workers=profile.get('checker', {}).get('workers', 8) if checker else 1) as executor:
        for checked, (pair, reason) in enumerate(executor.map(screen_pair, pending), 1):
            if time.monotonic() - last_progress >= 30:
                print(f"Screened {checked:,}/{len(pending):,} candidates; retained {len(pairs):,} pairs", flush=True)
                last_progress = time.monotonic()
            if reason:
                rejected[reason] += 1
                continue
            if pair['original'] in seen or pair['corrupted'] in seen:
                rejected['text_collision'] += 1
                continue
            if not near.add(pair['original']):
                rejected['near_duplicate'] += 1
                continue
            pairs.append(pair)
            seen.update((pair['original'], pair['corrupted']))
            selected.update(e['corruption_type'] for e in pair['edits'])
    if max_documents is None and any(not any(p['split'] == s for p in pairs) for s in SPLITS):
        raise ValueError('Full build has an empty split')
    code_dir = Path(__file__).parent
    manifest = dict(schema_version='2', language=profile['language'], name=profile['name'],
        prompts=profile['prompts'], source_description=profile.get('description', ''),
        profile={k:v for k,v in profile.items() if k != '_path'},
        profile_sha256=input_hashes['profile_sha256'], quality_status=getattr(checker, 'quality_status', 'checker_screened') if checker else 'automatic_screening_only',
        language_checker=checker.software if checker else None,
        checker_runtime=getattr(checker, 'runtime', None),
        seed=seed, max_errors=max_errors, max_documents=max_documents,
        source_snapshots=[r for _, _, r in cached], sources_config_sha256=input_hashes['sources_config_sha256'],
        rulebook_sha256=input_hashes['rulebook_sha256'],
        source_exclusions_sha256=input_hashes['source_exclusions_sha256'],
        inputs_changed_during_build=any(input_hashes[k]!=sha256_file(p) for k,p in input_paths.items()),
        parser=dict(package=model, version=nlp.meta.get('version'), spacy=spacy.__version__),
        runtime=dict(python=platform.python_version(), **{name:version(name) for name in profile.get('dependencies', ['requests'])}),
        code_sha256=code_at_start,
        code_changed_during_build=code_at_start != {str(p.relative_to(code_dir)): sha256_file(p) for p in sorted(code_dir.rglob('*.py'))},
        counts=dict(counts), rejections=dict(sorted(rejected.items())),
        eligible_sentences_by_type=dict(eligible), selected_edits_by_type=dict(selected))
    report = export_dataset(pairs, output_dir, manifest, book, metadata, seed)
    print(json.dumps(dict(output=str(output_dir), pairs=len(pairs), splits=report['splits']), indent=2))
    return report


def build(*args, checker_mode='local', tools_dir='la_output/tools', **kwargs):
    if checker_mode == 'morphology':
        from .morphology_check import MorphologyCheck
        checker = MorphologyCheck(kwargs['profile'])
        try:
            if kwargs['profile'].get('build', {}).get('checkpoint_dir'):
                from .batch_pipeline import build_checkpointed
                return build_checkpointed(*args, checker=checker, **kwargs)
            return _build(*args, checker=checker, **kwargs)
        finally:
            checker.close()
    if checker_mode == 'off':
        return _build(*args, **kwargs)
    if checker_mode != 'local':
        raise ValueError('checker_mode must be local or off')
    from .language_check import LanguageCheck, local_servers
    cache = Path(kwargs.get('cache_dir', args[5] if len(args) > 5 else 'la_output/cache')) / 'languagetool.sqlite3'
    config = kwargs.get('profile', load_profile('en'))['checker']
    with local_servers(tools_dir, instances=config.get('server_instances', 1),
                       threads=config.get('server_threads', 8), heap=config.get('heap', '2g')) as url:
        checker = LanguageCheck(url, cache, dialects=config['dialects'],
                                commit_batch_size=config.get('cache_commit_batch_size', 1),
                                lexical_carrier=config.get('lexical_carrier', 'The word is '),
                                hard_rule_ids=config.get('hard_rule_ids', ()))
        checker.runtime = json.loads((Path(tools_dir) / 'runtime.json').read_text())
        try:
            if kwargs.get('profile', {}).get('build', {}).get('checkpoint_dir'):
                from .batch_pipeline import build_checkpointed
                return build_checkpointed(*args, checker=checker, **kwargs)
            return _build(*args, checker=checker, **kwargs)
        finally:
            checker.close()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='la_output/english_common_pile')
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--max-errors', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--rulebook', type=Path)
    parser.add_argument('--cache-dir', default='la_output/cache')
    parser.add_argument('--model')
    parser.add_argument('--profile', default='en')
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--checker', choices=['local', 'off'], default='local', help='Use off only for unchecked diagnostic builds')
    parser.add_argument('--tools-dir', default='la_output/tools')
    args = parser.parse_args()
    build(args.output_dir, args.seed, args.max_errors, args.sources, args.rulebook,
          args.cache_dir, args.model, args.max_documents, args.offline,
          checker_mode=args.checker, tools_dir=args.tools_dir, profile=load_profile(args.profile))


if __name__ == '__main__':
    cli()
