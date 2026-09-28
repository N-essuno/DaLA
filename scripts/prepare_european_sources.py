"""Prepare pinned clean-source candidates; no generation or linguistic certification.

Input recipes select formats and sources. All languages share normalization code.
The original downloads remain intact; this creates a bounded pilot document pool.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from dala.common_pile import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'la_output/resources/european-expansion'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def wikipedia(recipe):
    import pyarrow.parquet as pq
    receipt = json.loads((BASE / recipe['receipt']).read_text())
    if receipt['language'] != recipe['language']:
        raise ValueError('Source recipe language mismatch')
    if sha256_file(receipt['path']) != receipt['sha256']:
        raise ValueError('Wikipedia snapshot checksum mismatch')
    for batch in pq.ParquetFile(receipt['path']).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            yield dict(source_document_id=recipe['language'] + ':' + str(row['id']),
                       text=row['text'], title=row['title'], url=row['url'],
                       language=recipe['language'], source_file=receipt['file'])


def europarl(recipe):
    archive = BASE / recipe['archive']
    if sha256_file(archive) != recipe['sha256']:
        raise ValueError('Europarl archive checksum mismatch')
    # Multiple chapters of one sitting stay together. Dates are stable IDs;
    # this also avoids assigning different speeches from one debate to splits.
    groups = defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.xml'): continue
            match = re.search(r'ep-(\d\d-\d\d-\d\d)', name)
            if not match: raise ValueError('Missing sitting date')
            date = match.group(1)
            if any(date.startswith(prefix) for prefix in recipe.get('exclude_date_prefixes', [])):
                continue
            root = ET.fromstring(z.read(name))
            paragraphs = [' '.join(''.join(s.itertext()).strip() for s in p.findall('s'))
                          for p in root.iter('P')]
            text = '\n'.join(p for p in paragraphs if p)
            # Select bounded complete chapters, not chopped sentences. All
            # retained chapters from a sitting share its document/split ID.
            if recipe['min_chars'] <= len(text) <= recipe['max_chars']:
                groups[date].append((name, text))
    for date, chapters in groups.items():
        # A pilot samples one complete chapter per sitting; full archives remain.
        name, text = min(chapters, key=lambda x: hashlib.sha256(x[0].encode()).digest())
        yield dict(source_document_id=date, text=text, language=recipe['language'],
                   url=recipe['url'], source_file=name,
                   source_document_scope='one_complete_chapter_grouped_by_sitting_date')


def prepare(recipe, output, limit, target_book=None, target_families=()):
    generators = {'wikipedia_parquet': wikipedia, 'europarl_xml_zip': europarl}
    rows, counts = [], Counter()
    for row in generators[recipe['format']](recipe):
        counts['documents_scanned'] += 1
        if not recipe['min_chars'] <= len(row['text']) <= recipe['max_chars']:
            counts['outside_pilot_length_window'] += 1
            continue
        rows.append(row)
    rows.sort(key=lambda r: hashlib.sha256(r['source_document_id'].encode()).digest())
    if target_book:
        with gzip.open(target_book, 'rt') as f: book=json.load(f)
        if book['language']!=recipe['language']:raise ValueError('Target rulebook language mismatch')
        words=defaultdict(set)
        for rule in book['rules']:
            if rule['family'] not in target_families:continue
            for word in rule.get('mappings', {}):words[word].add(rule['family'])
        subject_words={r['family']:set(r.get('subjects', [])) for r in book['rules'] if r.get('context')=='finite_agreement' and r.get('feature')=='Person'}
        groups=defaultdict(list)
        for row in rows:
            tokens=set(re.findall(r'[^\W\d_]+',row['text'].casefold()))
            matched=set().union(*(words.get(w, set()) for w in tokens))
            matched={f for f in matched if f not in subject_words or subject_words[f] & tokens}
            # Retrieval only; syntax, lexical ambiguity and source checking still
            # happen in the normal pipeline. Preserve the full document.
            for family in matched:groups[family].append(row)
        selected=[];seen=set();positions={f:0 for f in groups}
        while len(selected)<limit:
            added=False
            for family in sorted(groups):
                group=groups[family];i=positions[family]
                while i<len(group) and group[i]['source_document_id'] in seen:i+=1
                if i<len(group):
                    row=group[i];selected.append(row);seen.add(row['source_document_id']);added=True;i+=1
                    if len(selected)>=limit:break
                positions[family]=i
            if not added:break
        selected += [r for r in rows if r['source_document_id'] not in seen][:max(0,limit-len(selected))]
        counts.update({'retrieval_matches_'+f:len(v) for f,v in groups.items()})
        rows=selected
    else:
        rows = rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    data = output / 'documents.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    source = {k: recipe[k] for k in ['name', 'dataset', 'revision', 'language', 'license']}
    source.update(path=str(data), sha256=sha256_file(data), preparation=recipe,
                  candidate_source_not_clean_gold=True)
    write(output / 'sources.json', dict(sources=[source]))
    counts.update(selected_documents=len(rows), selected_characters=sum(len(r['text']) for r in rows))
    write(output / 'source-preparation.json', dict(counts=dict(counts), recipe=recipe,
        output_sha256=source['sha256'], target_rulebook_sha256=sha256_file(target_book) if target_book else None, target_families=list(target_families), quality='source candidates; sentence screening still required'))
    profile = json.loads((ROOT / 'config/languages' / (recipe['language'] + '.json')).read_text())
    settings = json.loads((ROOT / 'la_output/resources/european' / recipe['language'] / 'live-parser-settings.json').read_text())
    settings['directory'] = str((ROOT / settings['directory']).resolve())
    profile.update(parser_backend='stanza', parser='stanza', parser_settings=settings,
                   parser_threads=2, parser_expand_mwt=True, sources={'adapter': 'canonical_jsonl'},
                   sources_config=str(output / 'sources.json'))
    profile['description'] = 'CPU live-parser source-expansion diagnostic; original conservative UD morphology retained. Candidate only, no native validation.'
    write(output / 'profile.json', profile)
    print(recipe['language'], dict(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, default=ROOT / 'config/european-expansion/sources.json')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--languages', nargs='+')
    parser.add_argument('--run-id', default='sources_v1')
    parser.add_argument('--target-profile-root',type=Path)
    parser.add_argument('--gap-report',type=Path,default=ROOT/'wiki/artifacts/european-expansion/gap-diagnosis.json')
    args = parser.parse_args()
    if args.limit < 1: parser.error('limit must be positive')
    for recipe in json.loads(args.recipes.read_text())['sources']:
        if args.languages and recipe['language'] not in args.languages: continue
        output = BASE / args.run_id / recipe['language']
        if output.exists(): raise FileExistsError(output)
        profile=json.loads((args.target_profile_root/recipe['language']/'profile.json').read_text()) if args.target_profile_root else {}
        families=[r['family'] for r in json.loads(args.gap_report.read_text())['languages'][recipe['language']]] if profile else []
        prepare(recipe, output, args.limit, profile.get('rulebook'), families)


if __name__ == '__main__': main()
