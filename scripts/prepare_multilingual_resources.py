"""Pin and fetch six language resources; record bytes and provenance, not precision.

UD training annotations supply conservative, attested inflection candidates, not
an exhaustive normative lexicon. Hunspell supplies a separate lexical guard.
Neither resource alone certifies a sentence's grammatical correctness.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

TREES = {
    'pl': ('Polish-PDB', 'pl_pdb'), 'sv': ('Swedish-Talbanken', 'sv_talbanken'),
    'nb': ('Norwegian-Bokmaal', 'no_bokmaal'), 'nn': ('Norwegian-Nynorsk', 'no_nynorsk'),
    'fo': ('Faroese-FarPaHC', 'fo_farpahc'), 'is': ('Icelandic-IcePaHC', 'is_icepahc'),
}
ROOT = Path('la_output/resources/multilingual')


def download(url, path):
    if not path.exists():
        r = requests.get(url, timeout=(20, 180)); r.raise_for_status()
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(r.content); temporary.replace(path)
    return {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'path': str(path)}


def prepare(language, dictionary_revision):
    tree, prefix = TREES[language]
    root = ROOT / language; root.mkdir(parents=True, exist_ok=True)
    pin = root / 'pins.json'
    if pin.exists():
        pins = json.loads(pin.read_text())
    else:
        r = requests.get(f'https://api.github.com/repos/UniversalDependencies/UD_{tree}/commits/r2.17', timeout=40)
        r.raise_for_status()
        pins = {'ud': r.json()['sha'], 'dictionary': dictionary_revision}
        pin.write_text(json.dumps(pins, indent=2) + '\n')
    receipts = []
    base = f'https://raw.githubusercontent.com/wooorm/dictionaries/{pins["dictionary"]}/dictionaries/{language}'
    for name in ('index.aff', 'index.dic', 'license', 'readme.md'):
        receipts.append(download(f'{base}/{name}', root / name))
    base = f'https://raw.githubusercontent.com/UniversalDependencies/UD_{tree}/{pins["ud"]}'
    for name in ('LICENSE.txt', 'README.md', f'{prefix}-ud-train.conllu'):
        receipts.append(download(f'{base}/{name}', root / name))
    forms = defaultdict(set)
    counts = Counter()
    for line in (root / f'{prefix}-ud-train.conllu').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) != 10 or not fields[0].isdigit(): continue
        _, form, lemma, pos, _, features, *_ = fields
        if not form.isalpha() or not lemma.isalpha() or pos not in {'NOUN', 'ADJ', 'VERB', 'AUX', 'DET', 'PRON'}: continue
        # Do not learn proper-name spellings as ordinary morphological forms.
        if form[0].isupper() and pos in {'NOUN', 'ADJ'}: continue
        forms[(lemma.casefold(), pos, features)].add(form.casefold())
        counts[(lemma.casefold(), pos, features, form.casefold())] += 1
    morphology = [{'lemma': lemma, 'pos': pos, 'features': dict(f.split('=', 1) for f in features.split('|') if '=' in f),
                   'forms': sorted(words), 'attestations': {w: counts[(lemma,pos,features,w)] for w in sorted(words)}}
                  for (lemma, pos, features), words in sorted(forms.items())]
    path = root / 'morphology.json'
    path.write_text(json.dumps(morphology, ensure_ascii=False, sort_keys=True) + '\n')
    record = {'language': language, 'pins': pins, 'files': receipts,
              'morphology_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'morphology_analyses': len(morphology), 'status': 'resources_fetched_not_linguistically_validated'}
    (root / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'language': language, 'analyses': len(morphology), 'pins': pins}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--languages', nargs='+', default=list(TREES), choices=TREES)
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    pin = ROOT / 'dictionary-pin.json'
    if not pin.exists():
        r = requests.get('https://api.github.com/repos/wooorm/dictionaries/commits/main', timeout=40); r.raise_for_status()
        pin.write_text(json.dumps({'revision': r.json()['sha']}) + '\n')
    revision = json.loads(pin.read_text())['revision']
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda language: prepare(language, revision), a.languages))
    Path('wiki/artifacts/six-language-expansion/resource-receipts.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__': main()
