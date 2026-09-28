"""Import explicit normative Faroese GiellaLT test paradigms, including variants."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
import yaml

from scripts.prepare_multilingual_resources import download


def main():
    root = Path('la_output/resources/multilingual/fo')
    tree = json.loads(Path('wiki/artifacts/six-language-expansion/fo-giella-tree.json').read_text())
    paths = [x['path'] for x in tree['tree'] if '/gt-norm-yamls/' in x['path'] and x['path'].endswith('.yaml')]
    folder = root / 'giella'; folder.mkdir(exist_ok=True)
    base = f'https://raw.githubusercontent.com/giellalt/lang-fao/{tree["sha"]}'
    with ThreadPoolExecutor(max_workers=12) as pool:
        receipts = list(pool.map(lambda p: download(f'{base}/{p}', folder / Path(p).name), paths))
    receipts.append(download(f'{base}/LICENSE', folder / 'LICENSE'))
    tags = {'Msc': ('Gender', 'Masc'), 'Fem': ('Gender', 'Fem'), 'Neu': ('Gender', 'Neut'),
            'Sg': ('Number', 'Sing'), 'Pl': ('Number', 'Plur'), 'Nom': ('Case', 'Nom'),
            'Acc': ('Case', 'Acc'), 'Dat': ('Case', 'Dat'), 'Gen': ('Case', 'Gen'),
            'Indef': ('Definite', 'Ind'), 'Def': ('Definite', 'Def'), 'Comp': ('Degree', 'Cmp'),
            'Superl': ('Degree', 'Sup'), 'Inf': ('VerbForm', 'Inf'), 'Sup': ('VerbForm', 'Sup'),
            'PrfPtc': ('VerbForm', 'Part'), 'Prs': ('Tense', 'Pres'), 'Prt': ('Tense', 'Past'),
            'Ind': ('Mood', 'Ind')}
    analyses = []
    for record in receipts:
        if not record['path'].endswith('.yaml'): continue
        data = yaml.safe_load(Path(record['path']).read_text())
        for paradigm in (data or {}).get('Tests', {}).values():
            if not isinstance(paradigm, dict): continue
            for analysis, forms in paradigm.items():
                parts = analysis.split('+'); lemma = parts[0]
                if len(parts) < 2 or parts[1] not in {'A', 'N', 'V'}: continue
                features = dict(tags[t] for t in parts[2:] if t in tags)
                for t in parts[2:]:
                    if len(t) == 3 and t[0] in '123' and t[1:] in {'Sg', 'Pl'}:
                        features.update(Person=t[0], Number='Sing' if t[1:] == 'Sg' else 'Plur')
                if 'Mood' in features: features['VerbForm'] = 'Fin'
                if parts[1] == 'A': features.setdefault('Degree', 'Pos')
                words = forms if isinstance(forms, list) else [forms]
                words = [w for w in words if isinstance(w, str) and w.isalpha() and w.islower()]
                if words: analyses.append(dict(lemma=lemma, pos={'A': 'ADJ', 'N': 'NOUN', 'V': 'VERB'}[parts[1]], features=features, forms=words))
    path = root / 'morphology.json'
    path.write_text(json.dumps(analyses, ensure_ascii=False, sort_keys=True) + '\n')
    receipt_path = root / 'receipt.json'; receipt = json.loads(receipt_path.read_text())
    receipt.update(morphology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), morphology_analyses=len(analyses),
                   morphology_source='GiellaLT normative paradigm test expectations', giella_revision=tree['sha'], giella_files=receipts)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'analyses': len(analyses), 'lemmas': len({a['lemma'] for a in analyses}), 'revision': tree['sha']}))


if __name__ == '__main__': main()
