"""Convert pinned nominal lexicons using language-owned, exact tag tables.

Only existing UD training lemma/POS combinations may expand. Unsupported tags
remain veto-only analyses. Every generated form still requires the independent
language/standard dictionary; this does not certify mixed-variety upstream data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from dala.common_pile import sha256_file
from scripts.prepare_european_sources import write

BASE=Path('la_output/resources/european-expansion')

def convert(fields, recipe, inventory):
    if len(fields) <= max(recipe['columns'].values()): return None
    form, lemma, tag = (fields[recipe['columns'][key]] for key in ['form','lemma','tag'])
    pos = recipe.get('fixed_pos') or recipe['positions'].get(fields[recipe['columns']['pos']])
    if not pos or not form.isalpha() or not lemma.isalpha() or form != form.lower():return None
    if (lemma.casefold(),pos) not in inventory:return None
    features=recipe['tags'].get(tag)
    return dict(lemma=lemma.casefold(),pos=pos,forms=[form.lower()],features=features or {},
                generation_eligible=features is not None,evidence_kind='lexicon_descriptive_paradigm',raw_tags=tag)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipes',type=Path,default=Path('config/european-expansion/nominal'));args=p.parse_args()
    for path in sorted(args.recipes.glob('*.json')):
        recipe=json.loads(path.read_text());lang=recipe['language']
        if path.stem!=lang:raise ValueError('Recipe language mismatch')
        upstream=BASE/'nominal'/lang/'receipt.json'
        receipt=json.loads(upstream.read_text())
        if receipt['language']!=lang or receipt['revision']!=recipe['revision']:raise ValueError('Upstream mismatch')
        inventory={(a['lemma'],a['pos']) for a in json.loads((Path('la_output/resources/european')/lang/'morphology.json').read_text())}
        rows=[];counts=Counter()
        for file in receipt['files']:
            source=Path(file['path'])
            if sha256_file(source)!=file['sha256']:raise ValueError('Lexicon checksum mismatch')
            if source.name not in recipe['files']:continue
            for line in source.read_text().splitlines():
                row=convert(line.split(recipe.get('delimiter')),recipe,inventory)
                if row:rows.append(row);counts['eligible' if row['generation_eligible'] else 'veto_only']+=1
        out=BASE/'nominal-converted'/lang
        write(out/'analyses.json',rows)
        write(out/'receipt.json',dict(language=lang,surface_normalization='unicode_lower',sha256=sha256_file(out/'analyses.json'),recipe=recipe,upstream=receipt,counts=dict(counts),source_url=recipe['source_url'],status='descriptive; not normative; candidate only'))
        print(lang,dict(counts),flush=True)

if __name__=='__main__':main()
