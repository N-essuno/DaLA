"""Complete native analyses of retained surfaces, including unseeded homonyms.

Lemma-seeded generation discovers vocabulary; it cannot prove that a surface has
no other reading. Preserve every supported normative reading of each surface
before compiling syncretism-sensitive corruption rules. No new surfaces are
introduced by this pass.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from scripts.prepare_inflection_lexicons import polish_features, icelandic_features


def close(language):
    root = Path('la_output/resources/multilingual') / language
    path = root / 'morphology.json'
    previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = json.loads(path.read_text())
    forms = defaultdict(set)
    determiner_forms = set()
    surfaces = set()
    for item in entries:
        forms[(item['lemma'], item['pos'], tuple(sorted(item['features'].items())))].update(item['forms'])
        surfaces.update(item['forms'])
        if item['pos'] == 'DET':
            determiner_forms.update(item['forms'])
    if language == 'pl':
        import morfeusz2
        lexicon = morfeusz2.Morfeusz()
        def analyses(word):
            for start, end, (surface, lemma, tag, names, labels) in lexicon.analyse(word):
                if surface.casefold() != word or labels or 'nazwa_własna' in names:
                    continue
                for pos, features in polish_features(tag):
                    yield lemma.split(':')[0], pos, features
    else:
        from islenska import Bin
        lexicon = Bin()
        def analyses(word):
            for item in lexicon.lookup_ksnid(word)[1]:
                if (item.bmynd.casefold() != word or item.einkunn != 1 or item.beinkunn != 1
                        or item.malsnid or item.bmalsnid or item.bgildi or item.malfraedi):
                    continue
                value = icelandic_features(item.ofl, item.mark)
                if value:
                    yield item.ord, value[0], value[1]
    for n, word in enumerate(sorted(surfaces), 1):
        for lemma, pos, features in analyses(word):
            forms[(lemma, pos, tuple(sorted(features.items())))].add(word)
            if word in determiner_forms and pos in {'ADJ', 'PRON'}:
                forms[(lemma, 'DET', tuple(sorted(features.items())))].add(word)
        if n % 50000 == 0:
            print(language, n, '/', len(surfaces), flush=True)
    result = [dict(lemma=lemma, pos=pos, features=dict(features), forms=sorted(words), normative=True)
              for (lemma, pos, features), words in sorted(forms.items())]
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n')
    receipt_path = root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.update(morphology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   morphology_analyses=len(result),
                   surface_analysis_closure=dict(input_sha256=previous_hash,
                       input_analyses=len(entries), output_analyses=len(result), surfaces=len(surfaces),
                       method='All supported normative native analyses of every retained surface; no new surfaces'))
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(language, receipt['surface_analysis_closure'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages', nargs='+', choices=['pl', 'is'], default=['pl', 'is'])
    args = parser.parse_args()
    for language in args.languages:
        close(language)
