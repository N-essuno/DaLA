"""Expand pinned GiellaLT normative generation, retaining all returned variants.

Run with Python 3.11 and hfst==3.16.0.1. Lemma seeds are not sentence data.
"""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import hfst
import yaml

ROOT=Path('la_output/resources/multilingual/fo')
TAGS={'Msc':('Gender','Masc'),'Fem':('Gender','Fem'),'Neu':('Gender','Neut'),
      'Sg':('Number','Sing'),'Pl':('Number','Plur'),'Nom':('Case','Nom'),'Acc':('Case','Acc'),
      'Dat':('Case','Dat'),'Gen':('Case','Gen'),'Indef':('Definite','Ind'),'Def':('Definite','Def'),
      'Comp':('Degree','Cmp'),'Superl':('Degree','Sup'),'Inf':('VerbForm','Inf'),'Sup':('VerbForm','Sup'),
      'PrfPtc':('VerbForm','Part'),'Prs':('Tense','Pres'),'Prt':('Tense','Past'),'Ind':('Mood','Ind')}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    if args.fetch:
        import zipfile
        from scripts.prepare_multilingual_resources import download
        name='grammar-fao-1.0.1-beta.1'
        url='https://github.com/giellalt/lang-fao/releases/download/grammar-fao/v1.0.1-beta.1/grammar-fao_1.0.1-beta.1%2Bbuild.93_noarch-all.zcheck'
        record=download(url,ROOT/(name+'.zcheck'))
        # Match the compact receipt written by the initial pinned fetch.
        (ROOT/(name+'.receipt.json')).write_text(json.dumps({k:record[k] for k in ['url','sha256']},indent=2))
        with zipfile.ZipFile(ROOT/(name+'.zcheck')) as archive:
            for member in archive.namelist():
                if not (ROOT/name/member).resolve().is_relative_to((ROOT/name).resolve()):raise ValueError('Unsafe archive member')
            archive.extractall(ROOT/name)
        revision=json.loads((ROOT/'receipt.json').read_text())['giella_revision']
        (ROOT/'giella-stems').mkdir(exist_ok=True)
        for stem in ['adjectives','nouns','verbs','determiners']:
            download(f'https://raw.githubusercontent.com/giellalt/lang-fao/{revision}/src/fst/morphology/stems/{stem}.lexc',ROOT/f'giella-stems/{stem}.lexc')
    generator=ROOT/'grammar-fao-1.0.1-beta.1/generator-gramcheck-gt-norm.hfstol'
    fst=hfst.HfstInputStream(str(generator)).read()
    suffixes=defaultdict(set)
    for p in (ROOT/'giella').glob('*.yaml'):
        for paradigm in (yaml.safe_load(p.read_text()) or {}).get('Tests',{}).values():
            if not isinstance(paradigm,dict):continue
            for key in paradigm:
                parts=key.split('+')
                if len(parts)>1:suffixes[parts[1]].add('+'.join(parts[1:]))
    for pos in ['A','N','Det']:
        for gender,number,case in itertools.product(['Msc','Fem','Neu'],['Sg','Pl'],['Nom','Acc','Dat','Gen']):
            for definite in (['','Indef','Def'] if pos=='Det' else ['Indef','Def']):
                suffixes[pos].add('+'.join([pos,gender,number,case]+([definite] if definite else [])))
    for tense,person in itertools.product(['Prs','Prt'],['1Sg','2Sg','3Sg','Pl']):
        suffixes['V'].add(f'V+Ind+{tense}+{person}')
    sources=[];forms=defaultdict(set);words=set();queries=0
    for filename,pos,upos in [('adjectives','A','ADJ'),('nouns','N','NOUN'),('verbs','V','VERB'),('determiners','Det','DET')]:
        p=ROOT/f'giella-stems/{filename}.lexc'
        sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        lemmas=set()
        for line in p.read_text().splitlines():
            line=line.split('!')[0].strip()
            if ':' not in line or ';' not in line:continue
            lemma=re.split(r'[:+]',line)[0]
            if lemma.isalpha() and lemma.islower():lemmas.add(lemma)
        for lemma in sorted(lemmas):
            for suffix in sorted(suffixes[pos]):
                queries+=1
                answers=fst.lookup(lemma+'+'+suffix)
                if not answers:continue
                features=dict(TAGS[t] for t in suffix.split('+') if t in TAGS)
                for tag in suffix.split('+'):
                    if re.fullmatch('[123](Sg|Pl)',tag):features.update(Person=tag[0],Number='Sing' if tag[1:]=='Sg' else 'Plur')
                if 'Mood' in features:features['VerbForm']='Fin'
                # FarPaHC/Stanza uses Part+Past for the perfect supine.
                if pos=='V' and features.get('VerbForm')=='Sup':features.update(VerbForm='Part',Tense='Past')
                if pos=='A':features.setdefault('Degree','Pos')
                for word,weight in answers:
                    # HFST optimized lookup can expose internal flag diacritics.
                    word=re.sub(r'@[PNDRCU]\.[^@]*@','',word)
                    if word.isalpha() and word.islower():
                        variants=[features]
                        # Faroese plural finite forms do not inflect for person;
                        # GiellaLT's +Pl analysis explicitly covers all persons.
                        if pos=='V' and features.get('VerbForm')=='Fin' and features.get('Number')=='Plur' and 'Person' not in features:
                            variants=[dict(features,Person=str(p)) for p in [1,2,3]]
                        for f in variants:forms[(lemma,upos,tuple(sorted(f.items())))].add(word)
                        words.add(word)
        print(filename,len(lemmas),'lemmas',len(forms),'analyses',flush=True)
    path=ROOT/'morphology.json'
    backup=ROOT/'test-paradigm-morphology.json'
    if not backup.exists():backup.write_bytes(path.read_bytes())
    entries=[dict(lemma=l,pos=p,features=dict(f),forms=sorted(ws),normative=True) for (l,p,f),ws in sorted(forms.items())]
    path.write_text(json.dumps(entries,ensure_ascii=False,sort_keys=True)+'\n')
    wf=ROOT/'normative-words.txt';wf.write_text('\n'.join(sorted(words))+'\n')
    rp=ROOT/'receipt.json';receipt=json.loads(rp.read_text())
    receipt.update(morphology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),morphology_analyses=len(entries),
                   normative_words_sha256=hashlib.sha256(wf.read_bytes()).hexdigest(),
                   morphology_source={'name':'GiellaLT normative generator','version':'grammar-fao/v1.0.1-beta.1',
                   'package':json.loads((ROOT/'grammar-fao-1.0.1-beta.1.receipt.json').read_text()),
                   'generator_sha256':hashlib.sha256(generator.read_bytes()).hexdigest(),'lemma_seed_revision':receipt['giella_revision'],
                   'lemma_files':sources,'hfst_version':'3.16.0.1','queries':queries})
    rp.write_text(json.dumps(receipt,indent=2)+'\n')
    print(len(entries),'analyses',len(words),'forms',flush=True)


if __name__=='__main__':main()
