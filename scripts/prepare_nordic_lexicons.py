"""Compile independently curated Ordbank/SALDO forms into the common morphology schema."""
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path('la_output/resources/multilingual')


def fetch():
    import tarfile
    from scripts.prepare_multilingual_resources import download
    for language,name in [('nb','20220201_norsk_ordbank_nob_2005.tar.gz'),('nn','20220201_norsk_ordbank_nno_2012.tar.gz')]:
        root=ROOT/language
        record=download('https://www.nb.no/sbfil/leksikalske_databaser/ordbank/'+name,root/name)
        (root/'ordbank-download.json').write_text(json.dumps(record,indent=2)+'\n')
        if not (root/'ordbank').exists():
            with tarfile.open(root/name) as archive:archive.extractall(root/'ordbank',filter='data')
    record=download('https://svn.spraakbanken.gu.se/sb-arkiv/pub/lmf/saldom/saldom.xml',ROOT/'sv/saldom.xml')
    (ROOT/'sv/saldo-download.json').write_text(json.dumps(record,indent=2)+'\n')


def ordbank(language):
    root=ROOT/language/'ordbank'
    suffix='_2012' if language=='nn' else ''
    lemma_file=root/f'lemma{suffix}.txt'
    lemmas={r['LEMMA_ID']:r['GRUNNFORM'] for r in csv.DictReader(lemma_file.open(encoding='latin1'),delimiter='\t')}
    full=root/('fullformer_2012.txt' if language=='nn' else 'fullformsliste.txt')
    mapping={'mask':('Gender','Masc'),'fem':('Gender','Fem'),'nøyt':('Gender','Neut'),
             'ent':('Number','Sing'),'eint':('Number','Sing'),'fl':('Number','Plur'),
             'be':('Definite','Def'),'bu':('Definite','Def'),'ub':('Definite','Ind'),
             'pos':('Degree','Pos'),'komp':('Degree','Cmp'),'sup':('Degree','Sup'),
             'inf':('VerbForm','Inf'),'perf-part':('VerbForm','Part'),'nom':('Case','Nom'),'akk':('Case','Acc')}
    forms=defaultdict(set); words=set()
    for row in csv.DictReader(full.open(encoding='latin1'),delimiter='\t'):
        if row['TILDATO']!='4000' or row['NORMERING']!='normert': continue
        word=row['OPPSLAG'];lemma=lemmas.get(row['LEMMA_ID'],'')
        if not word.isalpha() or not word.islower(): continue
        words.add(word)
        tags=row['TAG'].split();pos={'subst':'NOUN','adj':'ADJ','verb':'VERB','det':'DET','pron':'PRON'}.get(tags[0])
        if not pos or not lemma.isalpha() or 'prop' in tags: continue
        features=dict(mapping[t] for t in tags if t in mapping)
        if pos=='VERB':
            if 'inf' in tags and 'pass' in tags:features.update(Voice='Pass')
            elif 'pres' in tags:features.update(VerbForm='Fin',Tense='Pres',Mood='Ind')
            elif 'pret' in tags:features.update(VerbForm='Fin',Tense='Past',Mood='Ind')
            elif 'imp' in tags:features.update(VerbForm='Fin',Mood='Imp')
        if pos=='DET':
            if lemma in {'en','ein'}:features['PronType']='Art'
            elif 'dem' in tags:features['PronType']='Dem'
        if pos=='ADJ' and '<perf-part>' in tags:features['VerbForm']='Part'
        genders=['Masc','Fem'] if 'm/f' in tags else [features.get('Gender')]
        for gender in genders:
            f=dict(features)
            if gender:f['Gender']=gender
            forms[(lemma.casefold(),pos,tuple(sorted(f.items())))].add(word)
    return forms,words,dict(url='https://www.nb.no/sprakbanken/ressurskatalog/'+('oai-nb-no-sbr-41/' if language=='nn' else 'en/oai-nb-no-sbr-5/'),
                            license='CC-BY',file=str(full),sha256=hashlib.sha256(full.read_bytes()).hexdigest())


def saldo():
    path=ROOT/'sv/saldom.xml';forms=defaultdict(set);words=set()
    mapping={'sg':('Number','Sing'),'pl':('Number','Plur'),'indef':('Definite','Ind'),'def':('Definite','Def'),
             'u':('Gender','Com'),'n':('Gender','Neut'),'pos':('Degree','Pos'),'komp':('Degree','Cmp'),
             'super':('Degree','Sup'),'nom':('Case','Nom'),'gen':('Case','Gen'),'ack':('Case','Acc'),
             'inf':('VerbForm','Inf'),'sup':('VerbForm','Sup'),'aktiv':('Voice','Act'),'s-form':('Voice','Pass')}
    for _,e in ET.iterparse(path,events=['end']):
        if e.tag!='LexicalEntry':continue
        d={f.attrib['att']:f.attrib['val'] for f in e.findall('Lemma/FormRepresentation/feat')}
        lemma=d.get('writtenForm','');pos={'nn':'NOUN','av':'ADJ','vb':'VERB','pn':'PRON','al':'DET'}.get(d.get('partOfSpeech'))
        inherent=[f.attrib['val'] for f in e.findall('Lemma/FormRepresentation/feat') if f.attrib['att']=='inherent']
        for w in e.findall('WordForm'):
            a={f.attrib['att']:f.attrib['val'] for f in w.findall('feat')};word=a.get('writtenForm','');tags=a.get('msd','').split()
            if not word.isalpha() or not word.islower():continue
            # Compound stems are not standalone inflected words.
            if any(t in tags for t in ['c','ci','cm','sms']):continue
            words.add(word)
            if not pos or not lemma.isalpha():continue
            features=dict(mapping[t] for t in [*inherent,*tags] if t in mapping)
            if pos=='VERB':
                if 'pres' in tags:features.update(VerbForm='Fin',Tense='Pres',Mood='Ind')
                elif 'pret' in tags:features.update(VerbForm='Fin',Tense='Past',Mood='Ind')
                elif 'imper' in tags:features.update(VerbForm='Fin',Mood='Imp')
            if pos=='DET':features['PronType']='Art'
            genders=['Com','Neut'] if pos=='NOUN' and 'v' in inherent else [features.get('Gender')]
            for gender in genders:
                f=dict(features)
                if gender:f['Gender']=gender
                forms[(lemma.casefold(),pos,tuple(sorted(f.items())))].add(word)
        e.clear()
    return forms,words,dict(url='https://sprakbanken.se/resurser/saldom',license='CC-BY-4.0',file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    if args.fetch:fetch()
    for language in ['nb','nn','sv']:
        forms,words,source=saldo() if language=='sv' else ordbank(language)
        path=ROOT/language/'morphology.json'
        # Preserve corpus evidence for inspection, but don't use its unverified
        # variants to expand the authoritative paradigms.
        ud=path.with_name('ud-morphology.json')
        if not ud.exists():ud.write_bytes(path.read_bytes())
        entries=[dict(lemma=lemma,pos=pos,features=dict(features),forms=sorted(values),normative=True)
                 for (lemma,pos,features),values in sorted(forms.items())]
        path.write_text(json.dumps(entries,ensure_ascii=False,sort_keys=True)+'\n')
        wordfile=path.with_name('normative-words.txt');wordfile.write_text('\n'.join(sorted(words))+'\n')
        receiptpath=path.with_name('receipt.json');receipt=json.loads(receiptpath.read_text())
        receipt.update(morphology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),morphology_analyses=len(entries),
                       morphology_source=source,normative_words_sha256=hashlib.sha256(wordfile.read_bytes()).hexdigest())
        receiptpath.write_text(json.dumps(receipt,indent=2)+'\n')
        print(language,len(entries),'analyses',len(words),'words',flush=True)


if __name__=='__main__':main()
