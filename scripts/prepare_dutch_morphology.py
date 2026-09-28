"""Derive conservative Dutch inflection inventories from pinned LT 6.6 / OpenTaal."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile


def main():
    root=Path('la_output/resources/nl');tool=Path('la_output/tools/LanguageTool-6.6')
    jar=tool/'libs/dutch-pos-dict.jar';words=set((root/'wordlist.txt').read_text().casefold().splitlines())
    with zipfile.ZipFile(jar) as z:
        for name in ['dutch.dict','dutch.info','README.txt']:
            (root/('morphology-LICENSE.txt' if name=='README.txt' else name)).write_bytes(z.read('org/languagetool/resource/nl/'+name))
    dump=root/'dutch-pos.tsv'
    subprocess.run(['la_output/tools/jdk-17.0.16+8-jre/bin/java','-cp',str(tool/'languagetool.jar'),
                    'org.languagetool.tools.DictionaryExporter','-i',str(root/'dutch.dict'),'-info',str(root/'dutch.info'),'-o',str(dump)],check=True)
    tags=defaultdict(set);verbs=defaultdict(lambda:defaultdict(set));adjs=defaultdict(lambda:defaultdict(set))
    for line in dump.open():
        word,lemma,tag=line.rstrip('\n').split('\t')
        if word not in words or not word.isalpha() or not word.islower():continue
        tags[word].add(tag)
        if tag in {'WKW:TGW:1EP','WKW:TGW:3EP','WKW:TGW:INF','WKW:VTD:ONV'}:verbs[lemma][tag].add(word)
        if tag in {'BNW:STL:ONV','BNW:STL:VRB'}:adjs[lemma][tag].add(word)
    nouns={}
    for word,ts in tags.items():
        nt={t for t in ts if t.startswith('ZNW:')}
        genders={('het' if 'HET' in t else 'de' if 'DE_' in t else '?') for t in nt}
        if (nt and all(t.startswith('ZNW:EKV:') for t in nt) and len(genders)==1 and '?' not in genders
                and all(t.startswith('ZNW:') or t in {'WKW:TGW:1EP','WKW:TGW:3EP'} for t in ts)):
            nouns[word]=next(iter(genders))
    inflections={}
    for lemma,ts in verbs.items():
        if len(ts.get('WKW:TGW:3EP',set()))!=1 or ts.get('WKW:TGW:INF')!={lemma}:continue
        sg=next(iter(ts['WKW:TGW:3EP']))
        if sg==lemma:continue
        inflections[lemma]=dict(singular=sg,plural=lemma,first_person=sorted(ts.get('WKW:TGW:1EP',set())),participle=sorted(ts.get('WKW:VTD:ONV',set())))
    adjectives={}
    for lemma,ts in adjs.items():
        if ts.get('BNW:STL:ONV')=={lemma} and len(ts.get('BNW:STL:VRB',set()))==1:
            inflected=next(iter(ts['BNW:STL:VRB']))
            if inflected!=lemma:adjectives[lemma]=inflected
    result=dict(schema_version=1,source='LanguageTool 6.6 bundled dutch-pos-dict 0.1; intersected with pinned OpenTaal',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [jar,root/'dutch.dict',root/'wordlist.txt']},
                noun_articles=nouns,verb_inflections=inflections,adjective_inflections=adjectives)
    output=root/'morphology.json';output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps(dict(nouns=len(nouns),verbs=len(inflections),adjectives=len(adjectives),sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
