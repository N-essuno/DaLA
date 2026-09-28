"""Prepare language-isolated pilot inputs for the existing shared pair pipeline."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import requests

from dala.conllu_source import records, eligible
from dala.rule_compiler import compile_inflections

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'la_output/resources/european'
ART = ROOT/'wiki/artifacts/european-pilots'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def fetch(url, path):
    if not path.exists():
        response=requests.get(url,timeout=(20,180));response.raise_for_status()
        path.write_bytes(response.content)
    return dict(url=url,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def attested_morphology(path, source, normalize_surface=str.lower):
    forms=defaultdict(set);counts=Counter()
    source_counts=Counter()
    for rec in records(path):
        source_counts['total']+=1
        if not eligible(rec,source):continue
        source_counts['eligible']+=1
        for row in rec['rows']:
            if not row[0].isdigit():continue
            _,form,lemma,pos,_,features,*_=row
            if not form.isalpha() or not lemma.isalpha() or pos not in {'NOUN','ADJ','VERB','AUX','DET','PRON'}:continue
            key=(lemma.casefold(),pos,features)
            forms[key].add(normalize_surface(form));counts[(*key,normalize_surface(form))]+=1
    morphology=[dict(lemma=lemma,pos=pos,features=dict(f.split('=',1) for f in feats.split('|') if '=' in f),
                     forms=sorted(words),attestations={w:counts[(lemma,pos,feats,w)] for w in words})
                for (lemma,pos,feats),words in sorted(forms.items())]
    return morphology, source_counts


def prepare(spec_path, live_parsers=False):
    spec=json.loads(spec_path.read_text());language=spec['language']
    if spec_path.stem != language or spec.get('dictionary') not in {None,language}:
        raise ValueError('Language input/dictionary standard mismatch')
    root=RES/language;root.mkdir(parents=True,exist_ok=True)
    if spec.get('resource_packages'):
        package_root=root/'voikko';package_root.mkdir(exist_ok=True)
        for package in spec['resource_packages']:
            name,version=package.split('=',1)
            if not list(package_root.glob(f'{name}_{version}_*.deb')):
                subprocess.run(['apt-get','download',package],cwd=package_root,check=True)
        for archive in package_root.glob('*.deb'):
            subprocess.run(['dpkg-deb','-x',str(archive),'root'],cwd=package_root,check=True)
    pin=root/'pins.json'
    if not pin.exists():
        r=requests.get(f'https://api.github.com/repos/UniversalDependencies/UD_{spec["treebank"]}/commits/r2.17',timeout=60);r.raise_for_status()
        write(pin,dict(ud=r.json()['sha'],dictionary=json.loads((ROOT/'la_output/resources/multilingual/dictionary-pin.json').read_text())['revision']))
    pins=json.loads(pin.read_text());files=[]
    base=f'https://raw.githubusercontent.com/UniversalDependencies/UD_{spec["treebank"]}/{pins["ud"]}'
    for name in [spec.get('treebank_readme','README.md'),'LICENSE.txt']:files.append(fetch(f'{base}/{name}',root/name))
    inventory=root/'tree.json'
    if not inventory.exists():
        r=requests.get(f'https://api.github.com/repos/UniversalDependencies/UD_{spec["treebank"]}/git/trees/{pins["ud"]}',timeout=60);r.raise_for_status();write(inventory,r.json())
    names=sorted(x['path'] for x in json.loads(inventory.read_text())['tree'] if '-ud-train' in x['path'] and x['path'].endswith('.conllu'))
    # One immutable training shard per language is enough for a bounded pilot.
    # No official UD development or test text is used for rule mining or sources.
    name=names[0];files.append(fetch(f'{base}/{name}',root/name))
    if spec['dictionary']:
        dicbase=f'https://raw.githubusercontent.com/wooorm/dictionaries/{pins["dictionary"]}/dictionaries/{spec["dictionary"]}'
        for filename in spec.get('dictionary_files',['index.aff','index.dic','license','readme.md']):files.append(fetch(f'{dicbase}/{filename}',root/filename))
    else:
        # Resource configuration selects the backend; no language branch in generation.
        for p in sorted((root/'voikko').rglob('*')):
            if p.is_file() and not p.is_symlink():files.append(dict(path=str(p),relative_path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url='https://archive.ubuntu.com/ubuntu/'))
    source=dict(pilot_max_sentences=spec.get('pilot_max_sentences',3000),name='UD_'+spec['treebank'],language=language,path=str(root/name),revision=pins['ud'],
                sha256=hashlib.sha256((root/name).read_bytes()).hexdigest(),url=f'{base}/{name}',
                license=re.search(r'(?mi)^License:\s*(.+)',(root/spec.get('treebank_readme','README.md')).read_text()).group(1).strip())
    for key in ['sentence_id_pattern','document_id_pattern','text_normalization']:
        if spec.get(key): source[key]=spec[key]
    morphology,source_counts=attested_morphology(root/name,source)
    write(root/'morphology.json',morphology)
    receipt=dict(language=language,surface_normalization='unicode_lower',pins=pins,files=files,morphology_sha256=hashlib.sha256((root/'morphology.json').read_bytes()).hexdigest(),
                 morphology_analyses=len(morphology),morphology_scope='incomplete_training_attestations_not_normative',source_counts=dict(source_counts))
    write(root/'receipt.json',receipt)
    rules=compile_inflections(language,morphology,spec['grammar'],spec['compiler_options'])
    for definition in spec['spelling']:
        rules.append(dict(definition,id=language+'_spelling_'+definition['operation'],family='spelling',operator='character_edit',
                          requires_lexical_screen=True,sources=['dictionary'],evidence_kind='synthetic_nonword_generator_not_observed_substitution',audit_status='pending',active=True))
    for rule in rules:rule['language']=language
    book=dict(schema_version=1,language=language,surface_normalization='unicode_lower',edit_validator='dala.language_packs.morphology:permits',sources={
         'morphology':receipt,'dictionary':{'language':language,'revision':pins['dictionary'] if spec['dictionary'] else 'Ubuntu-noble-voikko-fi-2.5'},
         'ud_syntax':{'language':language,'url':f'https://universaldependencies.org/{language.split("-")[0]}/index.html'}},rules=rules)
    with gzip.GzipFile(filename=str(root/'rules.json.gz'),mode='wb',mtime=0) as stream:stream.write(json.dumps(book,ensure_ascii=False).encode())
    sources_path=ROOT/f'config/european/{language}-sources.json';write(sources_path,dict(sources=[source]))
    exclusions_path=ROOT/f'config/european/{language}-exclusions.json'
    if not exclusions_path.exists(): write(exclusions_path,[])
    profile=dict(schema_version=1,language=language,name='DaLA '+spec['name']+' pilot',mode='pairs',parser='annotated-conllu',parser_backend='conllu',
       sources={'adapter':'conllu'},sources_config=str(sources_path),rulebook=str(root/'rules.json.gz'),exclusions=str(exclusions_path),resource_directory=str(root),
       adapter='dala.language_packs.morphology:MorphologyPack',selection={'strategy':'hash_rotate_grammar_then_spelling','max_errors':1,'priority':list(dict.fromkeys(r['family'] for r in rules))},
       checker={'mode':'morphology','workers':1},prompts=spec['prompts'],curation={'min_words':7,'max_words':35,'max_sentence_chars':320,'max_paragraph_chars':6000,'source_risk_patterns':spec.get('source_risk_patterns',[])},
       description='Pinned UD training prose pilot; annotations are not a guarantee of correct originals. Incomplete attested morphology, synthetic constraints and independently screened nonword spelling. Unknown source-document boundaries are grouped conservatively by file. No human linguistic validation.',
       dependencies=['requests','spacy','spylls'],export_used_rule_mappings=True,release_policy=spec['release_policy'],morphology_options=spec.get('morphology_options',{}),
       dictionary_case_variants=spec.get('dictionary_case_variants',False),capitalized_corruption_pos=spec.get('capitalized_corruption_pos',[]))
    if spec.get('lexical_backend'): profile['lexical_backend'] = spec['lexical_backend']
    if live_parsers:
        from scripts.prepare_multilingual_parsers import prepare_settings
        config=spec['production_parser']
        settings=prepare_settings(config['language'],config['processors'],RES/'stanza')
        write(root/'live-parser-settings.json',settings)
    write(ROOT/f'config/languages/{language}.json',profile)
    coverage=dict(language=language,source_counts=dict(source_counts),rules=[dict(id=r['id'],family=r['family'],active=r['active'],mappings=sum(len(v) for v in r.get('mappings',{}).values())) for r in rules])
    write(ART/f'{language}-configured.json',coverage)
    print(language,'prepared',source_counts,'active',sum(r['active'] for r in rules),flush=True)
    return coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--languages',nargs='+');parser.add_argument('--live-parsers',action='store_true');args=parser.parse_args()
    paths=[p for p in sorted((ROOT/'config/european').glob('*.json')) if not p.stem.endswith(('-sources','-exclusions')) and (not args.languages or p.stem in args.languages)]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda path: prepare(path,args.live_parsers),paths))


if __name__=='__main__':main()
