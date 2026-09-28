"""Rebuild surface forms from pinned evidence, without reverse case-fold guesses."""
import argparse, copy, gzip, json
from pathlib import Path
from dala.common_pile import sha256_file
from dala.rule_compiler import compile_inflections
from scripts.prepare_european_pilots import attested_morphology, write
from scripts.prepare_unimorph_inputs import convert


def rebuild(profile_path, output):
    profile=json.loads(profile_path.read_text());language=profile['language']
    old=Path(profile['resource_directory']);receipt=json.loads((old/'receipt.json').read_text())
    original=json.loads((old/'morphology.json').read_text())
    if sha256_file(old/'morphology.json')!=receipt['morphology_sha256']:raise ValueError('Changed morphology')
    for record in receipt['files']:
        path=old/record.get('relative_path',Path(record['path']).name)
        if sha256_file(path)!=record['sha256']:raise ValueError(f'Changed evidence: {path}')
    source=json.loads(Path(f'config/european/{language}-sources.json').read_text())['sources'][0]
    training=Path(source['path'])
    if sha256_file(training)!=source['sha256']:raise ValueError('Changed UD training data')
    legacy,_=attested_morphology(training,source,str.casefold)
    if legacy!=[a for a in original if 'evidence_kind' not in a]:raise ValueError('UD reconstruction differs from pinned parent')
    attested,_=attested_morphology(training,source)
    lexreceipt=json.loads((old/'lexical-evidence.json').read_text());recipe=lexreceipt['recipe']
    lemmas={a['lemma'] for a in attested};lexical=[]
    folder=Path('la_output/resources/european-expansion/unimorph')/recipe['resource']
    for record in lexreceipt['upstream']['files']:
        if record['file'] not in recipe['files']:continue
        path=folder/record['file']
        if sha256_file(path)!=record['sha256']:raise ValueError('Changed UniMorph evidence')
        for line in path.open():
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=3:continue
            lemma,form,tags=fields
            if lemma.casefold() not in lemmas or not lemma.isalpha() or not form.isalpha():continue
            entry=convert(lemma,form,tags,recipe)
            if entry:lexical.append(entry)
    legacy_lexical=copy.deepcopy(lexical)
    for entry in legacy_lexical:entry['forms']=[w.casefold() for w in entry['forms']]
    if legacy_lexical!=[a for a in original if a.get('evidence_kind')=='unimorph_descriptive_paradigm']:
        raise ValueError('UniMorph reconstruction differs from pinned parent')
    if len(legacy)+len(lexical)!=len(original):raise ValueError('Unaccounted morphology evidence')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    for path in old.iterdir():
        if path.name not in {'morphology.json','receipt.json','rules.json.gz','profile.json','coverage.json'}:
            (output/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    write(output/'morphology.json',attested+lexical)
    receipt.update(morphology_sha256=sha256_file(output/'morphology.json'),morphology_analyses=len(attested)+len(lexical),
                   surface_normalization='unicode_lower_preserve_orthography',parent_morphology_sha256=sha256_file(old/'morphology.json'))
    write(output/'receipt.json',receipt)
    with gzip.open(profile['rulebook'],'rt') as f:book=json.load(f)
    # Use exactly the active, language-owned grammar definitions (including overrides).
    generated={'id','operator','sources','evidence_kind','audit_status','mappings','pairs','active','language'}
    definitions=[{k:v for k,v in r.items() if k not in generated} for r in book['rules'] if r['operator']=='dictionary_inflection']
    spec=json.loads(Path(f'config/european/{language}.json').read_text())
    options=dict(spec['compiler_options'],allow_lexical_evidence=True,lexical_evidence_kinds=['unimorph_descriptive_paradigm','lexicon_descriptive_paradigm'])
    rules=compile_inflections(language,attested+lexical,definitions,options)
    old_by_id={r['id']:r for r in book['rules']}
    for rule in rules:
        for key in ['language','sources','evidence_kind','audit_status']:
            if key in old_by_id[rule['id']]:rule[key]=old_by_id[rule['id']][key]
    for rule in book['rules']:
        if rule['operator']=='dictionary_inflection':continue
        rule=copy.deepcopy(rule)
        if rule['operator']=='observed_nonword':rule['mappings']={rule['correct'].lower():[rule['incorrect'].lower()]}
        rules.append(rule)
    book['rules']=rules;book['sources']['morphology']=receipt
    book['surface_normalization']='unicode_lower'
    with gzip.GzipFile(filename=str(output/'rules.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(book,ensure_ascii=False).encode())
    profile.update(resource_directory=str(output.resolve()),rulebook=str((output/'rules.json.gz').resolve()))
    profile['build'].update(checkpoint_dir=str(Path('la_output/european_pilots/orthography_v2_checkpoints',language).resolve()),parser_processes=40,backfill_reused_batches=False)
    write(output/'profile.json',profile)
    write(output/'repair-receipt.json',dict(language=language,parent_profile=str(profile_path.resolve()),parent_profile_sha256=sha256_file(profile_path),
        reconstructed_legacy_evidence_exact=True,source_normalization='lower; lemma IDs retain casefold',fresh_candidates_required=True,
        old_morphology_sha256=sha256_file(old/'morphology.json'),new_morphology_sha256=receipt['morphology_sha256'],
        old_rulebook_sha256=sha256_file(Path(json.loads(profile_path.read_text())['rulebook'])),new_rulebook_sha256=sha256_file(output/'rules.json.gz')))
    print(language,'rebuilt',len(attested)+len(lexical),'analyses',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--languages',nargs='+',default=['de','et','el']);a=p.parse_args()
    for lang in a.languages:rebuild(Path('la_output/resources/european-expansion/european_scale_v2')/lang/'profile.json',Path('la_output/resources/european-expansion/orthography_v2')/lang)
