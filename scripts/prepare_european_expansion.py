"""Assemble versioned lexical/evidence inputs for the existing shared pipeline."""
import argparse
import gzip
import json
from pathlib import Path

from dala.common_pile import sha256_file
from dala.rule_compiler import compile_inflections
from scripts.prepare_european_sources import write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'la_output/resources/european-expansion'


def prepare(language,run_id,source_run):
    old=ROOT/'la_output/resources/european'/language
    converted=BASE/'converted'/language
    output=BASE/run_id/language
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    for path in old.iterdir():
        if path.name not in {'morphology.json','receipt.json','rules.json.gz'}:
            (output/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    receipt=json.loads((old/'receipt.json').read_text())
    lexical_receipt=json.loads((converted/'receipt.json').read_text())
    if receipt.get('surface_normalization') != 'unicode_lower' or lexical_receipt.get('surface_normalization') != 'unicode_lower':
        raise ValueError('Rebuild base and lexical resources with orthography-preserving preparation')
    if lexical_receipt['language']!=language:raise ValueError('Lexical language mismatch')
    if sha256_file(converted/'lexical-analyses.json')!=lexical_receipt['sha256']:raise ValueError('Lexical analysis checksum mismatch')
    base=json.loads((old/'morphology.json').read_text())
    lexical=json.loads((converted/'lexical-analyses.json').read_text())
    # Unknown tags are retained for ambiguity vetoes, never as generators.
    # No lexical form is promoted into the normative spell-check wordlist.
    morphology=base+lexical
    supplemental=BASE/'nominal-converted'/language/'analyses.json'
    if supplemental.exists():
        supplemental_receipt=json.loads(supplemental.with_name('receipt.json').read_text())
        if supplemental_receipt.get('surface_normalization') != 'unicode_lower':
            raise ValueError('Rebuild supplemental morphology with orthography-preserving preparation')
        if supplemental_receipt['language']!=language or sha256_file(supplemental)!=supplemental_receipt['sha256']:
            raise ValueError('Supplemental morphology mismatch')
        morphology+=json.loads(supplemental.read_text())
        write(output/'nominal-evidence.json',supplemental_receipt)
        receipt['files'].append(dict(path=str(output/'nominal-evidence.json'),sha256=sha256_file(output/'nominal-evidence.json'),url=supplemental_receipt['source_url']))
    write(output/'morphology.json',morphology)
    write(output/'lexical-evidence.json',lexical_receipt)
    receipt['files'].append(dict(path=str(output/'lexical-evidence.json'),sha256=sha256_file(output/'lexical-evidence.json'),url='https://github.com/unimorph/'+lexical_receipt['recipe']['resource']))
    receipt.update(morphology_sha256=sha256_file(output/'morphology.json'),morphology_analyses=len(morphology),
                   morphology_scope='UD training attestations plus descriptive UniMorph paradigms; not normative',lexical_analyses=len(lexical))
    write(output/'receipt.json',receipt)
    spec=json.loads((ROOT/'config/european'/f'{language}.json').read_text())
    override_path=ROOT/'config/european-expansion/grammar'/f'{language}.json'
    if override_path.exists():
        overrides=json.loads(override_path.read_text())
        if overrides['language']!=language:raise ValueError('Grammar input language mismatch')
        for definition in spec['grammar']:
            definition.update(overrides['rule_overrides'].get(definition['family'],{}))
    options=dict(spec['compiler_options'],allow_lexical_evidence=True,lexical_evidence_kinds=['unimorph_descriptive_paradigm','lexicon_descriptive_paradigm'])
    rules=compile_inflections(language,morphology,spec['grammar'],options)
    with gzip.open(old/'rules.json.gz','rt') as f:book=json.load(f)
    rules.extend(r for r in book['rules'] if r['operator']=='character_edit')
    for rule in rules:
        rule['language']=language
        if rule['operator']=='dictionary_inflection':
            rule['evidence_kind']='grammar_constraint_with_attested_and_descriptive_lexical_forms'
    book['sources']['morphology']=receipt
    review_path=ROOT/'config/european-expansion/spelling'/f'{language}.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        if review['language']!=language:raise ValueError('Spelling review language mismatch')
        evidence_path=BASE/'mined-evidence'/language/'spelling-candidates.json'
        if sha256_file(evidence_path)!=review['candidates_sha256']:raise ValueError('Spelling evidence checksum mismatch')
        candidates=json.loads(evidence_path.read_text())['candidates']
        index={(r['correct'],r['incorrect']):r for r in candidates}
        evidence={}
        for item in review['accepted']:
            key=(item['correct'],item['incorrect'])
            if key not in index:raise ValueError('Reviewed mapping absent from observed evidence')
            correct,incorrect=key
            if not correct.isalpha() or not incorrect.isalpha() or correct.lower()==incorrect.lower():raise ValueError('Invalid nonword mapping')
            evidence[correct+'→'+incorrect]=index[key]
            rules.append(dict(id=f'{language}_observed_spelling_{len(evidence):03}',language=language,family='spelling',operator='observed_nonword',
                              pos=['NOUN','ADJ','VERB','ADV'],mappings={correct.lower():[incorrect.lower()]},
                              correct=correct,incorrect=incorrect,requires_lexical_screen=True,sources=['observed_spelling'],
                              evidence_kind='observed_correction_agent_reviewed_nonword',audit_status='agent_provisional_not_native_validated',active=True))
        book['sources']['observed_spelling']=dict(language=language,review=review,evidence=evidence)
    if override_path.exists():
        book['sources']['grammar_extensions']=dict(language=language,sha256=sha256_file(override_path),input=overrides)
        for rule in rules:
            if rule['operator']=='dictionary_inflection':rule['sources'].append('grammar_extensions')
    book['rules']=rules
    book['surface_normalization']='unicode_lower'
    with gzip.GzipFile(filename=str(output/'rules.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(book,ensure_ascii=False).encode())
    profile=json.loads((BASE/source_run/language/'profile.json').read_text())
    profile.update(resource_directory=str(output),rulebook=str(output/'rules.json.gz'),parser_skip_unalignable_sentences=True,
                   description='CPU expansion candidate; separate language inputs, descriptive lexical evidence, screened sources. No human linguistic validation.')
    if override_path.exists():profile.update(overrides.get('profile_overrides', {}))
    profile['selection']['operator_priority']={'observed_nonword':-1}
    write(output/'profile.json',profile)
    write(output/'coverage.json',dict(language=language,analyses=len(morphology),rules=[dict(id=r['id'],family=r['family'],active=r['active'],mappings=sum(len(v) for v in r.get('mappings',{}).values())) for r in rules]))
    print(language,len(morphology),sum(sum(len(v) for v in r.get('mappings',{}).values()) for r in rules),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',default='lexical_v1');p.add_argument('--source-run',default='sources_v1');p.add_argument('--languages',nargs='+');args=p.parse_args()
    languages=args.languages or [p.name for p in sorted((BASE/'converted').iterdir()) if p.is_dir()]
    for language in languages:prepare(language,args.run_id,args.source_run)


if __name__=='__main__':main()
