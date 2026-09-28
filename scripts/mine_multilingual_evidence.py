"""Mine training-only correction inventories, keeping language and annotation provenance.

Output is evidence for review, not an automatically activated corruption rulebook.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

from dala.common_pile import sha256_file
from dala.error_evidence import mine
from dala.profiles import load_profile
from dala.language_packs.morphology import MorphologyPack


def annotated_patterns(recipe):
    module_path=Path(recipe['annotation_reader'])
    if sha256_file(module_path)!=recipe['annotation_reader_sha256']:raise ValueError('Annotation reader checksum mismatch')
    spec=importlib.util.spec_from_file_location('upstream_annotated_text',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    patterns=defaultdict(lambda:dict(documents=set(),occurrences=[]))
    counts=Counter();file_hashes={}
    for filename,digest in recipe['files'].items():
        path=Path(filename)
        if sha256_file(path)!=digest:raise ValueError('Evidence checksum mismatch')
        file_hashes[filename]=digest
        document_id=path.name.split('.')[0]
        parsed=module.AnnotatedText(path.read_text())
        for annotation in parsed.iter_annotations():
            error_type=annotation.meta.get('error_type','')
            counts[error_type]+=1
            if error_type.startswith('F/'):
                raise ValueError('Fluency annotation in grammar-only input')
            if len(annotation.suggestions)!=1:continue
            wrong,correct=annotation.source_text,annotation.suggestions[0]
            if wrong==correct:continue
            record=patterns[(error_type,wrong,correct)]
            record['documents'].add(document_id)
            if len(record['occurrences'])<3:
                record['occurrences'].append(dict(file=filename,document_id=document_id,start=annotation.start,end=annotation.end))
    rows=[dict(error_type=kind,observed_error=wrong,observed_correction=correct,
               distinct_document_support=len(rec['documents']),evidence_locations=rec['occurrences'],
               status='unreviewed_evidence_not_a_rule')
          for (kind,wrong,correct),rec in patterns.items() if len(rec['documents'])>=recipe['min_support']]
    rows.sort(key=lambda r:(-r['distinct_document_support'],r['error_type'],r['observed_error'],r['observed_correction']))
    return dict(patterns=rows,files=file_hashes,annotation_counts=dict(counts),support_unit='distinct source document; annotators deduplicated')


def prepare(recipe,output):
    if recipe['split']!='train':raise ValueError('Only training evidence may be mined')
    if any('test' in Path(p).parts or 'dev' in Path(p).parts for p in recipe['files']):
        raise ValueError('Held-out data cannot be mined')
    for filename,digest in recipe['files'].items():
        if sha256_file(filename)!=digest:raise ValueError('Evidence checksum mismatch')
    if recipe['format']=='m2':
        report=mine(list(recipe['files']),min_support=recipe['min_support'])
    elif recipe['format']=='inline_annotation':report=annotated_patterns(recipe)
    else:raise ValueError('Unknown correction format')
    report.update(language=recipe['language'],recipe=recipe,
                  limitations='Observed corrections are contextual evidence, not safe substitutions. Taxonomy provenance is specified in the recipe. No held-out mining; no human linguistic validation.')
    pack=MorphologyPack(load_profile(recipe['language']))
    candidates=[]
    for record in report['patterns']:
        wrong,correct=record['observed_error'],record['observed_correction']
        spelling=record['error_type'] in recipe['spelling_categories'] and wrong.isalpha() and correct.isalpha() and wrong.lower()!=correct.lower()
        record['single_word_spelling_candidate']=bool(spelling)
        if not spelling:continue
        # Preserve case: lowercasing can conflate proper names and German nouns.
        if wrong[0].isupper()!=correct[0].isupper():continue
        if not pack.recognized(correct.lower()) or pack.recognized(wrong.lower()):continue
        candidates.append(dict(record,correct=correct,incorrect=wrong,
                               status='dictionary_screened_pending_agent_and_native_review'))
    output.mkdir(parents=True,exist_ok=True)
    for filename,value in [('patterns.json',report),('spelling-candidates.json',dict(language=recipe['language'],candidates=candidates))]:
        (output/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    summary=dict(language=recipe['language'],supported_patterns=len(report['patterns']),spelling_candidates=len(candidates),recipe=recipe,
                 by_type=dict(Counter(p['error_type'] for p in report['patterns'])))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(recipe['language'],summary['supported_patterns'],len(candidates),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipes',type=Path,default=Path('config/european-expansion/evidence'))
    p.add_argument('--output',type=Path,default=Path('la_output/resources/european-expansion/mined-evidence'))
    args=p.parse_args()
    for path in sorted(args.recipes.glob('*.json')):
        recipe=json.loads(path.read_text())
        if path.stem!=recipe['language']:raise ValueError('Evidence recipe language mismatch')
        prepare(recipe,args.output/recipe['language'])


if __name__=='__main__':main()
