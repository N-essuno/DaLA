"""Materialize pinned full sources and resumable CPU profiles, without launching builds.

Full Wikipedia articles and Europarl sitting groups retain stable document IDs.
This prepares candidate runs, not a production-quality or publication decision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dala.common_pile import sha256_file
from scripts.prepare_european_sources import write

BASE=Path('la_output/resources/european-expansion')

def wiki_documents(language, lock):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for record in lock['files']:
        path=hf_hub_download(repo_id=lock['repo_id'],repo_type='dataset',revision=lock['revision'],filename=record['file'])
        if not record['sha256'] or sha256_file(path)!=record['sha256']:raise ValueError('Full source checksum mismatch')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield dict(source_document_id=language+':'+str(row['id']),text=row['text'],title=row['title'],url=row['url'],language=language,source_file=record['file'])


def sitting_documents(language, recipe):
    archive=BASE/recipe['archive']
    if sha256_file(archive)!=recipe['sha256']:raise ValueError('Europarl checksum mismatch')
    groups=defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.xml'):continue
            date=re.search(r'ep-(\d\d-\d\d-\d\d)',name).group(1)
            if any(date.startswith(p) for p in recipe.get('exclude_date_prefixes',[])):continue
            groups[date].append(name)
        for date,names in sorted(groups.items()):
            paragraphs=[]
            for name in names:
                root=ET.fromstring(z.read(name))
                paragraphs.extend(' '.join(''.join(s.itertext()).strip() for s in p.findall('s')) for p in root.iter('P'))
            yield dict(source_document_id=date,text='\n'.join(p for p in paragraphs if p),language=language,url=recipe['url'],source_files=names,source_document_scope='all_complete_chapters_grouped_by_sitting_date')


def prepare(language,recipe,lock,profile_root,output_root,workers):
    output=output_root/language;output.mkdir(parents=True,exist_ok=True)
    receipt_path=output/'preparation.json'
    source_identity=lock or recipe
    identity=hashlib.sha256(json.dumps(source_identity,sort_keys=True).encode()).hexdigest()
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['source_identity']!=identity or sha256_file(output/'documents.jsonl')!=receipt['sha256']:raise ValueError('Existing scale source differs')
    else:
        generators={'wikipedia_parquet':lambda:wiki_documents(language,lock),'europarl_xml_zip':lambda:sitting_documents(language,recipe)}
        rows=characters=0;temporary=output/'documents.jsonl.partial'
        with temporary.open('w') as f:
            for row in generators[recipe['format']]():
                f.write(json.dumps(row,ensure_ascii=False)+'\n');rows+=1;characters+=len(row['text'])
        temporary.replace(output/'documents.jsonl')
        receipt=dict(language=language,documents=rows,characters=characters,sha256=sha256_file(output/'documents.jsonl'),source_identity=identity,upstream=source_identity)
        write(receipt_path,receipt)
    source={k:recipe[k] for k in ['name','dataset','revision','language','license']}
    source.update(path=str((output/'documents.jsonl').resolve()),sha256=receipt['sha256'],candidate_source_not_clean_gold=True)
    write(output/'sources.json',dict(sources=[source]))
    profile=json.loads((profile_root/language/'profile.json').read_text())
    if profile['language']!=language:raise ValueError('Scale profile language mismatch')
    profile['sources_config']=str((output/'sources.json').resolve())
    profile['selection'].pop('coverage_targets',None)
    profile['build']=dict(parser_processes=workers,document_batch_size=8,max_task_chars=30000,checkpoint_dir=str((output/'checkpoints').resolve()),allow_shortfall=True,target_pairs=478930,split_targets=dict(train=383144,validation=47893,test=47893))
    profile['description']='Prepared full-source CPU candidate run; coverage pilot and native validation are separate gates. No production precision claim.'
    write(output/'profile.json',profile)
    print(language,receipt['documents'],receipt['characters'],flush=True)
    return dict(language=language,profile=str((output/'profile.json').resolve()),documents=receipt['documents'],source_sha256=receipt['sha256'],target=478930,status='prepared_not_started')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile-root',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--languages',nargs='+');p.add_argument('--workers',type=int,default=4);p.add_argument('--parser-workers',type=int,default=8);a=p.parse_args()
    if a.workers<1 or a.parser_workers<1:p.error('workers must be positive')
    locks=json.loads(Path('config/european-expansion/scale-source-lock.json').read_text())
    recipes=json.loads(Path('config/european-expansion/sources.json').read_text())['sources']
    recipes=[r for r in recipes if not a.languages or r['language'] in a.languages]
    output=BASE/a.run_id
    def work(r):return prepare(r['language'],r,locks.get(r['language']),a.profile_root,output,a.parser_workers)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(work,recipes))
    write(output/'readiness.json',dict(compute='CPU only',builds_started=False,languages=results))

if __name__=='__main__':main()
