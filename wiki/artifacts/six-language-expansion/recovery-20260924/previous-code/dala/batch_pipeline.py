"""Resumable, language-pack-driven scaling of the shared pair pipeline.

Candidate construction, screening, edit validation and task exports are shared
with the ordinary build. Immutable batch receipts make restarts safe; globally
ordered deduplication and split quotas are applied after screening.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json
import platform
import time

from .common_pile import documents, snapshots, sha256_file
from .near_duplicates import IndexedNearDuplicates
from .pair_pipeline import load_adapter, prepare_sentence, export_dataset, write_jsonl, source_functions, order_documents
from .profiles import load_profile, resource

_WORKER = None


def initialize_worker(profile, seed, max_errors, model):
    global _WORKER
    from .parsing import load_parser
    adapter = load_adapter(profile)
    book = adapter.load_rulebook(resource(profile,'rulebook'))
    nlp = load_parser(profile, model)
    _WORKER=(profile,seed,max_errors,adapter,book,nlp,
             {r['incorrect'] for r in book['rules'] if r['family']=='spelling' and 'incorrect' in r},
             {r['id']:r for r in book['rules']})


def prepare_batch(task):
    number, source_docs, directory = task
    profile,seed,max_errors,adapter,book,nlp,misspellings,rules = _WORKER
    root=Path(directory); path=root/f'{number:05d}.candidates.jsonl';receipt=path.with_suffix('.receipt.json')
    by_id={d['document_id']:d for d in source_docs}
    counts=Counter(documents=len(source_docs));rejected=Counter();eligible=Counter()
    def inputs():
        for d in source_docs:
            view=(dict(d,text=d['text'][d['_fragment_start']:d['_fragment_end']]) if '_fragment_start' in d else d)
            for text,offset in adapter.paragraphs(view):
                counts['paragraphs']+=1
                yield text,(d['document_id'],offset+d.get('_fragment_start',0))
    temporary=path.with_suffix('.part')
    with temporary.open('w') as handle:
        for parsed,(doc_id,offset) in nlp.pipe(inputs(),as_tuples=True,batch_size=64):
            for sent in parsed.sents:
                counts['sentences']+=1
                item,reason,families=prepare_sentence(sent,by_id[doc_id],offset,adapter,book,profile,
                                                     seed,max_errors,misspellings,rules)
                eligible.update(families)
                if reason:rejected[reason]+=1;continue
                handle.write(json.dumps(item,ensure_ascii=False)+'\n');counts['candidates']+=1
    temporary.replace(path)
    rejected.update(getattr(nlp, 'rejections', {}))
    if hasattr(nlp, 'rejections'): nlp.rejections.clear()
    record=dict(sha256=sha256_file(path),counts=dict(counts),rejections=dict(rejected),eligible=dict(eligible),
                document_ids=sorted(by_id))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    return record


def verified_receipt(path):
    receipt=path.with_suffix('.receipt.json')
    if not path.exists() or not receipt.exists():return None
    data=json.loads(receipt.read_text())
    if sha256_file(path)!=data['sha256']:raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return data


def task_documents(docs, adapter, max_chars=None):
    """Parallelize long documents at existing paragraph boundaries, without caps."""
    for document in docs:
        if not max_chars or len(document['text']) <= max_chars:
            yield document;continue
        start=end=None
        for text,offset in adapter.paragraphs(document):
            if start is not None and offset+len(text)-start>max_chars:
                yield dict(document,_fragment_start=start,_fragment_end=end)
                start=None
            if start is None:start=offset
            end=offset+len(text)
        if start is not None:
            yield dict(document,_fragment_start=start,_fragment_end=end)


def screen_batch(number, root, checker, workers):
    source=root/f'{number:05d}.candidates.jsonl'
    target=root/f'{number:05d}.screened.jsonl'
    existing=verified_receipt(target)
    if existing:return existing
    rejected=Counter();count=0
    def screen(item):
        pair,named=item
        evidence,reason=checker.screen(pair['original'],pair['corrupted'],pair['edits'],named)
        pair.update(checker=evidence,quality_status=getattr(checker,'quality_status','checker_screened'))
        return pair,reason
    temporary=target.with_suffix('.part')
    started = last_progress = time.monotonic()
    with source.open() as reader,temporary.open('w') as writer,ThreadPoolExecutor(max_workers=workers) as executor:
        items=(json.loads(line) for line in reader)
        for checked,(pair,reason) in enumerate(executor.map(screen,items),1):
            if time.monotonic()-last_progress>=30:
                print(f'Batch {number+1}: screened {checked:,}; checker retained {count:,}; {checked/(time.monotonic()-started):.0f} pairs/s',flush=True)
                last_progress=time.monotonic()
            if reason:rejected[reason]+=1;continue
            writer.write(json.dumps(pair,ensure_ascii=False)+'\n');count+=1
    temporary.replace(target)
    receipt=dict(sha256=sha256_file(target),pairs=count,rejections=dict(rejected))
    target.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def build_checkpointed(output_dir='la_output/english_scale',seed=4242,max_errors=2,
                       sources_path=None,rulebook_path=None,cache_dir='la_output/cache',model=None,
                       max_documents=None,offline=False,checker=None,profile=None):
    if checker is None:raise ValueError('Checkpointed builds require local screening')
    if Path(output_dir).exists():raise FileExistsError(output_dir)
    if not 1<=max_errors<=3:raise ValueError('max_errors must be 1, 2 or 3')
    if max_documents is not None and max_documents<1:raise ValueError('max_documents must be positive')
    profile=load_profile('en_scale') if profile is None else profile
    if rulebook_path is not None and Path(rulebook_path).resolve()!=resource(profile,'rulebook'):
        raise ValueError('Set the rulebook in the checkpointed profile')
    settings=profile['build'];model=model or profile['parser']
    sources_path=sources_path or resource(profile,'sources_config')
    adapter=load_adapter(profile);book=adapter.load_rulebook(resource(profile,'rulebook'))
    source_snapshots, source_documents = source_functions(profile)
    cached=source_snapshots(sources_path,cache_dir,offline)
    docs=order_documents(source_documents(cached), seed, profile)
    if max_documents is not None:docs=docs[:max_documents]
    metadata=[{k:v for k,v in d.items() if k!='text'} for d in docs]
    root=Path(settings['checkpoint_dir']);root.mkdir(parents=True,exist_ok=True)
    code_dir=Path(__file__).parent
    code={str(p.relative_to(code_dir)):sha256_file(p) for p in sorted(code_dir.rglob('*.py'))}
    from .parsing import parser_metadata
    parser=parser_metadata(profile,model)
    signature=dict(profile={k:v for k,v in profile.items() if k!='_path'},
                   profile_sha256=sha256_file(profile['_path']),rulebook_sha256=sha256_file(resource(profile,'rulebook')),
                   source_exclusions_sha256=sha256_file(resource(profile,'exclusions')),
                   sources_config_sha256=sha256_file(sources_path),sources=[r for _,_,r in cached],
                   code_sha256=code,seed=seed,max_errors=max_errors,max_documents=max_documents,
                   parser=parser,checker=checker.software,
                   document_order_sha256=hashlib.sha256('\n'.join(d['document_id'] for d in docs).encode()).hexdigest())
    identity=root/'run.json'
    if identity.exists():
        if json.loads(identity.read_text())!=signature:raise ValueError('Checkpoint inputs/code changed; choose a new checkpoint directory')
    else:identity.write_text(json.dumps(signature,indent=2)+'\n')
    batch_size=settings['document_batch_size'];workers=settings['parser_processes']
    work_docs=list(task_documents(docs,adapter,settings.get('max_task_chars')))
    tasks=[(i,work_docs[a:a+batch_size],str(root)) for i,a in enumerate(range(0,len(work_docs),batch_size))]
    targets=settings.get('split_targets') if max_documents is None else None
    if targets and sum(targets.values())!=settings['target_pairs']:raise ValueError('Split quotas do not sum to target')
    # Audited source exclusions are a final-selection input, separate from expensive
    # candidate/checker receipts. Their current hash is recorded in the export.
    review_path=settings.get('review_exclusions')
    excluded=set(json.loads(Path(review_path).read_text())) if review_path else set()
    counts=Counter();rejected=Counter();eligible=Counter();selected=Counter();split_counts=Counter()
    near=IndexedNearDuplicates();seen=set();pairs=[];processed=[];processed_docids=set()
    print(f'Checkpointed build: {len(docs):,} documents, {len(tasks)} batches; target {sum(targets.values()) if targets else "all"}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),
                             initializer=initialize_worker,initargs=(profile,seed,max_errors,model)) as pool:
        futures={}
        def submit(i):
            if i<len(tasks) and verified_receipt(root/f'{i:05d}.candidates.jsonl') is None:
                futures[i]=pool.submit(prepare_batch,tasks[i])
        for i in range(min(workers,len(tasks))):submit(i)
        for i in range(len(tasks)):
            receipt=verified_receipt(root/f'{i:05d}.candidates.jsonl')
            if receipt is None:receipt=futures.pop(i).result()
            submit(i+workers)
            counts.update(receipt['counts']);rejected.update(receipt['rejections']);eligible.update(receipt['eligible'])
            if settings.get('max_task_chars'):
                processed_docids.update(receipt['document_ids'])
                counts['document_fragments']+=len(tasks[i][1])
                counts['documents']=len(processed_docids)
            print(f'Batch {i+1}/{len(tasks)}: prepared {receipt["counts"].get("candidates",0):,} candidates; screening',flush=True)
            screened=screen_batch(i,root,checker,profile['checker'].get('workers',32))
            rejected.update(screened['rejections']);processed.append(i)
            for line in (root/f'{i:05d}.screened.jsonl').open():
                pair=json.loads(line);split=pair['split']
                if hashlib.sha256(pair['original'].encode()).hexdigest() in excluded:
                    rejected['review_exclusion']+=1;continue
                if targets and split_counts[split]>=targets[split]:
                    rejected['split_quota']+=1;continue
                if pair['original'] in seen or pair['corrupted'] in seen:
                    rejected['text_collision']+=1;continue
                if not near.add(pair['original']):
                    rejected['near_duplicate']+=1;continue
                pairs.append(pair);split_counts[split]+=1;seen.update((pair['original'],pair['corrupted']))
                selected.update(e['corruption_type'] for e in pair['edits'])
            (root/'progress.json').write_text(json.dumps(dict(processed_batches=processed,pairs=len(pairs),splits=dict(split_counts),counts=dict(counts),rejections=dict(rejected)),indent=2)+'\n')
            print(f'Batch {i+1} complete: {len(pairs):,} retained; splits {dict(split_counts)}',flush=True)
            if targets and all(split_counts[s]>=n for s,n in targets.items()):
                for future in futures.values():future.cancel()
                break
    shortfall={s:max(0,n-split_counts[s]) for s,n in (targets or {}).items()}
    if any(shortfall.values()) and not settings.get('allow_shortfall',False):
        raise ValueError(f'Source pool insufficient for target; preserved checkpoints and {len(pairs)} screened pairs. Splits: {dict(split_counts)}')
    manifest=dict(schema_version='2',language=profile['language'],name=profile['name'],prompts=profile['prompts'],
                  source_description=profile.get('description',''),profile={k:v for k,v in profile.items() if k!='_path'},
                  profile_sha256=signature['profile_sha256'],quality_status=getattr(checker,'quality_status','checker_screened'),
                  language_checker=checker.software,checker_runtime=getattr(checker,'runtime',None),seed=seed,
                  max_errors=max_errors,max_documents=max_documents,source_snapshots=signature['sources'],
                  sources_config_sha256=signature['sources_config_sha256'],rulebook_sha256=signature['rulebook_sha256'],
                  source_exclusions_sha256=signature['source_exclusions_sha256'],parser=parser,
                  runtime=dict(python=platform.python_version(),**{name:version(name) for name in profile.get('dependencies',[])}),
                  code_sha256=code,counts=dict(counts),rejections=dict(rejected),eligible_by_type=dict(eligible),
                  selected_by_type=dict(selected),checkpoint_receipt_sha256=sha256_file(identity),processed_batches=processed,
                  target_pairs=settings.get('target_pairs'),review_exclusions_sha256=sha256_file(review_path) if review_path else None,
                  target_shortfall=shortfall,source_pool_exhausted=len(processed)==len(tasks),
                  near_duplicate_method='Indexed acceleration of original trigram + SequenceMatcher >=0.9 rule')
    report=export_dataset(pairs,output_dir,manifest,book,metadata,seed)
    print(json.dumps(dict(output=str(output_dir),pairs=len(pairs),splits=report['splits']),indent=2),flush=True)
    return report
