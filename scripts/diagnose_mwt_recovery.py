"""Bounded paired CPU diagnostic; never modifies production checkpoints."""
import json,time,argparse
from pathlib import Path
from collections import Counter
from dala.profiles import load_profile,resource
from dala.parsing import StanzaParser
from dala.language_packs.morphology import MorphologyPack

p=argparse.ArgumentParser();p.add_argument('language');p.add_argument('--paragraphs',type=int,default=30);a=p.parse_args()
root=Path('/work/mimir/DaLA');out=root/'wiki/artifacts/european-expansion/mwt-recovery';out.mkdir(exist_ok=True)
profile=load_profile(root/f'la_output/resources/european-expansion/european_scale_v1/{a.language}/profile.json')
profile['parser_recover_mwt']=True
profile['checker']['cache_path']=str(root/'la_output/cache'/('mwt-recovery-'+a.language+'.sqlite3'))
pack=MorphologyPack(profile);book=pack.load_rulebook(resource(profile,'rulebook'));parser=StanzaParser(profile)
source=json.loads(resource(profile,'sources_config').read_text())

# Explicit source path supplied by the canonical adapter configuration.
from dala.canonical_source import documents
paragraphs=[]
for doc in documents([(s, {'data':Path(s['path'])},s) for s in source['sources']]):
 for text,offset in pack.paragraphs(doc):
  paragraphs.append(text)
  if len(paragraphs)>=a.paragraphs:break
 if len(paragraphs)>=a.paragraphs:break
report={};examples=[]
from dala.morphology_check import MorphologyCheck
checker=MorphologyCheck(profile)
try:
    for mode in [False,True]:
     parser.recover_mwt=mode;counts=Counter();families=Counter();rejections=Counter();started=time.monotonic()
     for text in paragraphs:
      parsed=parser(text)
      if parsed is None:continue
      assert parsed.doc.text==text
      for sent in parsed.sents:
       counts['sentences']+=1
       reason=pack.sentence_rejection(sent,set())
       if reason:rejections[reason]+=1;continue
       counts['eligible_sources']+=1
       edits=pack.candidates(sent,book)
       if not edits:continue
       chosen=pack.select_edits(sent.text,edits,42,1)
       for edit in chosen:assert sent.text[edit.start:edit.end]==edit.original
       counts['candidate_pairs']+=1;families.update(e.corruption_type for e in chosen)
       if mode and any(getattr(t,'surface_protected',False) for t in sent):
        counts['recovered_candidate_pairs']+=1
        examples.append(dict(original=sent.text,edits=[e.__dict__ for e in chosen]))
     report[str(mode)]=dict(counts=counts,families=families,rejections=rejections,seconds=time.monotonic()-started)
    screened=Counter()
    for example in examples:
     edits=[dict(e,corrupted_start=e['start'],corrupted_end=e['start']+len(e['replacement'])) for e in example['edits']]
     e=edits[0];text=example['original'];corrupted=text[:e['start']]+e['replacement']+text[e['end']:]
     evidence,reason=checker.screen(text,corrupted,edits)
     example.update(corrupted=corrupted,screen_rejection=reason)
     screened[reason or 'accepted']+=1
finally:
    checker.close()
report['recovered_screening']=screened
report['language']=a.language;report['paragraphs']=len(paragraphs)
(out/f'{a.language}-recovery.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
(out/f'{a.language}-recovered-examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
print(json.dumps(report,ensure_ascii=False),flush=True)
