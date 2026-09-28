"""Paired source eligibility and grammar-edit checks for the early text filter."""
import argparse,json,time
from pathlib import Path
from collections import Counter
from dala.profiles import load_profile,resource
from dala.parsing import StanzaParser
from dala.language_packs.morphology import MorphologyPack

p=argparse.ArgumentParser();p.add_argument('profile',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--paragraphs',type=int,default=40);a=p.parse_args()
profile=load_profile(a.profile);profile['parser_recover_mwt']=True
profile['parser_prefilter_text']=True
pack=MorphologyPack(profile);book=pack.load_rulebook(resource(profile,'rulebook'));parser=StanzaParser(profile)
paragraphs=[]
for source in json.loads(resource(profile,'sources_config').read_text())['sources']:
 with Path(source['path']).open() as stream:
  for line in stream:
   for text,offset in pack.paragraphs(json.loads(line)):
    paragraphs.append(text)
    if len(paragraphs)>=a.paragraphs:break
   if len(paragraphs)>=a.paragraphs:break
 if len(paragraphs)>=a.paragraphs:break
results=[]
for enabled in [False,True]:
 parser.prefilter_text=enabled;parser.rejections.clear();parser.stage_seconds.clear();accepted=set();grammar=set();count=0;started=time.monotonic()
 for index,text in enumerate(paragraphs):
  doc=parser(text)
  if doc is None:continue
  assert doc.doc.text==text
  for sent in doc.sents:
   if pack.sentence_rejection(sent,set()):continue
   accepted.add((index,sent.start_char,sent.text))
   for edit in pack.candidates(sent,book):
    assert sent.text[edit.start:edit.end]==edit.original
    count+=1
    if edit.corruption_type!='spelling':grammar.add((index,sent.start_char,edit.start,edit.end,edit.original,edit.replacement,edit.rule_id))
 results.append(dict(accepted=accepted,grammar=grammar,candidates=count,seconds=time.monotonic()-started,timings=dict(parser.stage_seconds),rejections=dict(parser.rejections)))
report=dict(language=profile['language'],paragraphs=len(paragraphs),accepted_sources_equal=results[0]['accepted']==results[1]['accepted'],grammar_candidates_equal=results[0]['grammar']==results[1]['grammar'],
 runs=[dict(accepted_sources=len(r['accepted']),grammar_candidates=len(r['grammar']),**{k:v for k,v in r.items() if k not in {'accepted','grammar'}}) for r in results])
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert report['accepted_sources_equal'] and report['grammar_candidates_equal'], 'Source eligibility or grammar edit changed'
