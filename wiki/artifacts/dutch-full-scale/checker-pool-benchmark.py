import json,random,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dala.language_check import local_servers,LanguageCheck
from dala.profiles import load_profile
config=load_profile('nl_scale')['checker']
items=[]
for path in Path('la_output/dutch_uncapped_probe_checkpoints').glob('*.candidates.jsonl'):
 items.extend(json.loads(line) for line in path.open())
items=random.Random(9917).sample(items,200)
with local_servers(instances=16,threads=4,heap='2g') as urls:
 args=dict(dialects=config['dialects'],lexical_carrier=config['lexical_carrier'],hard_rule_ids=config['hard_rule_ids'],commit_batch_size=128)
 baseline=LanguageCheck(urls[0],**args)
 def screen(checker,item):
  pair,named=item
  return checker.screen(pair['original'],pair['corrupted'],pair['edits'],named)
 expected=[screen(baseline,item) for item in items];baseline.close()
 with tempfile.TemporaryDirectory() as tmp:
  pooled=LanguageCheck(urls,cache=Path(tmp)/'cold.sqlite',**args)
  started=time.monotonic()
  with ThreadPoolExecutor(max_workers=128) as executor:
   actual=list(executor.map(lambda item:screen(pooled,item),items))
  elapsed=time.monotonic()-started
  assert actual==expected,[(i,a,b) for i,(a,b) in enumerate(zip(actual,expected)) if a!=b][:2]
  report=dict(instances=16,threads_per_server=4,workers=128,heap_per_server='2g',candidates=len(items),cold_cache_seconds=elapsed,cold_cache_pairs_per_second=len(items)/elapsed,exact_screen_evidence_and_reasons_equal=True,retained=sum(reason is None for _,reason in actual),software=pooled.software)
  pooled.close()
 Path('wiki/artifacts/dutch-full-scale/checker-pool-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
