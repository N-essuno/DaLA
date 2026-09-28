"""Read-only sampling of cached production checker rejections for review."""
import argparse,json,sqlite3,hashlib,random
from pathlib import Path
from collections import Counter
from dala.language_check import LanguageCheck
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path.cwd())
parser.add_argument('--run-id',default='european_scale_v1')
parser.add_argument('--languages',nargs='+',default=['fr','it','es','ro'])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
summary={}
for lang in args.languages:
 db=sqlite3.connect(f'file:{root}/la_output/cache/source-{lang}.sqlite3?mode=ro',uri=True,timeout=3)
 row=db.execute('select response from checks limit 1').fetchone();namespace=json.dumps(json.loads(row[0])['software'],sort_keys=True)
 paths=sorted((root/f'la_output/european_pilots/{args.run_id}_checkpoints/{lang}').glob('*.candidates.jsonl'))
 paths=random.Random(92626).sample(paths,min(80,len(paths)))
 rejected=[];checked=0;missing=0;rules=Counter()
 for path in paths:
  if not path.with_suffix('.receipt.json').exists():continue
  for line in path.open():
   pair,named=json.loads(line);text=pair['original'];key=hashlib.sha256(f'{namespace}\0{lang}\0{text}'.encode()).hexdigest()
   response=db.execute('select response from checks where key=?',(key,)).fetchone()
   if not response:missing+=1;continue
   checked+=1
   matches=LanguageCheck.hard_matches(json.loads(response[0]),text)
   matches=[m for m in matches if not(m['issue_type']=='misspelling' and any(a<=m['start'] and m['end']<=b for a,b in named))]
   if matches:
    rules.update(m['rule_id'] for m in matches)
    rejected.append(dict(pair_id=pair['pair_id'],original=text,matches=matches))
 random.Random(92626).shuffle(rejected)
 (out/f'{lang}-checker-rejections.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rejected[:30]))
 summary[lang]=dict(run_id=args.run_id,qualification='Automatic cached-checker diagnostic, not linguistic validation',checked=checked,uncached=missing,rejected=len(rejected),rules=rules.most_common(15),sampled=min(30,len(rejected)))
 db.close()
(out/'checker-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(summary,ensure_ascii=False))
