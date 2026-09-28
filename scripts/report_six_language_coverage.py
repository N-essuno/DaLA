"""Measure resource breadth and realized coverage without inferring linguistic precision."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    result={}
    for root in a.directories:
        manifest=json.loads((root/'manifest.json').read_text());lang=manifest['language']
        book=json.loads((root/'rules.json').read_text());byid={r['id']:r for r in book['rules']}
        rows=[json.loads(s) for f in sorted(root.glob('*/pairs.jsonl')) for s in f.open()]
        families=Counter();rules=Counter();words=defaultdict(set);changes=defaultdict(set);groups=defaultdict(list);sources=Counter()
        for row in rows:
            sources[row['source_name']]+=1
            for e in row['edits']:
                family=e['corruption_type'];rid=e['rule_id']
                families[family]+=1;rules[rid]+=1;words[family].add(e['original'].casefold());changes[family].add((e['original'].casefold(),e['replacement'].casefold()))
                groups[(rid,row['source_name'])].append(row)
        sample={}
        for key,group in sorted(groups.items()):
            for row in random.Random(8126).sample(group,min(5,len(group))):sample[row['pair_id']]=dict(row,audit_stratum=list(key))
        for row in random.Random(8242).sample(rows,min(100,len(rows))):sample.setdefault(row['pair_id'],dict(row,audit_stratum=['uniform_random']))
        (a.output/f'{lang}-sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample.values()))
        result[lang]=dict(dataset=str(root),pairs=len(rows),sources=dict(sources),quality_status=manifest['quality_status'],
                         grammatical_pairs=sum(n for f,n in families.items() if f!='spelling'),spelling_pairs=families['spelling'],
                         selected_families=dict(families),selected_rules=dict(rules),
                         distinct_original_words={k:len(v) for k,v in words.items()},distinct_substitutions={k:len(v) for k,v in changes.items()},
                         configured_unselected=[r['id'] for r in book['rules'] if not rules[r['id']]],
                         active_grammar_families=sorted({byid[k]['family'] for k in rules if byid[k]['family']!='spelling'}),
                         sample_rows=len(sample),sample_sha256=hashlib.sha256((a.output/f'{lang}-sample.jsonl').read_bytes()).hexdigest(),
                         linguistic_audit='pending; automatic counts are not a precision estimate')
    (a.output/'coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
