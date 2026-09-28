"""Create deterministic per-family review packets without inventing judgments."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-family',type=int,default=5);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sample=[]
    for root in a.directories:
        groups=defaultdict(list)
        for file in sorted(root.glob('*/pairs.jsonl')):
            for line in file.open():
                row=json.loads(line)
                for family in {e['corruption_type'] for e in row['edits']}:groups[family].append(row)
        for family,rows in sorted(groups.items()):
            rows.sort(key=lambda r:hashlib.sha256(('coverage-review-20260926'+r['pair_id']).encode()).digest())
            for rank,row in enumerate(rows[:a.per_family]):sample.append(dict(row,dataset=str(root.resolve()),review_family=family,review_rank=rank))
    (a.output/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample))
    with (a.output/'native-review.csv').open('w') as f:
        fields=['language','pair_id','review_family','original','corrupted','source_correct','corruption_incorrect','only_intended_change','valid_alternative','notes','reviewer']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in sample:writer.writerow({k:r.get(k,'') for k in fields})
    print('Prepared',len(sample),'review rows; judgments blank')
if __name__=='__main__':main()
