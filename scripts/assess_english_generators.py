"""Measure individual productive operators and prepare a stratified review queue."""
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from dala.dataset_review import load_pairs
from scripts.assess_dataset import assess


def main():
    destination=Path('wiki/artifacts/english-generators-comparison.json')
    report=assess('la_output/english_spelling_expanded','la_output/english_generators',destination)
    pairs=load_pairs('la_output/english_generators')
    counts=Counter();surfaces=defaultdict(set);eligible=defaultdict(list)
    for p in pairs:
        for e in p['edits']:
            if e['rule_id'].startswith(('fallback:','spelling_pattern:')):
                name=e['rule_id'];counts[name]+=1
                surfaces[name].add((e['original'].lower(),e['replacement'].lower()))
                eligible[name].append(p)
    report['new_operators']={name:dict(edits=n,distinct_surface_substitutions=len(surfaces[name])) for name,n in sorted(counts.items())}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    rows=[];seen=set()
    for name,ps in sorted(eligible.items()):
        ps=sorted({p['pair_id']:p for p in ps if p['pair_id'] not in seen}.values(),key=lambda p:p['pair_id'])
        for p in random.Random('20260922:'+name).sample(ps,min(12,len(ps))):
            seen.add(p['pair_id'])
            rows.append(dict(pair_id=p['pair_id'],operator=name,original=p['original'],corrupted=p['corrupted'],edits=p['edits'],url=p['url'],original_correct=None,corrupted_incorrect=None,edits_valid=None,reviewer=None))
    Path('la_output/english_generators_review.json').write_text(json.dumps(dict(method='Agent queue stratified by new operator, up to 12 per operator; not a precision estimate',rows=rows),indent=2)+'\n')
    print(json.dumps(report['new_operators'],indent=2))


if __name__=='__main__':main()
