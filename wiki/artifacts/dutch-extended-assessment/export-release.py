import json,hashlib
from pathlib import Path
from collections import Counter
from dala.dataset_review import load_pairs,load_jsonl
from dala.pair_pipeline import export_dataset
from dala.common_pile import sha256_file
from dala.validate_dataset import validate
base=Path('la_output/dutch_dynaword_extended');out=Path('la_output/dutch_dynaword_release');r=Path('wiki/artifacts/dutch-extended-assessment');flags=json.loads((r/'release-exclusions.json').read_text());excluded={p['sentence_sha256'] for p in flags}
allpairs=load_pairs(base);pairs=[p for p in allpairs if hashlib.sha256(p['original'].encode()).hexdigest() not in excluded];assert len(allpairs)-len(pairs)==len(excluded)==14
m=json.loads((base/'manifest.json').read_text());m.update(target_pairs=len(pairs),release_review=dict(parent=str(base),parent_manifest_sha256=sha256_file(base/'manifest.json'),exclusions_sha256=sha256_file(r/'release-exclusions.json'),review_summary_sha256=sha256_file(r/'review-summary.json'),removed_pairs=14,method='Unchanged subset excluding audited erroneous/uncertain originals; no replacement or oversampling; no independent post-exclusion precision claim.'),selected_by_type=dict(Counter(e['corruption_type'] for p in pairs for e in p['edits'])))
export_dataset(pairs,out,m,json.loads((base/'rules.json').read_text()),load_jsonl(base/'documents.jsonl'));checks=validate(out)
families=Counter(e['corruption_type'] for p in pairs for e in p['edits']);subs=Counter((e['original'].casefold(),e['replacement'].casefold()) for p in pairs for e in p['edits'])
summary=dict(pairs=len(pairs),rows_per_task=len(pairs)*2,sources=dict(Counter(p['source_name'] for p in pairs)),families=dict(families),active_rules=len({e['rule_id'] for p in pairs for e in p['edits']}),distinct_substitutions=len(subs),top_substitutions=[dict(original=a,replacement=b,count=n) for (a,b),n in subs.most_common(10)],validation=checks,unchanged_subset=True,removed_pairs=14,manifest_sha256=sha256_file(out/'manifest.json'),splits=dict(Counter(p['split'] for p in pairs)))
(r/'release-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
