import json,re,hashlib,random
from pathlib import Path
from collections import Counter
from dala.dataset_review import load_pairs,load_jsonl
from dala.profiles import load_profile,resource
from dala.common_pile import sha256_file
from dala.pair_pipeline import export_dataset
from dala.validate_dataset import validate
base=Path('la_output/dutch_extension_pilot');out=Path('la_output/dutch_extension_validation');evidence=Path('wiki/artifacts/dutch-extension-validation');evidence.mkdir(exist_ok=False)
p=load_profile('nl_extension_validation');excluded={r['sentence_sha256'] for r in json.loads(resource(p,'exclusions').read_text())};patterns=p['curation']['source_risk_patterns'];allpairs=load_pairs(base)
pairs=[r for r in allpairs if hashlib.sha256(r['original'].encode()).hexdigest() not in excluded and not any(re.search(pattern,r['original']) for pattern in patterns)]
m=json.loads((base/'manifest.json').read_text());m.update(name=p['name'],profile={k:v for k,v in p.items() if k!='_path'},profile_sha256=sha256_file(p['_path']),source_exclusions_sha256=sha256_file(resource(p,'exclusions')),curation_replay=dict(parent=str(base),parent_manifest_sha256=sha256_file(base/'manifest.json'),method='Unchanged subset: additional source regex filters and attributed exclusions only; original parser/checker evidence retained, no regeneration.',removed_pairs=len(allpairs)-len(pairs)),selected_by_type=dict(Counter(e['corruption_type'] for r in pairs for e in r['edits'])))
export_dataset(pairs,out,m,json.loads((base/'rules.json').read_text()),load_jsonl(base/'documents.jsonl'));checks=validate(out)
reviewed=set()
for d in ['dutch-extension-pilot-assessment','dutch-full-assessment','dutch-validation-assessment','dutch-pilot-assessment','dutch-final-assessment','dutch-curated-assessment']:
 for n in ['sample.json','family-supplement.json']:
  f=Path('wiki/artifacts')/d/n
  if f.exists():reviewed.update(r['original'] for r in json.loads(f.read_text())['rows'])
rows=[];pop={}
for source in ['de_rechtspraak','officiele_bekendmakingen']:
 pool=sorted([r for r in pairs if r['source_name']==source and r['original'] not in reviewed],key=lambda r:r['pair_id']);pop[source]=len(pool)
 for r in random.Random('dutch-extension-validation-v1:'+source).sample(pool,50):
  rows.append(dict(sample_id=f'ev-{len(rows):03d}',**{k:r[k] for k in ('pair_id','split','document_id','source_name','original','corrupted','edits')},source_judgment=None,intended_edits_judgment=None,notes=None))
(evidence/'sample.json').write_text(json.dumps(dict(method='Uniform 50 per source from unchanged filtered pilot subset, excluding all previously reviewed originals; stratified, not an overall uniform sample.',seed='dutch-extension-validation-v1',populations=pop,rows=rows),ensure_ascii=False,indent=2)+'\n')
receipt=dict(pairs=len(pairs),removed=len(allpairs)-len(pairs),sources=dict(Counter(p['source_name'] for p in pairs)),validation=checks,parent_source_checks='wiki/artifacts/dutch-extension-pilot-assessment/summary.json',unchanged_subset=True,manifest_sha256=sha256_file(out/'manifest.json'))
(evidence/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
