import json,re,random,hashlib
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
r=Path('wiki/artifacts/dutch-extension-investigation');paths={(n,k):p for n,k,p in json.loads((r/'downloads.json').read_text())}
results={}
for n in ['de_rechtspraak','officiele_bekendmakingen']:
 counts=Counter();allowed=set();rng=random.Random(721);samples=[];modern=[]
 for b in pq.ParquetFile(paths[n,'metadata']).iter_batches(batch_size=16384):
  for a in b.to_pylist():
   counts['quality:'+str(a['content_quality'])]+=1
   if a['content_quality']!='excellent' or a['content_integrity']!='complete' or a['content_ratio'] not in ['complete_content','mostly_content'] or a['content_length'] not in ['moderate','substantial']:continue
   if 'conversational' in a['content_type'] or re.search(r'\b(?:interview|podcast|speech|transcript|conversation|dialogue|debate|debates)\b',a['one_sentence_description'],re.I):continue
   allowed.add(a['id'])
 counts['annotation_selected']=len(allowed)
 for b in pq.ParquetFile(paths[n,'data']).iter_batches(batch_size=1024):
  for a in b.to_pylist():
   if a['id'] not in allowed or len(a['text'])<400:continue
   counts['selected_documents']+=1
   item=dict(id=a['id'],chars=len(a['text']),text=a['text'][:4500])
   if len(samples)<12:samples.append(item)
   else:
    j=rng.randrange(counts['selected_documents'])
    if j<12:samples[j]=item
   y=re.search(r'\b(?:19|20)\d{2}\b',a['text'][:2000])
   if n=='officiele_bekendmakingen' and (not y or int(y.group())<2000):continue
   counts['modern_selected']+=1
   counts['characters']+=len(a['text'])
   counts['paragraphs_50_6000']+=sum(50<=len(x.strip())<=6000 for x in a['text'].splitlines())
   if len(modern)<12:modern.append(item)
   else:
    j=rng.randrange(counts['modern_selected'])
    if j<12:modern[j]=item
 results[n]=dict(counts=dict(counts),annotation_sample=samples,modern_sample=modern)
 (r/'inventory.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(n,dict(counts),flush=True)
