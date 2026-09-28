import json,hashlib,datetime
from pathlib import Path
from collections import Counter,defaultdict
r=Path('wiki/artifacts/dutch-extended-assessment')
errors={5:'Broken compounds inkoop startformulier and inkoop pagina; expected joined compounds.',29:'Toegang and gebruik require different prepositions: toegang tot en gebruik van de ruimte.',39:'Stray de in omdat zij de in de relevante periode geen eigenaar is geweest.',49:'Merged heading Termijnen binnen een selectieprocedure directly precedes De te hanteren ... .',50:'Missing indefinite article before singular count noun bestuursmodel.',97:'Stop gezet should be stopgezet for the verb stopzetten.',106:'Adjacent er onder should be eronder in this pronominal-adverb construction.',112:'Ize Hofmanhaëlstichting appears damaged by personal-name substitution into an institution name.',141:'Passive worden ... weergeven requires past participle weergegeven.',146:'Beroep van op leaves an incomplete van phrase / stray van.'}
uncertain={77:'Het bedrijf shifts to zij; possible institutional/collective gender usage but not unambiguously clean.',94:'Wat may refer to the whole situation or incorrectly to psychische problematiek; uncertain antecedent/punctuation.',159:'Dat in voor zover dat betrekking heeft has an unclear antecedent after feminine schade.'}
review=dict(reviewer='Codex agent',native_speaker_gold=False,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All 200 sampled originals and all recorded substitutions inspected, including grammar edit contexts; separate 25-family supplement. Grammar/spelling/extraction quality, not factual/legal verification.')
flags=[];by=defaultdict(Counter)
for name in ['sample.json','family-supplement.json']:
 p=r/name;a=json.loads(p.read_text());a['review']=review
 for row in a['rows']:
  i=int(row['sample_id'].split('-')[-1]);e=errors if name=='sample.json' else {};u=uncertain if name=='sample.json' else {2:'The 5 between essentiele and duurzame gebruiksgoederen may be an attached footnote or a literal count; not an unambiguous clean extraction.'}
  row.update(source_judgment='erroneous' if i in e else 'uncertain' if i in u else 'acceptable',intended_edits_judgment='valid',notes=e.get(i,u.get(i,'Acceptable source; recorded substitution introduces a clear grammar or spelling error.')))
  if name=='sample.json':by[row['source_name']][row['source_judgment']]+=1
  if row['source_judgment']!='acceptable':flags.append(dict(sentence_sha256=hashlib.sha256(row['original'].encode()).hexdigest(),reason=row['notes'],evidence=str(p)+'#'+row['sample_id'],pair_id=row['pair_id']))
 a['release_action']='Exclude flagged originals from a separately exported release; preserve these pre-exclusion judgments.';p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
a=json.loads((r/'sample.json').read_text());b=json.loads((r/'family-supplement.json').read_text())
summary=dict(review=review,uniform_sources=dict(Counter(p['source_judgment'] for p in a['rows'])),uniform_edits=dict(Counter(p['intended_edits_judgment'] for p in a['rows'])),by_source={k:dict(v) for k,v in by.items()},supplement_sources=dict(Counter(p['source_judgment'] for p in b['rows'])),supplement_edits=dict(Counter(p['intended_edits_judgment'] for p in b['rows'])),flagged_originals=len(flags),sample_population=a['sample_population'],excluded_previously_reviewed=a['excluded_previously_reviewed'],language_advice=['https://taaladvies.net/de-of-het-idee/'],note='Pre-exclusion estimate; removal does not independently establish a new precision rate for the release.')
(r/'review-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'release-exclusions.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,indent=2))
