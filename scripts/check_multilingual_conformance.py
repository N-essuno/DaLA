"""Exercise meaningful grammatical errors and valid-variant counterexamples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def check(language):
    from dala.profiles import load_profile,resource
    from dala.parsing import load_parser
    from dala.pair_pipeline import load_adapter
    profile=load_profile(language);adapter=load_adapter(profile)
    book=adapter.load_rulebook(resource(profile,'rulebook'));parser=load_parser(profile)
    fixtures=json.loads(Path('config/multilingual_conformance.json').read_text())['languages'][language]
    records=[]
    for fixture in fixtures:
        parsed=parser(fixture['text']);sent=list(parsed.sents)[0]
        rejection=adapter.sentence_rejection(sent,set())
        edits=adapter.candidates(sent,book) if not rejection else []
        families={e.corruption_type for e in edits};changes={(e.original.casefold(),e.replacement.casefold()) for e in edits}
        missing=sorted(set(fixture.get('required_families',[]))-families)
        forbidden=sorted(changes & {tuple(x) for x in fixture.get('forbidden_edits',[])})
        conflicts=[]
        if rejection=='source_morphology_conflict':
            for t in sent:
                entries=[e for pos in [t.pos_]+(['VERB'] if t.pos_=='AUX' else []) for e in adapter.analyses.get((t.text.casefold(),pos),[])]
                matching=[e for e in entries if e['lemma']==t.lemma_.casefold() and all(k not in e['features'] or e['features'][k]==v for k,v in t.morph.to_dict().items())]
                if entries and not matching and t.pos_ in {'ADJ','DET','VERB','AUX'}:
                    conflicts.append(dict(word=t.text,lemma=t.lemma_,features=t.morph.to_dict(),lexical=entries[:8]))
        records.append(dict(text=fixture['text'],rejection=rejection,families=sorted(families),missing=missing,forbidden=forbidden,
                            passed=not rejection and not missing and not forbidden,lexical_conflicts=conflicts))
    return dict(language=language,passed=all(r['passed'] for r in records),fixtures=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--languages',nargs='+',default=['pl','sv','nb','nn','fo','is']);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:print(json.dumps(check(a.languages[0]),ensure_ascii=False));return
    root=Path('wiki/artifacts/six-language-expansion/conformance');root.mkdir(exist_ok=True)
    def run(language):
        result=subprocess.run([sys.executable,'-m','scripts.check_multilingual_conformance','--worker','--languages',language],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        record=json.loads(result.stdout);(root/f'{language}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(language,record['passed'],flush=True);return record
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(run,a.languages))
    if not all(r['passed'] for r in records):raise SystemExit(1)


if __name__=='__main__':main()
