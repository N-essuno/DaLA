"""Compare frozen pre-refactor code against the configured pipeline on real DDT.

All sentences are parsed once with the same model, then replayed to both engines.
Checks every targeted rule, labels/rows/splits, and exact RNG state transitions.
"""
import argparse
import contextlib
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import requests
import spacy

from dala.profiles import load_profile
from dala.rule_engine import ConfiguredRules, run_rule
from dala.pipeline import build


def main(output='wiki/artifacts/danish-equivalence.json'):
    sys.path.insert(0, str(Path('tests').resolve()))
    from reference_danish import dala_corrupt as old_rules
    from reference_danish import create_dala as old_build
    from reference_danish import load_ud as old_source
    cache = Path('la_output/cache/danish_equivalence');cache.mkdir(parents=True,exist_ok=True)
    receipt = cache/'sources.json'
    if receipt.exists():
        sources = json.loads(receipt.read_text())
    else:
        r = requests.get('https://api.github.com/repos/UniversalDependencies/UD_Danish-DDT/commits/master', timeout=60);r.raise_for_status()
        revision = r.json()['sha']
        sources = dict(revision=revision,files={})
        for split in ['train','dev','test']:
            url=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Danish-DDT/{revision}/da_ddt-ud-{split}.conllu'
            r=requests.get(url,timeout=60);r.raise_for_status()
            (cache/f'{split}.conllu').write_bytes(r.content)
            sources['files'][split]=dict(url=url,sha256=hashlib.sha256(r.content).hexdigest())
        receipt.write_text(json.dumps(sources,indent=2)+'\n')
    for split, record in sources['files'].items():
        assert hashlib.sha256((cache/f'{split}.conllu').read_bytes()).hexdigest()==record['sha256']
    def get(url):
        from types import SimpleNamespace
        split=url.rsplit('-',1)[-1].split('.')[0]
        return SimpleNamespace(text=(cache/f'{split}.conllu').read_text())
    with patch.object(old_source.requests,'get',side_effect=get):
        frames=old_source.load_dadt_pos()
    profile=load_profile('da');nlp=spacy.load(profile['parser'])
    texts=list(dict.fromkeys(t for f in frames.values() for t in f.doc))
    print(f'Parsing {len(texts)} unique DDT sentences',flush=True)
    parsed=dict(zip(texts,nlp.pipe(texts,batch_size=64)))
    model=lambda text: parsed[text]
    counts={};checks=0
    for rule in profile['rules']:
        if rule['operator']=='token_fallback':continue
        old=getattr(old_rules,rule['id']);hits=0
        for i,text in enumerate(texts):
            random.seed(i+4242)
            before=old(model,text,token_comparison=True);state=random.getstate()
            random.seed(i+4242)
            after=run_rule(rule,model,text,token_comparison=True)
            if before!=after or state!=random.getstate():
                raise AssertionError((rule['id'],text,before,after,'rng_equal',state==random.getstate()))
            hits+=before[0];checks+=1
        counts[rule['id']]=hits
        print(rule['id'],hits,'matching positive cases',flush=True)
    comparisons=[]
    for proportional in (False,True):
        for seed in (4242,73):
            with tempfile.TemporaryDirectory() as temp, patch.object(old_source,'load_dadt_pos',return_value=frames), patch.object(old_rules,'SpacyModelSingleton',return_value=model), contextlib.redirect_stdout(io.StringIO()):
                random.seed(seed)
                old=old_build.main(proportional,output_dir=Path(temp)/'old');state=random.getstate()
                random.seed(seed)
                new=build(profile,use_split_proportions=proportional,output_dir=Path(temp)/'new',source_frames=frames,model=model)
                assert state==random.getstate(),('pipeline RNG mismatch',proportional,seed)
                assert list(old)==list(new)
                for split in old:
                    assert old[split].to_list()==new[split].to_list(),('rows differ',proportional,seed,split)
                    assert (Path(temp)/'old'/f'dala_da_{split}.csv').read_bytes()==(Path(temp)/'new'/f'dala_da_{split}.csv').read_bytes()
                comparisons.append(dict(proportional=proportional,seed=seed,rows={k:len(v) for k,v in new.items()},corruptions=dict(Counter(r['corruption_type'] for r in new['train'] if r['label']=='incorrect'))))
            print('Full pipeline equal',proportional,seed,flush=True)
    result=dict(status='passed',sources=sources,parser=nlp.meta['name'],parser_package=profile['parser'],implementation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('dala/rule_engine.py'),Path('dala/token_operations.py'),Path('dala/text.py'),Path('dala/legacy_pipeline.py'),Path('dala/pipeline.py'),Path('dala/profiles.py')]},parser_version=nlp.meta['version'],spacy=spacy.__version__,unique_sentences=len(texts),rule_comparisons=checks,rule_positive_counts=counts,full_pipeline_comparisons=comparisons,exact_row_and_csv_equality=True,exact_rng_state_equality=True,scope='All active rules at default probability, all DDT sentences; both split modes and two seeds. Same cached parser annotations on both sides.',profile_sha256=hashlib.sha256(Path(profile['_path']).read_bytes()).hexdigest(),reference_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('tests/reference_danish').glob('*.py')})
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
