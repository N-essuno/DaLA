"""Compare historical POS noise and guarded English candidates on clean sources.

Run before a full build. Evidence is automatic; judgments require separate review.
"""
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import spacy
from dala.dataset_review import load_pairs
from dala.language_check import LanguageCheck, local_server
from dala.language_packs.english import EnglishPack
from dala.language_packs.english_fallback import candidates as fallback_candidates
from dala.profiles import load_profile, resource
from dala.edits import apply_edits
from dala.token_operations import delete, flip_neighbours


def main():
    profile = load_profile('en'); pack = EnglishPack(profile)
    book = pack.load_rulebook(resource(profile, 'rulebook'))
    rules = {r['id']:r for r in book['rules']}
    originals = sorted({p['original'] for p in load_pairs('la_output/english_spelling_expanded')})
    sample = random.Random(20260922).sample(originals, min(1500, len(originals)))
    nlp = spacy.load(profile['parser']); buckets = {}
    for index, doc in enumerate(nlp.pipe(sample, batch_size=64)):
        sent = next(doc.sents); text = sent.text
        named = [(e.start_char, e.end_char) for e in doc.ents]
        if index < 100:
            for operation in (delete, flip_neighbours):
                corrupted = operation([t.text for t in sent], [t.pos_ for t in sent], rng=random.Random(f'4242:{text}:{operation.__name__}'))
                if corrupted and corrupted != text:
                    name = 'legacy:' + operation.__name__
                    edit = dict(rule_id=name,corruption_type='word_deletion' if operation==delete else 'word_swap',corrupted_start=0,corrupted_end=len(corrupted))
                    buckets.setdefault(name,[]).append(dict(original=text,corrupted=corrupted,edits=[edit],named=named))
        edits = fallback_candidates(sent, book['rules']) + [e for e in pack.candidates(sent,book) if rules[e.rule_id].get('operator')=='character_edit']
        byrule = {}
        for e in edits: byrule.setdefault(e.rule_id,[]).append(e)
        for name, options in byrule.items():
            bucket=buckets.setdefault(name,[])
            if len(bucket)>=60: continue
            e=random.Random(f'4242:{text}:{name}').choice(options)
            record=asdict(e);record.update(corrupted_start=e.start,corrupted_end=e.start+len(e.replacement))
            if rules[e.rule_id].get('requires_lexical_screen'):record['requires_lexical_screen']=True
            bucket.append(dict(original=text,corrupted=apply_edits(text,[e]),edits=[record],named=named))
    with local_server() as url:
        checker=LanguageCheck(url)
        def screen(item):
            name,p=item
            evidence,reason=checker.screen(p['original'],p['corrupted'],p['edits'],p.pop('named'))
            return dict(operator=name,**p,checker=evidence,rejection=reason,agent_judgment=None)
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows=list(pool.map(screen,[(name,p) for name,ps in sorted(buckets.items()) for p in ps]))
            software=checker.software
        finally:checker.close()
    counts={name:dict(candidates=len(ps),accepted=sum(r['operator']==name and r['rejection'] is None for r in rows)) for name,ps in buckets.items()}
    report=dict(method='Seeded 1500-source probe from previous checker-screened build; up to 60 candidates per new rule and 100 per legacy operator. Not a natural frequency or precision estimate.',software=software,rulebook_sha256=hashlib.sha256(resource(profile,'rulebook').read_bytes()).hexdigest(),counts=counts,rows=rows)
    Path('wiki/artifacts/english-generator-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
