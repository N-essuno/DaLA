"""Rebuild the curated English rule inventory from W&I training M2 evidence.

This selects an explicit inventory; it does not admit arbitrary mined mappings.
Run from the repository root with `python -m scripts.build_english_rulebook M2 ...`.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from dala.error_evidence import read_m2
from dala.english import BASE_FORMS, PARTICIPLES

SPELLING = {
    'comfortable': ['confortable'], 'because': ['becasue', 'beacuse'],
    'pollution': ['polution'], 'technology': ['technolgy'],
    'environment': ['envirnoment', 'enviromment'], 'government': ['goverment'],
    'library': ['libabry'], 'famous': ['fameous'], 'public': ['puplic'],
    'challenge': ['challange'], 'that': ['taht'], 'football': ['footbal'],
}
GERUNDS = dict(go='going', be='being', have='having', do='doing', work='working',
               make='making', help='helping', use='using', take='taking', give='giving',
               see='seeing', know='knowing', find='finding', leave='leaving', come='coming',
               speak='speaking', eat='eating', write='writing', need='needing', want='wanting')
NOUNS = dict(child='children', person='people', man='men', woman='women', foot='feet',
             tooth='teeth', book='books', student='students', problem='problems',
             reason='reasons', question='questions', animal='animals', year='years',
             thing='things', way='ways', day='days', friend='friends', job='jobs')


def build(files):
    observations = defaultdict(lambda: dict(sentences=set(), locations=[], by_file=defaultdict(set)))
    sources = []
    for filename in files:
        p = Path(filename)
        raw = p.read_bytes()
        sources.append(dict(file=p.name, sha256=hashlib.sha256(raw).hexdigest()))
        for i, (tokens, edits) in enumerate(read_m2(raw.decode()), 1):
            sid = hashlib.sha256(' '.join(tokens).encode()).hexdigest()
            for e in edits:
                wrong = ' '.join(tokens[e.start:e.end]).lower()
                correct = e.replacement.lower()
                key = (e.error_type, correct, wrong)
                r = observations[key]
                r['sentences'].add(sid)
                r['by_file'][p.name].add(sid)
                if len(r['locations']) < 3 and not any(x['sentence_sha256'] == sid for x in r['locations']):
                    r['locations'].append(dict(file=p.name, sentence_number=i, sentence_sha256=sid,
                                               start=e.start, end=e.end, annotator=e.annotator))
    rules = []

    def add(family, error_type, correct, wrong, minimum=1):
        obs = observations[(error_type, correct, wrong)]
        if len(obs['sentences']) < minimum:
            return
        rid = f'{family}:{correct}:{wrong}'
        if any(r['id'] == rid for r in rules):
            return
        rules.append(dict(id=rid, family=family, error_type=error_type,
            correct=correct, incorrect=wrong, distinct_sentence_support=len(obs['sentences']),
            support_by_file={k: len(v) for k, v in obs['by_file'].items()},
            evidence_locations=obs['locations']))

    for correct, wrongs in SPELLING.items():
        for wrong in wrongs:
            add('spelling', 'R:SPELL', correct, wrong, 3)
    for base, (third, _) in BASE_FORMS.items():
        if base == 'be':
            continue
        add('subject_verb_agreement', 'R:VERB:SVA', base, third)
        add('subject_verb_agreement', 'R:VERB:SVA', third, base)
    for correct, wrong in [('is', 'are'), ('are', 'is'), ('am', 'is')]:
        add('subject_verb_agreement', 'R:VERB:SVA', correct, wrong)
    for correct, wrong in [('this', 'these'), ('these', 'this'), ('that', 'those'), ('those', 'that')]:
        add('demonstrative_number', 'R:DET', correct, wrong)
    for base, gerund in GERUNDS.items():
        add('modal_verb_form', 'R:VERB:FORM', base, gerund)
    for base, (_, past) in BASE_FORMS.items():
        if base != 'be':
            add('do_support_form', 'R:VERB:FORM', base, past)
    for part, past in PARTICIPLES.items():
        if part != 'been':
            add('perfect_participle', 'R:VERB:FORM', part, past)
    for singular, plural in NOUNS.items():
        add('noun_number', 'R:NOUN:NUM', plural, singular, 3)
    add('possessive_confusion', 'R:DET', 'their', 'there', 3)
    return dict(version='1', evidence_corpus='BEA-2019 W&I v2.1, learner training A/B/C',
        evidence_url='https://www.cl.cam.ac.uk/research/nl/bea2019st/',
        curation='Agent-screened exact mappings plus syntactic guards; not human precision validation.',
        scope='Contemporary standard written English; grammar and spelling only.',
        support_unit='distinct source sentences, not independent writers; lowercase word forms pooled',
        sources=sources, rules=rules)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=Path('config/english_rules.json'))
    a = p.parse_args()
    book = build(a.files)
    a.output.write_text(json.dumps(book, indent=2) + '\n')
    print(f"Wrote {len(book['rules'])} evidence-backed mappings to {a.output}")
