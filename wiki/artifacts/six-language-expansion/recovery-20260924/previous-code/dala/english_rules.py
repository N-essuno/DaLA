"""Evidence-backed English corruptions using native spaCy dependencies.

Rules deliberately undergenerate. Exact attested surface substitutions are
licensed only in a matching syntactic context; no arbitrary deletion fallback.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

DEFAULT_RULEBOOK = Path(__file__).resolve().parents[1] / 'config/english_rules.json'
PRIORITY = ('perfect_participle', 'do_support_form', 'noun_number', 'modal_verb_form',
            'demonstrative_number', 'subject_verb_agreement', 'spelling')


from .edits import Corruption, apply_edits


def load_rulebook(path=DEFAULT_RULEBOOK):
    book = json.loads(Path(path).read_text())
    ids = set()
    for r in book['rules']:
        if r['id'] in ids or r['family'] not in PRIORITY:
            raise ValueError(f"Duplicate or unsupported rule: {r['id']}")
        if not r['correct'].isalpha() or not r['incorrect'].isalpha() or r['correct'] == r['incorrect']:
            raise ValueError(f"Invalid surface substitution: {r['id']}")
        if r['distinct_sentence_support'] < 1:
            raise ValueError(f"Missing evidence: {r['id']}")
        ids.add(r['id'])
    return book


def candidates(sent, book):
    """Offsets are relative to the sentence; input is a parsed spaCy Span."""
    lookup = {}
    for rule in book['rules']:
        lookup.setdefault((rule['family'], rule['correct']), []).append(rule)
    result = []

    def add(family, token, protect=(), allowed=None):
        if token.ent_type_ or token.pos_ == 'PROPN' or not token.is_alpha:
            return
        if not (token.text.islower() or token.text.istitle()):
            return
        for rule in lookup.get((family, token.lower_), []):
            if allowed is not None and rule['incorrect'] not in allowed:
                continue
            replacement = rule['incorrect']
            if token.text.istitle():
                replacement = replacement.capitalize()
            result.append(Corruption(rule['id'], family, token.idx - sent.start_char,
                token.idx - sent.start_char + len(token.text), token.text, replacement,
                tuple(sorted({token.i, *(t.i for t in protect)}))))

    for t in sent:
        # Closed, individually screened spelling lexicon; excludes named entities.
        add('spelling', t)
        head = t.head
        if (t.lower_ in {'this', 'that', 'these', 'those'} and t.dep_ == 'det'
                and t.pos_ == 'DET' and head.pos_ == 'NOUN' and head.i == t.i + 1
                and not any(c.dep_ in {'conj', 'cc', 'compound', 'nummod'} for c in head.children)):
            if (t.lower_ in {'this', 'that'} and head.tag_ == 'NN'
                    or t.lower_ in {'these', 'those'} and head.tag_ == 'NNS'):
                add('demonstrative_number', t, [head])
        if t.tag_ == 'NNS' and t.pos_ == 'NOUN':
            quantifiers = [c for c in t.children if c.lower_ in {'many', 'several'}
                           and c.i + 1 == t.i and c.dep_ in {'amod', 'det'}]
            if quantifiers and not any(c.dep_ in {'conj', 'cc', 'compound', 'nummod'} for c in t.children):
                add('noun_number', t, quantifiers)
        if t.pos_ in {'VERB', 'AUX'}:
            for aux in (c for c in t.children if c.dep_ == 'aux' and c.i < t.i):
                between = list(t.doc[aux.i + 1:t.i])
                if any(w.dep_ not in {'neg', 'advmod'} or w.head != t for w in between):
                    continue
                if t.tag_ == 'VB':
                    if aux.tag_ == 'MD' and aux.lower_ in {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}:
                        add('modal_verb_form', t, [aux])
                    elif aux.lemma_ == 'do' and aux.lower_ in {'do', 'does', 'did'}:
                        add('do_support_form', t, [aux])
                elif t.tag_ == 'VBN' and aux.lemma_ == 'have' and aux.lower_ in {'have', 'has', 'had'}:
                    add('perfect_participle', t, [aux])
        # Personal-pronoun subjects avoid collective/notional noun agreement.
        if (t.dep_ != 'nsubj' or t.pos_ != 'PRON' or t.lower_ not in {'i', 'he', 'she', 'it', 'we', 'they', 'you'}
                or list(t.children) or t.i >= head.i):
            continue
        if any(c.dep_ in {'mark', 'conj', 'cc'} for c in head.children):
            continue
        finite = [v for v in [head, *head.children]
                  if v.pos_ in {'VERB', 'AUX'} and v.tag_ in {'VBP', 'VBZ'}
                  and (v == head or v.dep_ in {'aux', 'cop'})]
        if len(finite) != 1:
            continue
        verb = finite[0]
        if t.i + 1 != verb.i:  # Avoid parentheticals, coordination, and ellipsis.
            continue
        singular = t.lower_ in {'he', 'she', 'it'}
        expected = 'VBZ' if singular else 'VBP'
        if verb.tag_ != expected:
            continue
        if verb.lemma_ == 'be':
            correct = 'is' if singular else 'am' if t.lower_ == 'i' else 'are'
            if verb.lower_ != correct:
                continue
        # The registry includes only number/person contrasts for this family.
        add('subject_verb_agreement', verb, [t])
    return result


def select_edits(text, edits, seed=4242, max_errors=2, priority=PRIORITY, rule_priority=None):
    """At most one grammar edit plus independent spelling edits.

Never combine agreement edits with another grammar edit: e.g. changing both a
subject and verb can restore agreement. A spelling edit cannot touch a token
that licenses the grammar edit. Return fewer edits when independence is unclear.
"""
    if not 1 <= max_errors <= 3:
        raise ValueError('max_errors must be 1, 2 or 3')
    if not edits:
        return []
    rule_priority = rule_priority or {}
    def rank(e):
        return (hashlib.sha256(f'{seed}\0{text}\0{e.rule_id}\0{e.start}'.encode()).digest(),
                hashlib.sha256(f'{seed}\0{e.replacement}'.encode()).digest())
    first = min(edits, key=lambda e: (priority.index(e.corruption_type), rule_priority.get(e.rule_id, 0), rank(e)))
    selected = [first]
    protected = set(first.protected_tokens)
    desired = 1 + int.from_bytes(hashlib.sha256(f'{seed}\0{text}'.encode()).digest()[:8], 'big') % max_errors
    for e in sorted((e for e in edits if e.corruption_type == 'spelling'),
                    key=lambda e: (rule_priority.get(e.rule_id, 0), rank(e))):
        if len(selected) >= desired:
            break
        if protected.intersection(e.protected_tokens):
            continue
        if any(not (e.end < old.start or old.end < e.start) for old in selected):
            continue
        selected.append(e)
        protected.update(e.protected_tokens)
    return sorted(selected, key=lambda e: e.start)
