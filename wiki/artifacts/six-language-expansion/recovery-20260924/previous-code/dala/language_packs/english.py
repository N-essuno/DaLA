"""English input adapter: guarded contexts plus dictionary-backed inflection.

Lexical inventories, grammar vocabulary, priorities and resources live in JSON.
Productive character edits require lexical screening. Token fallback has explicit
English syntactic guards; legacy Danish heuristics are not applied unchanged.
"""
import hashlib
import json
from pathlib import Path
from functools import lru_cache

from .. import curation
from ..english_rules import Corruption, select_edits
from ..profiles import resource
from ..character_operations import replacements as character_replacements
from .english_fallback import candidates as fallback_candidates


@lru_cache(maxsize=32768)
def forms(lemma, tag):
    from lemminflect import getInflection
    return tuple(getInflection(lemma, tag, inflect_oov=False))


def permits(rule, edit):
    if rule.get('operator') == 'character_edit':
        return edit.replacement.lower() in character_replacements(edit.original, rule)
    if rule.get('operator') == 'guarded_token':
        import re
        match = re.fullmatch(r'(\w+)([^\S\n]+)(\w+)', edit.original)
        if not match:
            return False
        left, space, right = match.groups()
        if not left.isalpha() or not right.isalpha():
            return False
        if rule['operation'] == 'delete_progressive_auxiliary':
            return left.lower() in rule['auxiliaries'] and right == edit.replacement
        return (rule['operation'] == 'swap_article_noun' and left.lower() in rule['articles']
                and edit.replacement == right + space + left)
    if rule.get('operator') != 'dictionary_inflection':
        return rule['correct'] == edit.original.lower() and rule['incorrect'] == edit.replacement.lower()
    # A concrete instantiation is recorded per edit; regenerate it independently.
    from lemminflect import getLemma
    source, target = edit.original.lower(), edit.replacement.lower()
    for before_tag, after_tag in rule['tag_transitions'].items():
        upos = 'NOUN' if before_tag.startswith('NN') else 'VERB'
        # The dictionary's reverse lemmatizer is not complete (e.g. lectures).
        # Validate both directions against inflection tables; do not guess OOV forms.
        lemmas = {source, target, *getLemma(source, upos=upos, lemmatize_oov=False),
                  *getLemma(target, upos=upos, lemmatize_oov=False)}
        for lemma in lemmas:
            if source in forms(lemma,before_tag) and target in forms(lemma,after_tag) and target not in forms(lemma,before_tag):
                return True
    return False


class EnglishPack:
    def __init__(self, profile):
        self.profile = profile
        if profile['selection']['strategy'] != 'rare_first':
            raise ValueError('English adapter currently supports rare_first selection')
        self.grammar = profile['grammar']
        self.rule_priority = {}
        self.exclusions = {r['sentence_sha256'] for r in json.loads(resource(profile,'exclusions').read_text())}

    def load_rulebook(self, path):
        book=json.loads(Path(path).read_text())
        ids=set()
        for r in book['rules']:
            if r['id'] in ids or r['family'] not in self.profile['selection']['priority']:
                raise ValueError('Invalid or duplicate rule')
            if r.get('operator') == 'character_edit':
                if (r['family'] != 'spelling' or not r.get('requires_lexical_screen')
                        or not r.get('evidence_rule_ids')
                        or any(e not in {x['id'] for x in book['rules']} for e in r['evidence_rule_ids'])
                        or r['operation'] not in {'transpose_internal', 'delete_doubled_consonant', 'duplicate_consonant'}):
                    raise ValueError('Character rule lacks evidence or lexical guard')
            elif r.get('operator') == 'guarded_token':
                if r.get('evidence_kind') != 'synthetic_syntax' or r['operation'] not in {'delete_progressive_auxiliary', 'swap_article_noun'}:
                    raise ValueError('Unsupported token fallback')
            elif r.get('operator') == 'dictionary_inflection':
                if not r['evidence_examples'] or not r['tag_transitions']:
                    raise ValueError('Productive rule lacks evidence or transition')
            elif not (r.get('distinct_sentence_support') or 0) >= 1:
                sources = r.get('published_sources', [])
                catalog = book.get('published_sources', {})
                if (r['family'] != 'spelling' or r.get('evidence_kind') != 'published_list'
                        or not sources or any(s not in catalog or not catalog[s].get('url') for s in sources)):
                    raise ValueError('Lexical rule lacks evidence')
            ids.add(r['id'])
        preference = self.profile['selection'].get('spelling_operator_priority', [])
        self.rule_priority = {r['id']: preference.index(r.get('operator', 'lexical'))
                              for r in book['rules'] if r['family'] == 'spelling'
                              and r.get('operator', 'lexical') in preference}
        return book

    def paragraphs(self, document):
        return curation.paragraphs(document)

    def sentence_rejection(self, sent, misspellings):
        if hashlib.sha256(sent.text.encode()).hexdigest() in self.exclusions:
            return 'audited_source_exclusion'
        return curation.sentence_rejection(sent,misspellings)

    def select_edits(self, text, edits, seed, max_errors):
        return select_edits(text,edits,seed,max_errors,priority=self.profile['selection']['priority'],
                            rule_priority=self.rule_priority)

    def candidates(self,sent,book):
        g=self.grammar
        lookup={}
        for r in book['rules']:
            if r.get('operator') in {'character_edit', 'guarded_token'}:
                continue
            lookup.setdefault(r['family'],[]).append(r)
        out=[]
        def add(family,t,protect=()):
            if t.ent_type_ or t.pos_=='PROPN' or not t.is_alpha or not (t.text.islower() or t.text.istitle()):return
            for r in lookup.get(family,[]):
                if r.get('operator')=='dictionary_inflection':
                    target=r['tag_transitions'].get(t.tag_)
                    if not target or t.lower_ not in forms(t.lemma_.lower(),t.tag_):continue
                    replacements=g['replacement_overrides'].get(t.lemma_.lower(),{}).get(target,forms(t.lemma_.lower(),target))
                    replacements=[x for x in replacements if x not in forms(t.lemma_.lower(),t.tag_)]
                else:
                    replacements=[r['incorrect']] if t.lower_==r['correct'] else []
                for replacement in replacements:
                    if not replacement.isalpha() or replacement==t.lower_:continue
                    if t.text.istitle():replacement=replacement.capitalize()
                    out.append(Corruption(r['id'],family,t.idx-sent.start_char,t.idx-sent.start_char+len(t.text),t.text,replacement,tuple(sorted({t.i,*(p.i for p in protect)}))))
        for t in sent:
            add('spelling',t)
            h=t.head
            if t.dep_=='det' and t.pos_=='DET' and h.pos_=='NOUN' and h.i==t.i+1 and g['demonstratives'].get(t.lower_)==h.tag_ and not any(c.dep_ in {'conj','cc','compound','nummod'} for c in h.children):
                # A number switch must not be licensed by an unchanged syncretic noun
                # (this fish -> these fish, this series -> these series).
                opposite = 'NNS' if h.tag_ == 'NN' else 'NN'
                if h.lemma_.lower() not in g['exclude_number_lemmas'] and h.lower_ not in g['exclude_number_lemmas'] and h.lower_ not in forms(h.lemma_.lower(), opposite):
                    add('demonstrative_number',t,[h])
            if t.tag_=='NNS' and t.pos_=='NOUN' and t.lemma_.lower() not in g['exclude_number_lemmas']:
                quantifiers=[c for c in t.children if c.lower_ in g['quantifiers'] and c.i+1==t.i and c.dep_ in {'amod','det'}]
                if quantifiers and not any(c.dep_ in {'conj','cc','compound','nummod'} for c in t.children):add('noun_number',t,quantifiers)
            if t.pos_ in {'VERB','AUX'}:
                for aux in (c for c in t.children if c.dep_=='aux' and c.i<t.i):
                    if any(w.dep_ not in {'neg','advmod'} or w.head!=t for w in t.doc[aux.i+1:t.i]):continue
                    if t.tag_=='VB':
                        if aux.tag_=='MD' and aux.lower_ in g['modals']:add('modal_verb_form',t,[aux])
                        elif aux.lemma_=='do' and aux.lower_ in g['do_forms']:add('do_support_form',t,[aux])
                    elif t.tag_=='VBN' and aux.lemma_=='have' and aux.lower_ in g['have_forms']:add('perfect_participle',t,[aux])
            if t.dep_!='nsubj' or t.i>=h.i:continue
            if any(c.dep_ in {'mark','conj','cc'} for c in h.children):continue
            singular=None;protect=[t]
            if t.pos_=='PRON' and not list(t.children):
                if t.lower_ in g['singular_pronouns']:singular=True
                elif t.lower_ in g['other_pronouns']:singular=False
            elif t.pos_=='NOUN' and t.lemma_.lower() not in g['exclude_agreement_lemmas']:
                children=list(t.children)
                if any(c.dep_ not in {'det','amod'} for c in children):continue
                dets=[c for c in children if c.dep_=='det']
                if t.tag_=='NN' and any(c.lower_ in g['singular_determiners'] for c in dets):singular=True
                elif t.tag_=='NNS' and any(c.lower_ in g['plural_determiners'] for c in children):singular=False
                protect+=children
            if singular is None:continue
            finite=[v for v in [h,*h.children] if v.pos_ in {'VERB','AUX'} and v.tag_ in {'VBP','VBZ'} and (v==h or v.dep_ in {'aux','cop'})]
            if len(finite)!=1:continue
            v=finite[0]
            if t.i+1!=v.i or v.tag_!=('VBZ' if singular else 'VBP'):continue
            if v.lemma_=='be':
                expected='is' if singular else 'am' if t.lower_=='i' else 'are'
                if v.lower_!=expected:continue
            add('subject_verb_agreement',v,protect)
        # Token fallback is considered only when the attested lexical/grammar
        # inventory abstains. Productive character noise is a separate extension.
        if not out and self.profile['selection'].get('token_fallback') == 'no_targeted_grammar_or_lexical':
            fallback = fallback_candidates(sent, book['rules'])
            if fallback:
                return fallback
        for r in book['rules']:
            if r.get('operator') != 'character_edit':
                continue
            for t in sent:
                if (t.ent_type_ or t.pos_ not in r['allowed_pos'] or not t.is_alpha
                        or not (t.text.islower() or t.text.istitle())
                        or not (t.has_vector or t.doc.vocab[t.lemma_].has_vector)):
                    continue
                for replacement in character_replacements(t.text, r):
                    # Vector vocabulary is a conservative early real-word veto;
                    # both dialect dictionaries remain mandatory after selection.
                    if t.doc.vocab[replacement].has_vector:
                        continue
                    if t.text.istitle():
                        replacement = replacement.capitalize()
                    out.append(Corruption(r['id'], 'spelling', t.idx-sent.start_char,
                                          t.idx-sent.start_char+len(t.text), t.text,
                                          replacement, (t.i,)))
        return out
