"""Dutch contextual guards; inventories and resource choices live in the profile."""
import hashlib
import json
import re
from pathlib import Path

from ..edits import Corruption
from ..english_rules import select_edits
from ..character_operations import replacements
from ..profiles import resource


def permits(rule, edit):
    if rule.get('operator') == 'character_edit':
        return edit.replacement.lower() in replacements(edit.original, rule)
    if rule.get('operator') == 'dictionary_mapping':
        return rule['mappings'].get(edit.original.lower()) == edit.replacement.lower()
    return rule['correct'] == edit.original.lower() and rule['incorrect'] == edit.replacement.lower()


class DutchPack:
    def __init__(self, profile):
        self.profile = profile
        self.grammar = profile['grammar']
        self.exclusions = {r['sentence_sha256'] for r in json.loads(resource(profile, 'exclusions').read_text())}
        self.boilerplate = re.compile(profile['curation']['boilerplate_regex'], re.I)
        self.rule_priority = {}
        lexical_path = resource(profile, 'lexicon')
        if hashlib.sha256(lexical_path.read_bytes()).hexdigest() != profile['lexicon_sha256']:
            raise ValueError('Dutch lexicon checksum mismatch')
        self.words = {w.casefold() for w in lexical_path.read_text().splitlines()}
        self.morphology = None
        self.noun_articles = self.grammar['noun_articles']
        if profile.get('morphology'):
            path = resource(profile, 'morphology')
            if hashlib.sha256(path.read_bytes()).hexdigest() != profile['morphology_sha256']:
                raise ValueError('Dutch morphology checksum mismatch')
            self.morphology = json.loads(path.read_text())
            self.noun_articles = self.morphology['noun_articles']
        self.loaded_book = None


    def load_rulebook(self, path):
        book = json.loads(Path(path).read_text())
        ids = set()
        for rule in book['rules']:
            if rule['id'] in ids or rule['family'] not in self.profile['selection']['priority']:
                raise ValueError('Invalid/duplicate Dutch rule')
            if not rule.get('sources') or any(s not in book['sources'] for s in rule['sources']):
                raise ValueError('Dutch rule lacks source evidence')
            if rule.get('operator') == 'character_edit' and not rule.get('requires_lexical_screen'):
                raise ValueError('Character errors need lexical screening')
            if rule['family'] == 'spelling':
                self.rule_priority[rule['id']] = 1 if rule.get('operator') == 'character_edit' else 0
            ids.add(rule['id'])
        self.loaded_book = book
        self.lookup = self.rule_lookup(book)
        return book

    @staticmethod
    def rule_lookup(book):
        rules = {}
        for r in book['rules']:
            if r.get('operator') == 'character_edit':
                continue
            for word in r.get('mappings', {r.get('correct'): r.get('incorrect')}):
                rules.setdefault((r['family'], word), []).append(r)
        return rules

    def paragraphs(self, document):
        blocks = []
        for match in re.finditer(r'[^\r\n\u2028\u2029]+', document['text']):
            raw = match.group(); text = raw.strip()
            if not 50 <= len(text) <= self.profile['curation'].get('max_paragraph_chars', 6000) or self.boilerplate.search(text):
                continue
            if any(x in text for x in ('|', '\ufffd', '<', '>', '\\[')):
                continue
            blocks.append((text, match.start() + len(raw) - len(raw.lstrip())))
        cap = self.profile['curation'].get('max_paragraphs_per_document')
        if cap and len(blocks) > cap:
            blocks = sorted(blocks, key=lambda b: hashlib.sha256(
                f"{document['document_id']}\0{b[1]}\0{b[0]}".encode()).digest())[:cap]
        yield from sorted(blocks, key=lambda b: b[1])

    def sentence_rejection(self, sent, misspellings):
        text = sent.text
        if hashlib.sha256(text.encode()).hexdigest() in self.exclusions:
            return 'audited_source_exclusion'
        words = [t for t in sent if t.is_alpha]
        if not 8 <= len(words) <= self.profile['curation'].get('max_words', 40) or not 40 <= len(text) <= self.profile['curation'].get('max_sentence_chars', 350):
            return 'length'
        if not text[0].isupper() or text[-1] != '.' or any(c in text for c in '\n\r\u2028\u2029'):
            return 'fragment_or_format'
        if any(c in text for c in '"“”«»…') or '...' in text or any(t.is_quote for t in sent):
            return 'quotation_or_ellipsis'
        for pattern in self.profile['curation'].get('source_risk_patterns', []):
            if re.search(pattern, text):
                return 'source_extraction_risk'
        if re.search(self.profile['curation']['punctuation_without_space_regex'], text):
            return 'punctuation_spacing'
        if self.boilerplate.search(text) or any(t.like_url or t.like_email for t in sent):
            return 'boilerplate'
        if any(t.lower_ in misspellings for t in words):
            return 'known_misspelling'
        if any(t.pos_ in {'X', 'SYM'} for t in sent):
            return 'nonprose'
        roots = [t for t in sent if t.dep_ == 'ROOT']
        if len(roots) != 1 or roots[0].pos_ not in {'VERB', 'AUX'}:
            return 'no_verbal_root'
        root = roots[0]
        if any(t.dep_ == 'mark' and t.head == root for t in sent):
            return 'standalone_subordinate_clause'
        if not any(t.dep_.split(':')[0] in {'nsubj', 'csubj'} and t.head == root for t in sent):
            return 'no_main_subject'
        if not any('Fin' in t.morph.get('VerbForm') and (t == root or t.head == root and t.dep_.split(':')[0] in {'aux', 'cop'}) for t in sent):
            return 'no_finite_main_verb'
        if self.profile['curation'].get('strict_source_grammar'):
            if re.search(self.profile['curation']['fragment_prefix_regex'], text):
                return 'fragment_risk'
            for token in sent:
                clause = token.head
                expected_number = self.profile['curation'].get('source_pronoun_numbers', {}).get(token.lower_)
                if (expected_number and token.pos_ == 'PRON' and token.dep_ == 'nsubj'
                        and not any(c.dep_.split(':')[0] in {'conj', 'cc'} for c in token.children)
                        and not any(c.dep_.split(':')[0] in {'conj', 'cc'} or c.dep_ == 'nsubj' and c != token for c in clause.children)):
                    finite = [v for v in [clause, *clause.children]
                              if v.morph.get('VerbForm') == ['Fin']
                              and (v == clause or v.dep_.split(':')[0] in {'aux', 'cop'})]
                    if len(finite) == 1 and finite[0].morph.get('Number') and finite[0].morph.get('Number') != [expected_number]:
                        return 'source_pronoun_disagreement'
                if token.dep_ == 'mark' and token.lower_ in {'dat', 'omdat', 'terwijl', 'hoewel', 'doordat', 'zodat'}:
                    if not any(v.morph.get('VerbForm') == ['Fin'] for v in [clause, *[c for c in clause.children if c.dep_.split(':')[0] in {'aux', 'cop'}]]):
                        return 'missing_subordinate_finite_verb'
                if token.dep_ == 'nsubj' and token.pos_ == 'NOUN' and token.morph.get('Number') == ['Plur']:
                    finite = [v for v in [clause, *clause.children] if v.morph.get('VerbForm') == ['Fin']
                              and (v == clause or v.dep_.split(':')[0] in {'aux', 'cop'})]
                    if len(finite) == 1 and finite[0].morph.get('Number') == ['Sing']:
                        return 'source_number_disagreement'
        return None

    def select_edits(self, text, edits, seed, max_errors):
        return select_edits(text, edits, seed, max_errors,
                            priority=self.profile['selection']['priority'], rule_priority=self.rule_priority)

    def candidates(self, sent, book):
        rules = self.lookup if book is self.loaded_book else self.rule_lookup(book)
        out = []

        def add(family, token, protected=(), context=None):
            if not token.is_alpha or token.ent_type_ or token.pos_ == 'PROPN':
                return
            if not (token.text.islower() or token.text.istitle()):
                return
            for rule in rules.get((family, token.lower_), []):
                if context is not None and rule.get('context') != context:
                    continue
                replacement = rule['mappings'][token.lower_] if rule.get('operator') == 'dictionary_mapping' else rule['incorrect']
                if family == 'spelling' and (token.lower_ not in self.words or replacement in self.words):
                    continue
                if token.text.istitle():
                    replacement = replacement.capitalize()
                out.append(Corruption(rule['id'], family, token.idx-sent.start_char,
                                      token.idx-sent.start_char+len(token.text), token.text, replacement,
                                      tuple(sorted({token.i, *(p.i for p in protected)}))))

        for t in sent:
            if t.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}:
                add('spelling', t)
            # Determiner must attach to a nearby common noun; only licensed adjectival modifiers may intervene.
            # Singular nouns require an explicit invariant-gender allowlist.
            h = t.head
            if (t.pos_ in {'DET', 'PRON'} and t.dep_ in {'det', 'nmod:poss'} and h.pos_ == 'NOUN'
                    and 1 <= h.i-t.i <= 1+self.grammar.get('max_intervening_adjectives', 0)
                    and all(x.pos_ == 'ADJ' and x.head == h for x in sent.doc[t.i+1:h.i])
                    and not (self.grammar.get('reject_infinitive_objects') and self.morphology and h.head.lower_ in self.morphology['verb_inflections'] and (h.head.morph.get('VerbForm') != ['Fin'] or h.head.dep_ == 'csubj'))
                    and not h.ent_type_ and not any(c.dep_.split(':')[0] in {'conj', 'cc', 'nummod', 'compound'} for c in h.children)):
                number = h.morph.get('Number')
                if (number == ['Plur'] and t.lower_ in {'de', 'deze', 'die'}
                        and (not self.grammar.get('require_plural_lemma_change') or h.lower_ != h.lemma_.lower())
                        and (not self.grammar.get('plural_determiners_require_plural_subject') or
                             h.dep_ == 'nsubj' and any(
                                 v.morph.get('VerbForm') == ['Fin'] and v.morph.get('Number') == ['Plur']
                                 and (v == h.head or v.dep_.split(':')[0] in {'aux', 'cop'})
                                 for v in [h.head, *h.head.children]))):
                    family = 'article_number' if t.lower_ == 'de' else 'demonstrative_agreement'
                    add(family, t, [h], 'plural')
                if number == ['Sing']:
                    article = self.noun_articles.get(h.lower_ if self.morphology else h.lemma_.lower())
                    expected_gender = 'Neut' if article == 'het' else 'Com'
                    if article and h.morph.get('Gender') == [expected_gender]:
                        if t.lower_ == article:
                            add('article_gender', t, [h], article)
                        if t.lower_ in ({'dit', 'dat'} if article == 'het' else {'deze', 'die'}):
                            add('demonstrative_agreement', t, [h], article)
                        if t.lower_ == ('ons' if article == 'het' else 'onze'):
                            add('possessive_agreement', t, [h], article)
            # A non-ambiguous pronoun before one adjacent finite verb. Inverted
            # forms, u, zij/ze and coordinated subjects are deliberately absent.
            if (t.pos_ == 'PRON' and t.dep_ == 'nsubj' and t.lower_ in (set(self.grammar['pronoun_verbs']) | set(self.grammar.get('productive_pronouns', {})))
                    and not list(t.children) and t.i < h.i
                    and not any(c.dep_.split(':')[0] in {'conj', 'cc', 'mark'} for c in h.children)):
                finite = [v for v in [h, *h.children] if v.pos_ in {'VERB', 'AUX'}
                          and v.morph.get('VerbForm') == ['Fin'] and v.morph.get('Tense') == ['Pres']
                          and (v == h or v.dep_.split(':')[0] in {'aux', 'cop'})]
                if len(finite) == 1 and finite[0].i == t.i+1:
                    v = finite[0]
                    allowed = self.grammar['pronoun_verbs'].get(t.lower_, {})
                    if v.lower_ in allowed:
                        family = 'verb_dt' if (v.lower_, allowed[v.lower_]) in {('word','wordt'),('wordt','word'),('vind','vindt'),('vindt','vind')} else 'subject_verb_agreement'
                        add(family, v, [t], t.lower_)
                    context = self.grammar.get('productive_pronouns', {}).get(t.lower_)
                    if context:
                        add('subject_verb_agreement', v, [t], context)
            if self.morphology and t.pos_ == 'ADJ' and t.dep_ == 'amod' and h.pos_ == 'NOUN' and h.i == t.i+1 and t.i > sent.start:
                det = sent.doc[t.i-1]
                if (det.lower_ == 'een' and det.dep_ == 'det' and det.head == h
                        and not h.ent_type_ and h.morph.get('Number') == ['Sing']
                        and self.noun_articles.get(h.lower_) == 'het' and h.morph.get('Gender') == ['Neut']
                        and not any(c.dep_.split(':')[0] in {'conj', 'cc', 'compound'} for c in h.children)):
                    add('adjective_inflection', t, [det, h])
            if t.lower_ == 'dat' and t.pos_ == 'PRON' and t.dep_ in {'nsubj', 'obj'} and h.dep_ == 'acl:relcl':
                antecedent = h.head
                if (antecedent.pos_ == 'NOUN' and not antecedent.ent_type_ and t.i == antecedent.i+1
                        and antecedent.lower_ in self.grammar.get('relative_inanimate_nouns', [])
                        and antecedent.morph.get('Number') == ['Sing']
                        and self.noun_articles.get(antecedent.lower_) == 'het'
                        and any(c.dep_ == 'det' and c.lower_ in {'het', 'dit', 'dat'} for c in antecedent.children)):
                    add('relative_pronoun_agreement', t, [antecedent, h])
        # Productive spelling is independently checked as recognized->nonword,
        # then checked in context. No parser-only certification of spelling.
        for rule in book['rules']:
            if rule.get('operator') != 'character_edit':
                continue
            for t in sent:
                if (t.ent_type_ or t.pos_ not in rule['allowed_pos'] or not t.is_alpha
                        or not t.text.islower() or t.lower_ not in self.words
                        or not (t.has_vector or t.doc.vocab[t.lemma_].has_vector)):
                    continue
                for replacement in replacements(t.text, rule):
                    if replacement in self.words or t.doc.vocab[replacement].has_vector:
                        continue
                    out.append(Corruption(rule['id'], 'spelling', t.idx-sent.start_char,
                                          t.idx-sent.start_char+len(t.text), t.text, replacement, (t.i,)))
        return out
