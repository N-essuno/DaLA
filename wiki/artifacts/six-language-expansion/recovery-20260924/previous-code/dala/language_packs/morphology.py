"""Conservative, data-driven UD constraints with independent dictionary guards.

These are candidate generators, not an independent grammaticality oracle.
Production eligibility requires a language-specific audit of each active family.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import gzip
import json
import re
from pathlib import Path

from ..character_operations import replacements
from ..edits import Corruption
from ..english_rules import select_edits
from ..profiles import resource


def permits(rule, edit):
    if rule['operator'] == 'character_edit':
        return edit.replacement.casefold() in replacements(edit.original, rule)
    return edit.replacement.casefold() in rule.get('mappings', {}).get(edit.original.casefold(), [])


def features_compatible(parsed, lexical):
    """Comma-separated UD values are alternatives, not one literal value."""
    return all(k not in lexical or set(v.split(',')) & set(lexical[k].split(',')) for k,v in parsed.items())


class MorphologyPack:
    @property
    def options(self): return self.profile.get('morphology_options',{})

    def __init__(self, profile):
        from spylls.hunspell import Dictionary
        self.profile = profile
        self.language = profile['language']
        self.exclusions = set(json.loads(resource(profile, 'exclusions').read_text()))
        root = resource(profile, 'resource_directory')
        receipt = json.loads((root / 'receipt.json').read_text())
        self.resource_receipt = receipt
        for record in receipt['files']:
            path = root / Path(record['path']).name
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Language resource checksum mismatch: {path}')
        self.dictionary = Dictionary.from_files(str(root / 'index'))
        self.normative_words = set()
        wordfile = root / 'normative-words.txt'
        if receipt.get('normative_words_sha256'):
            if hashlib.sha256(wordfile.read_bytes()).hexdigest() != receipt['normative_words_sha256']:
                raise ValueError('Normative wordlist checksum mismatch')
            self.normative_words = set(wordfile.read_text().splitlines())
        self.recognized = lru_cache(maxsize=200000)(lambda word: word in self.normative_words or self.dictionary.lookup(word))
        path = root / 'morphology.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['morphology_sha256']:
            raise ValueError('Morphology resource checksum mismatch')
        self.analyses = defaultdict(list)
        for entry in json.loads(path.read_text()):
            for form in entry['forms']: self.analyses[(form, entry['pos'])].append(entry)
        self.rule_priority = {}
        self.book = None

    def load_rulebook(self, path):
        path = Path(path)
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
            book = json.load(stream)
        if book['language'] != self.language: raise ValueError('Rulebook standard mismatch')
        if book['sources']['morphology'] != self.resource_receipt:
            raise ValueError('Rulebook was compiled against different morphology resources')
        ids = set()
        self.by_word = defaultdict(list)
        for rule in book['rules']:
            if rule['id'] in ids or not rule.get('sources') or any(s not in book['sources'] for s in rule['sources']):
                raise ValueError('Duplicate or unsourced rule')
            ids.add(rule['id'])
            if rule['operator'] != 'character_edit':
                for word in rule['mappings']: self.by_word[word].append(rule)
        self.book = book
        return book

    def paragraphs(self, document):
        # No paragraph-count cap. Preserve raw source coordinates.
        for m in re.finditer(r'[^\r\n\u2028\u2029]+', document['text']):
            raw = m.group(); text = raw.strip()
            offset = m.start() + len(raw) - len(raw.lstrip())
            limit = self.profile['curation']['max_paragraph_chars']
            cursor = 0
            while len(text) - cursor > limit:
                boundaries = list(re.finditer(r'[.!?]\s+', text[cursor:cursor+limit]))
                if not boundaries: break
                end = cursor + boundaries[-1].start() + 1
                chunk = text[cursor:end]
                if len(chunk) >= 35: yield chunk, offset + cursor
                cursor = end
                while cursor < len(text) and text[cursor].isspace(): cursor += 1
            if 35 <= len(text) - cursor <= limit:
                yield text[cursor:], offset + cursor

    def sentence_rejection(self, sent, misspellings):
        # Fill only unanimous lexical features compatible with the parser's
        # analysis. Faroese uses surface forms because its parser has no lemmas.
        for token in sent:
            if not token.is_alpha: continue
            feats = token.morph.to_dict()
            if token.pos_ in {'VERB','AUX'} and feats.get('Mood') in {'Ind','Sub','Imp'}:
                feats.setdefault('VerbForm','Fin')
            pronoun=self.profile.get('personal_pronoun_features',{}).get(token.text.casefold())
            if pronoun and token.pos_=='PRON' and feats.get('Case')=='Nom':
                if any(k in feats and feats[k]!=v for k,v in pronoun.items()):return 'source_pronoun_lexical_conflict'
                feats.update(pronoun)
            analyses = self.analyses.get((token.text.casefold(), token.pos_), [])
            if token.pos_ == 'AUX': analyses = [*analyses,*self.analyses.get((token.text.casefold(),'VERB'),[])]
            known = analyses
            compatible=[a for a in analyses if features_compatible(feats,a['features'])]
            lemmas={a['lemma'] for a in compatible}
            if (self.options.get('lemma_strategy')!='surface_unique' and len(lemmas)==1
                    and token.lemma_.casefold() not in lemmas):
                # Dictionary and parser conventions differ (den/en, langt/langur).
                # Canonicalize only when the compatible dictionary lemma is unique.
                token.lemma_=next(iter(lemmas))
            analyses=[a for a in compatible if self.options.get('lemma_strategy')=='surface_unique' or a['lemma']==token.lemma_.casefold()]
            if known and not compatible and token.pos_ in {'VERB','AUX','ADJ','DET'}:
                return 'source_morphology_conflict'
            for key in {'Gender','Number','Person','VerbForm','Definite','Degree','Voice','Case','Tense','Mood','Animacy'} - feats.keys():
                values = {a['features'].get(key) for a in analyses}
                if len(values) == 1 and None not in values: feats[key] = next(iter(values))
            token.set_morph(feats)
        # Reject independently visible source agreement errors before injecting
        # any edit elsewhere. Do not confuse possessive weak adjectives with
        # errors: only definite NOUN -> indefinite ADJECTIVE is disallowed.
        for token in sent:
            head=token.head
            if (token.pos_!='ADJ' or token.dep_.split(':')[0]!='amod' or head.pos_!='NOUN'
                    or not self.simple(token) or not self.simple(head)): continue
            left=self.analyses.get((token.text.casefold(),'ADJ'),[])
            right=self.analyses.get((head.text.casefold(),'NOUN'),[])
            if not left or not right: continue
            for feature in ['Gender','Number','Case','Definite']:
                a={x['features'].get(feature) for x in left}
                b={x['features'].get(feature) for x in right}
                if None in a or None in b: continue
                if feature=='Definite' and b!={'Def'}: continue
                if not a & b: return 'source_nominal_agreement_conflict'
        if self.options.get('source_pronoun_agreement'):
            for token in sent:
                if token.pos_ not in {'VERB','AUX'} or token.morph.get('VerbForm')!=['Fin']:continue
                head=token.head if token.dep_.split(':')[0] in {'aux','cop'} else token
                subjects=[c for c in head.children if c.dep_ in {'nsubj','nsubj:pass'}]
                if len(subjects)!=1:continue
                subject=subjects[0]
                if subject.pos_!='PRON' or subject.morph.get('Case')!=['Nom'] or not self.simple(subject):continue
                if self.options.get('complete_verb_agreement_from_subject'):
                    analyses=[a for pos in ['VERB','AUX'] for a in self.analyses.get((token.text.casefold(),pos),[])
                              if a['features'].get('VerbForm')=='Fin'
                              and all(not token.morph.get(k) or token.morph.get(k)==[v] for k,v in a['features'].items())
                              and all(not subject.morph.get(k) or a['features'].get(k) in {None,subject.morph.get(k)[0]} for k in ['Number','Person'])]
                    feats=token.morph.to_dict()
                    for key in ['Number','Person']:
                        values={a['features'].get(key) for a in analyses}
                        if len(values)==1 and None not in values:feats.setdefault(key,next(iter(values)))
                    token.set_morph(feats)
                for feature in ['Number','Person']:
                    a=subject.morph.get(feature);b=token.morph.get(feature)
                    if a and b and a!=b:return 'source_pronoun_agreement_conflict'
        text = sent.text
        if hashlib.sha256(text.encode()).hexdigest() in self.exclusions: return 'audited_source_exclusion'
        words = [t for t in sent if t.is_alpha]
        c = self.profile['curation']
        if not c['min_words'] <= len(words) <= c['max_words'] or not 35 <= len(text) <= c['max_sentence_chars']: return 'length'
        if not text[0].isupper() or text[-1] != '.': return 'fragment_or_format'
        if re.search(r'\b[^\W\d_]\.$',text) and text[-2].isupper():return 'truncated_initial_reference'
        if re.search(r'\s+[.,;:!?]|[,;:!?][^\W\d_]', text): return 'punctuation_spacing'
        if any(re.search(pattern, text) for pattern in self.profile['curation'].get('source_risk_patterns', [])):
            return 'source_extraction_risk'
        if re.search(r'[\[\]<>|~\\\n\r\u00AD\uFFFD]|https?://|www\.|[a-zà-öø-ÿ][A-Z]|\d[^\W\d_]', text): return 'source_extraction_risk'
        if any(t.is_quote or t.like_url or t.like_email or t.pos_ in {'X', 'SYM'} for t in sent): return 'nonprose_or_quotation'
        roots = [t for t in sent if t.dep_ == 'ROOT']
        if len(roots) != 1: return 'no_unique_root'
        root = roots[0]
        children = list(root.children)
        if not any(t.dep_.split(':')[0] in {'nsubj', 'csubj'} for t in children): return 'no_main_subject'
        if any(t.dep_ == 'mark' for t in children): return 'standalone_subordinate'
        finite = [t for t in [root, *children] if t.morph.get('VerbForm') == ['Fin']]
        if not finite: return 'no_finite_main_verb'
        # Names are excluded from corruption and may be absent from a dictionary.
        if any(not self.recognized(t.text.casefold()) for t in words if t.pos_ != 'PROPN'):
            return 'source_dictionary_unknown'
        if any(t.text.casefold() in self.profile.get('standard_exclusions', []) for t in words):
            return 'other_written_standard'
        return None

    def select_edits(self, text, edits, seed, max_errors):
        priority = self.profile['selection']['priority']
        if self.profile['selection'].get('strategy') == 'hash_rotate_grammar_then_spelling':
            grammar = sorted((f for f in priority if f != 'spelling'),
                             key=lambda f: hashlib.sha256(f'{seed}\0{text}\0{f}'.encode()).digest())
            priority = grammar + ['spelling']
        return select_edits(text, edits, seed, max_errors, priority=priority)

    @staticmethod
    def simple(token):
        return not any(c.dep_.split(':')[0] in {'conj', 'cc', 'appos'} for c in token.children)

    def licensed_context(self, token, rule, pair):
        feature = rule.get('feature')
        before = pair.get('before', {})
        after = pair.get('after', {})
        if pair.get('lemma'):
            if self.options.get('lemma_strategy')=='surface_unique':
                positions=[token.pos_]+(['VERB'] if token.pos_=='AUX' else [])
                lemmas = {a['lemma'] for pos in positions for a in self.analyses.get((token.text.casefold(), pos), [])}
                if lemmas != {pair['lemma']}: return False
            elif token.lemma_.casefold() != pair['lemma']:
                return False
        if token.pos_ not in rule['pos'] or any(not set(token.morph.get(k)) & set(v.split(',')) for k, v in before.items()): return False
        if not self.simple(token): return False
        kind = rule['context']
        if kind == 'nominal_agreement':
            head = token.head
            if token.dep_.split(':')[0] not in rule['relations'] or head.pos_ != 'NOUN' or not self.simple(head): return False
            # No coordinated noun phrases, possessive ellipsis, or cross-standard gender conversion.
            if before[feature] not in head.morph.get(feature): return False
            if feature == 'Gender' and any(set((before[feature],after[feature])) <= set(group)
                    for group in self.options.get('gender_equivalences',[])):return False
            if feature == 'Gender' and head.morph.get('Number') != ['Sing']: return False
            if feature == 'Definite' and before[feature] == 'Ind':
                # Weak adjectives can have licensed bare/possessive uses. Only
                # inject weak declension after an explicit indefinite article.
                articles=self.profile.get('indefinite_articles',[])
                if not any(c.dep_=='det' and c.text.casefold() in articles for c in head.children):return False
            # A parser chooses one analysis; that does not exclude other valid
            # readings (e.g. Swedish singular/plural "hus" or "resultat").
            head_analyses = self.analyses.get((head.text.casefold(), 'NOUN'), [])
            values = {a['features'].get(feature) for a in head_analyses}
            if (not head_analyses or None in values or before[feature] not in values
                    or after[feature] in values): return False
            return True
        if kind == 'finite_agreement':
            if token.morph.get('VerbForm') != ['Fin']: return False
            subjects = [t for t in token.children if t.dep_ == 'nsubj']
            if token.dep_.split(':')[0] in {'aux', 'cop'}: subjects = [t for t in token.head.children if t.dep_ == 'nsubj']
            if len(subjects) != 1: return False
            subject = subjects[0]
            return (subject.pos_ == 'PRON' and subject.text.casefold() in rule['subjects'] and self.simple(subject)
                    and subject.morph.get(feature) == token.morph.get(feature))
        if kind == 'governed_verb':
            controllers = [t for t in token.children if t.dep_.split(':')[0] in {'aux', 'mark'}]
            if token.dep_.split(':')[0] == 'xcomp': controllers.append(token.head)
            return any(t.text.casefold() in rule['controllers'] for t in controllers)
        if kind == 'preposition_case':
            markers = [t for t in token.children if t.dep_ == 'case']
            return len(markers) == 1 and markers[0].text.casefold() in rule['controllers'] and token.dep_.split(':')[0] in {'obl', 'nmod'}
        if kind == 'subject_pronoun':
            return (token.dep_ == 'nsubj' and token.head.pos_ in {'VERB', 'AUX'} and self.simple(token.head)
                    and (not rule.get('controllers') or token.head.lemma_.casefold() in rule['controllers']))
        return False

    def candidates(self, sent, book):
        out = []
        for token in sent:
            if not token.is_alpha or token.pos_ in {'PROPN', 'X'} or token.ent_type_: continue
            if token.text[0].isupper() and token.idx != sent.start_char: continue
            original = token.text.casefold()
            if not self.recognized(original): continue
            for rule in self.by_word.get(original, []):
                for pair in rule['pairs'].get(original, []):
                    replacement = pair['replacement']
                    if replacement == original or not self.recognized(replacement) or not self.licensed_context(token, rule, pair): continue
                    out.append(self.edit(sent, token, replacement, rule))
            if token.pos_ not in {'NOUN', 'ADJ', 'VERB', 'ADV'}: continue
            for rule in book['rules']:
                if rule['operator'] != 'character_edit': continue
                candidates = [s for s in replacements(original, rule) if not self.recognized(s)]
                if candidates:
                    # Deterministic variation over the sentence and token, not only alphabetical choice.
                    replacement = min(candidates, key=lambda s: hashlib.sha256(f'{sent.text}\0{token.i}\0{s}'.encode()).digest())
                    out.append(self.edit(sent, token, replacement, rule))
        return out

    @staticmethod
    def edit(sent, token, replacement, rule):
        if token.text.istitle(): replacement = replacement.capitalize()
        elif token.text.isupper(): replacement = replacement.upper()
        start = token.idx - sent.start_char
        return Corruption(rule['id'], rule['family'], start, start + len(token.text), token.text, replacement,
                          tuple(sorted({token.i, token.head.i, *(t.i for t in token.children)})))
