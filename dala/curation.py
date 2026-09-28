"""Conservative source-sentence filters. They do not certify human acceptability."""
import re
import hashlib
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).resolve().parents[1] / "config/english_source_exclusions.json"
SOURCE_EXCLUSIONS = {r["sentence_sha256"] for r in json.loads(EXCLUSIONS_PATH.read_text())}
from difflib import SequenceMatcher

BOILERPLATE = re.compile(r'^(?:by |text by |published |last updated|image |photo |figure |table |read more|related |references|copyright|about the author)|https?://|www\.|creative commons|all rights reserved|republish|\b(?:exercise|worksheet|fill in the blanks)\b', re.I)
QUOTES = re.compile(r'["“”«»`]|(?<!\w)[\x27‘’]|[\x27‘’](?!\w)')


def paragraphs(document):
    """Yield unchanged single-line prose blocks with original document offsets."""
    for match in re.finditer(r'[^\r\n]+', document['text']):
        raw = match.group()
        content = raw.strip()
        start = match.start() + len(raw) - len(raw.lstrip())
        if not 60 <= len(content) <= 6000 or BOILERPLATE.search(content):
            continue
        if any(x in content for x in ('\\(', '\\[', '|', '\ufffd', '<', '>')):
            continue
        yield content, start


def sentence_rejection(sent, known_misspellings):
    text = sent.text
    if hashlib.sha256(text.encode()).hexdigest() in SOURCE_EXCLUSIONS:
        return "audited_source_exclusion"
    words = [t for t in sent if t.is_alpha]
    if not 8 <= len(words) <= 45 or not 40 <= len(text) <= 400:
        return 'length'
    if not text[0].isupper() or text[-1] not in '.?!' or '\n' in text:
        return 'fragment_or_format'
    if QUOTES.search(text) or '...' in text or '…' in text:
        return 'quotation_or_ellipsis'
    if BOILERPLATE.search(text) or any(t.like_url or t.like_email for t in sent):
        return 'boilerplate'
    if any(t.lower_ in known_misspellings for t in words):
        return 'known_misspelling'
    if any(t.pos_ in {'X', 'SYM'} for t in sent):
        return 'nonprose'
    roots = [t for t in sent if t.dep_ == 'ROOT']
    if len(roots) != 1 or roots[0].pos_ not in {'VERB', 'AUX'}:
        return 'no_verbal_root'
    root = roots[0]
    if not any(t.dep_ in {'nsubj', 'nsubjpass', 'csubj'} and t.head == root for t in sent):
        return 'no_main_subject'
    if not any(t.tag_ in {'VBP', 'VBZ', 'VBD', 'MD'} and (t == root or t.head == root and t.dep_ in {'aux', 'auxpass'}) for t in sent):
        return 'no_finite_main_verb'
    # Broad vocabulary screen, not a dictionary proof. Names/recognized entities
    # and inflected forms with known lemmas are allowed. Record every rejection.
    if any(len(t.text) > 2 and t.pos_ != 'PROPN' and not t.ent_type_
           and not t.has_vector and not t.doc.vocab[t.lemma_].has_vector for t in words):
        return 'unknown_non_name_word'
    # Reject obvious number mismatch already present in the clean source.
    for t in sent:
        if t.dep_ == 'det' and t.head.pos_ == 'NOUN':
            if t.lower_ in {'this', 'that', 'a', 'an'} and t.head.tag_ == 'NNS':
                return 'existing_determiner_mismatch'
            if t.lower_ in {'these', 'those'} and t.head.tag_ == 'NN':
                return 'existing_determiner_mismatch'
    return None


def normalized_tokens(text):
    return tuple(re.findall(r'\w+', text.casefold()))


class NearDuplicates:
    """Reject exact normalized matches or >=90% token-sequence similarity.

Candidate comparisons share a token trigram; this is a documented heuristic,
not a guarantee of semantic uniqueness. Long/short pairs below the possible
similarity bound are excluded before running SequenceMatcher.
"""
    def __init__(self):
        self.tokens = []
        self.index = {}
        self.exact = set()

    def add(self, text):
        tokens = normalized_tokens(text)
        if tokens in self.exact:
            return False
        grams = set(zip(tokens, tokens[1:], tokens[2:]))
        neighbours = set()
        for gram in grams:
            neighbours.update(self.index.get(gram, ()))
        for i in neighbours:
            other = self.tokens[i]
            if 2 * min(len(tokens), len(other)) / (len(tokens) + len(other)) >= .9:
                if SequenceMatcher(None, tokens, other, autojunk=False).ratio() >= .9:
                    return False
        i = len(self.tokens)
        self.tokens.append(tokens)
        self.exact.add(tokens)
        for gram in grams:
            self.index.setdefault(gram, []).append(i)
        return True
