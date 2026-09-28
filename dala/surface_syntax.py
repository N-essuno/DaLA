"""Exact source spans with a separate expanded syntactic graph.

Expanded words are never editable. No artificial tokenization is exported.
"""
from types import SimpleNamespace
from spacy.tokens import Doc


class SurfaceToken:
    def __init__(self, token, owner, span, protected):
        self.token, self.owner = token, owner
        self.idx, self.end = span
        self.surface_protected = protected

    def __getattr__(self, key):
        return getattr(self.token, key)

    @property
    def lemma_(self): return self.token.lemma_

    @lemma_.setter
    def lemma_(self, value): self.token.lemma_ = value

    @property
    def head(self): return self.owner.tokens[self.token.head.i]

    @property
    def children(self): return (self.owner.tokens[t.i] for t in self.token.children)


class SurfaceSentence:
    def __init__(self, text, sentence, vocab, token_base):
        self.pre_rejection = getattr(sentence, "dala_rejection", None)
        self.start_char = sentence.tokens[0].start_char
        self.end_char = sentence.tokens[-1].end_char
        self.text = text[self.start_char:self.end_char]
        records = []
        previous = self.start_char
        for surface in sentence.tokens:
            a, b = surface.start_char, surface.end_char
            if a is None or b is None or a < previous or b <= a or text[a:b] != surface.text:
                raise ValueError('Invalid source token offsets')
            previous = b
            protected = len(surface.words) != 1
            for word in surface.words:
                if not protected and word.text != surface.text:
                    raise ValueError('Single word differs from source token')
                records.append((word, (a, b), protected))
        indices = {w.id: i for i, (w, _, _) in enumerate(records)}
        if len(indices) != len(records): raise ValueError('Duplicate syntactic word ID')
        syntax = Doc(vocab, words=[w.text for w, _, _ in records],
                     heads=[indices[w.head] if w.head else i for i, (w, _, _) in enumerate(records)],
                     deps=[w.deprel if w.head else 'ROOT' for w, _, _ in records],
                     pos=[w.upos or 'X' for w, _, _ in records],
                     lemmas=[w.lemma or w.text for w, _, _ in records],
                     morphs=[w.feats or '' for w, _, _ in records])
        self.tokens = [SurfaceToken(syntax[i], self, span, protected)
                       for i, (_, span, protected) in enumerate(records)]
        self.start, self.end = token_base, token_base + len(records)
        self.doc = SimpleNamespace(ents=())

    def __iter__(self): return iter(self.tokens)
    def __len__(self): return len(self.tokens)
    def __getitem__(self, key): return self.tokens[key]


class SurfaceDocument:
    def __init__(self, text, sentences, vocab):
        self.doc = SimpleNamespace(text=text)
        self.sents = []
        base = 0
        for sentence in sentences:
            span = SurfaceSentence(text, sentence, vocab, base)
            self.sents.append(span)
            base = span.end
