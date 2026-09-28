"""Parser selection without changing existing spaCy language-pack behavior."""
from importlib.metadata import version
from collections import Counter
import time


def load_parser(profile, model=None):
    model = model or profile['parser']
    if profile.get('parser_backend') == 'conllu':
        from .conllu_source import AnnotatedParser
        return AnnotatedParser(profile)
    if profile.get('parser_backend') == 'stanza':
        return StanzaParser(profile)
    import spacy
    nlp = spacy.load(model)
    required = profile.get('parser_requirements', {'components': ['parser', 'tagger', 'lemmatizer', 'ner'], 'vectors': True})
    if not set(required['components']).issubset(nlp.pipe_names) or required.get('vectors') and not nlp.vocab.vectors_length:
        raise ValueError('Parser does not meet language-pack requirements')
    return nlp


def parser_metadata(profile, model=None):
    model = model or profile['parser']
    if profile.get('parser_backend') == 'conllu':
        return dict(package='annotated-conllu', version='conllu-v2-pinned', language=profile['language'])
    if profile.get('parser_backend') == 'stanza':
        return dict(package='stanza', version=version('stanza'), language=profile['language'],
                    model=profile['parser_settings'], spacy=version('spacy'))
    return dict(package=model, version=version(model), spacy=version('spacy'))


class StanzaParser:
    """Expose exact-offset spaCy docs to existing adapters and task exporters.

    Unalignable multiword expansions fail closed; text is never rewritten to
    match parser tokenization. No named-entity exemptions are inferred.
    """
    def __init__(self, profile):
        import spacy
        import stanza
        import torch
        torch.set_num_threads(profile.get('parser_threads', 2))
        settings = profile['parser_settings']
        from pathlib import Path
        from .common_pile import sha256_file
        for filename, digest in settings.get('model_sha256', {}).items():
            if sha256_file(Path(settings['directory']) / filename) != digest:
                raise ValueError(f'Parser model checksum mismatch: {filename}')
        self.nlp = stanza.Pipeline(lang=settings.get('language', profile['language']), dir=settings['directory'],
            processors=settings['processors'], package=None, use_gpu=False,
            download_method=None, verbose=False)
        self.vocab = spacy.blank('xx').vocab
        self.meta = {'version': version('stanza')}
        self.rejections = {}
        self.max_sentence_tokens = profile.get('parser_max_sentence_tokens', 80)
        self.expand_mwt = profile.get('parser_expand_mwt', False)
        self.skip_unalignable_sentences = profile.get('parser_skip_unalignable_sentences', False)
        self.recover_mwt = profile.get('parser_recover_mwt', False)
        self.prefilter_text = profile.get('parser_prefilter_text', False)
        self.curation = profile.get('curation', {})
        self.stage_seconds = Counter()
        if self.prefilter_text and not self.recover_mwt:
            raise ValueError('Text prefilter requires the source-span adapter')

    def __call__(self, text):
        from spacy.tokens import Doc
        processors = ['tokenize']
        if self.expand_mwt and 'mwt' in self.nlp.processors:
            processors.append('mwt')
        started = time.monotonic()
        parsed = self.nlp(text, processors=processors)
        if hasattr(self, 'stage_seconds'): self.stage_seconds['tokenize'] += time.monotonic() - started
        if self.expand_mwt and not self.skip_unalignable_sentences and not getattr(self, 'recover_mwt', False) and any(len(t.words) != 1 for s in parsed.sentences for t in s.tokens):
            self.rejections['parser_unalignable_paragraph'] = self.rejections.get('parser_unalignable_paragraph', 0) + 1
            return None
        kept = [s for s in parsed.sentences if len(s.words) <= self.max_sentence_tokens]
        rejected = len(parsed.sentences) - len(kept)
        if rejected:
            self.rejections['parser_overlength_sentence'] = self.rejections.get('parser_overlength_sentence', 0) + rejected
            import stanza
            parsed = stanza.Document([sentence.to_dict() for sentence in kept], text=text)
        if not kept: return None
        early = []
        if getattr(self, 'prefilter_text', False):
            from .source_screen import source_text_rejection
            eligible = []
            for sentence in parsed.sentences:
                a, b = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
                reason = source_text_rejection(text[a:b], self.curation) if a is not None and b is not None else None
                if reason:
                    sentence.dala_rejection = reason
                    # Keep each syntactic word and exact source span, so later
                    # sentence offsets and token identities do not shift.
                    for index, word in enumerate(sentence.words):
                        word.head = 0 if index == 0 else sentence.words[0].id
                        word.deprel = 'root' if index == 0 else 'dep'
                        word.upos = 'X'; word.lemma = word.text; word.feats = ''
                    early.append(sentence)
                    self.rejections['parser_prefilter_' + reason] = self.rejections.get('parser_prefilter_' + reason, 0) + 1
                else:
                    eligible.append(sentence)
            if early:
                import stanza
                parsed = stanza.Document([sentence.to_dict() for sentence in eligible], text=text)
        started = time.monotonic()
        if parsed.sentences:
            parsed = self.nlp(parsed, processors=['pos','lemma','depparse'])
        if hasattr(self, 'stage_seconds'): self.stage_seconds['syntax'] += time.monotonic() - started
        if early:
            from types import SimpleNamespace
            parsed = SimpleNamespace(sentences=sorted([*parsed.sentences, *early], key=lambda s: s.tokens[0].start_char))
        if getattr(self, 'recover_mwt', False) and (getattr(self, 'prefilter_text', False) or any(len(t.words) != 1 for s in parsed.sentences for t in s.tokens)):
            from .surface_syntax import SurfaceDocument
            try:
                return SurfaceDocument(text, parsed.sentences, self.vocab)
            except (ValueError, KeyError):
                self.rejections['parser_invalid_surface_graph'] = self.rejections.get('parser_invalid_surface_graph', 0) + 1
                return None
        entries = []
        cursor = 0
        for sentence in parsed.sentences:
            if self.skip_unalignable_sentences and any(len(t.words) != 1 for t in sentence.tokens):
                # Preserve the entire unalignable sentence as an opaque X root.
                # MorphologyPack rejects X sentences; neighboring sentences keep
                # their exact source offsets and independent dependency trees.
                from types import SimpleNamespace
                start, end = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
                if start is None or end is None: return None
                if start > cursor:
                    entries.append((text[cursor:start], None, sentence, False))
                surface = text[start:end]
                opaque = SimpleNamespace(id=1, head=0, deprel='root', upos='X', lemma=surface, feats='')
                entries.append((surface, opaque, sentence, True))
                cursor = end
                self.rejections['parser_unalignable_sentence'] = self.rejections.get('parser_unalignable_sentence', 0) + 1
                continue
            for index, token in enumerate(sentence.tokens):
                if len(token.words) != 1 or token.start_char is None or token.end_char is None:
                    self.rejections['parser_unalignable_paragraph'] = self.rejections.get('parser_unalignable_paragraph', 0) + 1
                    return None
                if text[token.start_char:token.end_char] != token.text:
                    self.rejections['parser_unalignable_paragraph'] = self.rejections.get('parser_unalignable_paragraph', 0) + 1
                    return None
                if token.start_char > cursor:
                    entries.append((text[cursor:token.start_char], None, sentence, False))
                entries.append((token.text, token.words[0], sentence, index == 0))
                cursor = token.end_char
        if cursor < len(text): entries.append((text[cursor:], None, None, False))
        if not entries: return None
        indices = {(id(s), w.id): i for i, (_, w, s, _) in enumerate(entries) if w}
        words, heads, deps, pos, lemmas, morphs, starts = [], [], [], [], [], [], []
        for i, (surface, w, sentence, first) in enumerate(entries):
            words.append(surface); starts.append(first if w else False)
            if w:
                heads.append(indices[(id(sentence), w.head)] if w.head else i)
                deps.append('ROOT' if not w.head else w.deprel)
                pos.append(w.upos or 'X'); lemmas.append(w.lemma or surface)
                morphs.append(w.feats or '')
            else:
                heads.append(max(0, i - 1)); deps.append('dep'); pos.append('SPACE'); lemmas.append(surface); morphs.append('')
        starts[0] = True
        doc = Doc(self.vocab, words=words, spaces=[False] * len(words), heads=heads,
                  deps=deps, pos=pos, lemmas=lemmas, morphs=morphs, sent_starts=starts)
        if doc.text != text: raise ValueError('Parser offset reconstruction failed')
        return ParsedDocument(doc)

    def pipe(self, inputs, as_tuples=False, batch_size=64):
        for item in inputs:
            text, context = item if as_tuples else (item, None)
            doc = self(text)
            if doc is not None:
                yield (doc, context) if as_tuples else doc


class ParsedDocument:
    def __init__(self, doc): self.doc = doc

    @property
    def sents(self):
        for span in self.doc.sents:
            start, end = span.start, span.end
            while start < end and self.doc[start].pos_ == 'SPACE': start += 1
            while end > start and self.doc[end - 1].pos_ == 'SPACE': end -= 1
            if start < end: yield self.doc[start:end]
