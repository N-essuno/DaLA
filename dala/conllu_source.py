"""Pinned annotated prose for bounded pilots; shared by all language packs.

Keep original text, reject unalignable multiword tokens and annotated mistakes.
Never treat sentence IDs as proof of independent source documents.
"""
import hashlib
import json
from pathlib import Path
from .common_pile import sha256_file


def records(path):
    document = None
    for block in Path(path).read_text().strip().split('\n\n'):
        meta, rows = {}, []
        for line in block.splitlines():
            if line.startswith('# ') and ' = ' in line:
                key, value = line[2:].split(' = ', 1); meta[key] = value
            elif line and not line.startswith('#'):
                fields = line.split('\t')
                if len(fields) == 10: rows.append(fields)
        document = meta.get('newdoc id', meta.get('newdoc_id', meta.get('newdoc', document)))
        if rows and meta.get('text'):
            yield dict(text=meta['text'], sentence_id=meta.get('sent_id'),
                       document=document, metadata=meta, rows=rows)


def source_records(path, source):
    import re
    candidates = [r for r in records(path) if eligible(r, source)]
    if source.get('pilot_max_sentences'):
        candidates = sorted(candidates, key=lambda r: hashlib.sha256(
            str(r['sentence_id']).encode()).digest())[:source['pilot_max_sentences']]
    for record in candidates:
        if source.get('text_normalization') == 'punctuation_detokenization':
            # Some treebanks distribute token-spaced text rather than raw prose.
            # Only whitespace is removed; the original bytes stay pinned upstream.
            record['text'] = re.sub(r' +([,.;:!?])', r'\1', record['text'])
            record['text'] = re.sub(r'([([]) +', r'\1', record['text'])
            record['text'] = re.sub(r' +([)\]])', r'\1', record['text'])
        if source.get('document_id_pattern') and not record['document']:
            match = re.match(source['document_id_pattern'], record['sentence_id'] or '')
            if match: record['document'] = match.group(1)
        yield record


def eligible(record, source):
    import re
    if source.get('sentence_id_pattern') and not re.match(source['sentence_id_pattern'], record['sentence_id'] or ''):
        return False
    return not any('-' in r[0] or 'Typo=Yes' in r[5] or 'Foreign=Yes' in r[5]
                   or 'CorrectForm=' in r[9] for r in record['rows'])


def snapshots(config_path, cache_dir=None, offline=False):
    result = []
    for source in json.loads(Path(config_path).read_text())['sources']:
        path = (Path(config_path).parent / source['path']).resolve()
        if sha256_file(path) != source['sha256']: raise ValueError('CoNLL-U checksum mismatch')
        result.append((source, {'data':path}, dict(source, path=str(path))))
    return result


def documents(snapshot_list):
    for source, files, receipt in snapshot_list:
        groups = {}
        for record in source_records(files['data'], source):
            if not eligible(record, source): continue
            # Unknown document boundaries: keep the entire file together.
            key = record['document'] or 'unknown-document-boundaries'
            groups.setdefault(key, []).append(record)
        for key, group in groups.items():
            identifier = source['name'] + ':' + key
            text = '\n'.join(r['text'] for r in group)
            yield dict(document_id=hashlib.sha256(identifier.encode()).hexdigest(),
                       text=text, source_name=source['name'], source_dataset=source['name'],
                       document_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       language=source['language'], url=source['url'], license=source['license'],
                       source_revision=source['revision'], source_document_id=key,
                       document_boundary_status='unavailable_file_grouped' if key=='unknown-document-boundaries' else 'source_annotated',
                       source_sentence_count=len(group),
                       source_sentence_ids_sha256=hashlib.sha256(json.dumps([r['sentence_id'] for r in group]).encode()).hexdigest(),
                       source_text_normalization=source.get('text_normalization','none'))


class AnnotatedParser:
    """Exact-offset adapter to the same spaCy interface used by live parsers."""
    def __init__(self, profile):
        import spacy
        from .profiles import resource
        self.vocab = spacy.blank('xx').vocab
        self.meta = {'version':'conllu-v2-pinned'}
        self.rejections = {}
        self.by_text = {}
        for source, files, _ in snapshots(resource(profile, 'sources_config')):
            if source['language'] != profile['language']: raise ValueError('Source language mismatch')
            for record in source_records(files['data'], source):
                if eligible(record, source):
                    self.by_text.setdefault(record['text'], []).append(record)

    def __call__(self, text):
        from spacy.tokens import Doc
        from .parsing import ParsedDocument
        records_ = self.by_text.get(text, [])
        if not records_: return None
        # Ambiguous duplicate annotation is excluded, not arbitrarily selected.
        if any(r['rows'] != records_[0]['rows'] for r in records_[1:]): return None
        rows = [r for r in records_[0]['rows'] if r[0].isdigit()]
        words, spaces, heads, deps, pos, lemmas, morphs = [], [], [], [], [], [], []
        cursor = 0
        for r in rows:
            if not text.startswith(r[1], cursor):
                self.rejections['unalignable_annotation'] = self.rejections.get('unalignable_annotation',0)+1
                return None
            words.append(r[1]); cursor += len(r[1])
            space = cursor < len(text) and text[cursor] == ' '
            spaces.append(space); cursor += int(space)
            heads.append(int(r[6])-1 if int(r[6]) else len(words)-1)
            deps.append('ROOT' if r[6]=='0' else r[7]); pos.append(r[3]); lemmas.append(r[2])
            morphs.append('' if r[5]=='_' else r[5])
        if cursor != len(text): return None
        if any(h < 0 or h >= len(words) for h in heads): return None
        doc = Doc(self.vocab, words=words, spaces=spaces, heads=heads, deps=deps,
                  pos=pos, lemmas=lemmas, morphs=morphs)
        if doc.text != text: raise ValueError('Annotation offset mismatch')
        return ParsedDocument(doc)

    def pipe(self, inputs, as_tuples=False, batch_size=64):
        for item in inputs:
            text, context = item if as_tuples else (item, None)
            doc = self(text)
            if doc is not None: yield (doc, context) if as_tuples else doc
