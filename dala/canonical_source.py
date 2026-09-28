"""Pinned, document-preserving JSONL input for shared corpus preparation tools."""
import hashlib
import json
from pathlib import Path

from .common_pile import sha256_file


def snapshots(config_path, cache_dir=None, offline=False):
    result = []
    config_path = Path(config_path)
    for source in json.loads(config_path.read_text())['sources']:
        path = (config_path.parent / source['path']).resolve()
        if sha256_file(path) != source['sha256']:
            raise ValueError('Canonical source checksum mismatch')
        result.append((source, {'data': path}, dict(source, path=str(path))))
    return result


def documents(snapshot_list):
    seen = set()
    for source, files, _ in snapshot_list:
        for line in files['data'].open():
            row = json.loads(line)
            for key in ('language', 'source_document_id', 'text', 'url'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f'Missing canonical document field: {key}')
            if row['language'] != source['language']:
                raise ValueError('Canonical source language mismatch')
            # Revision is deliberately absent: the same article must retain its
            # split when a later snapshot is used.
            identifier = source['dataset'] + '\0' + row['source_document_id']
            if identifier in seen:
                raise ValueError('Duplicate canonical source document')
            seen.add(identifier)
            text = row['text']
            yield dict(row, document_id=hashlib.sha256(identifier.encode()).hexdigest(),
                       document_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       source_name=source['name'], source_dataset=source['dataset'],
                       source_revision=source['revision'], license=source['license'],
                       document_boundary_status='source_document')
