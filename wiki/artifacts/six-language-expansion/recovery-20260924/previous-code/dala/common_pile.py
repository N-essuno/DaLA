"""Pinned Common Pile JSONL snapshots with document-level provenance."""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_SOURCES = Path(__file__).resolve().parents[1] / 'config/english_sources.json'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshots(config_path=DEFAULT_SOURCES, cache_dir='la_output/cache', offline=False):
    """Download exact revisions once; cached content is checked against its receipt."""
    config = json.loads(Path(config_path).read_text())
    result = []
    for source in config['sources']:
        if len(source['revision']) != 40 or any(c not in '0123456789abcdef' for c in source['revision']):
            raise ValueError('Source revisions must be full immutable commit hashes')
        for filename in source['files']:
            destination = Path(cache_dir) / source['name'] / source['revision'] / Path(filename).name
            receipt = destination.with_suffix(destination.suffix + '.sha256')
            if not destination.exists():
                if offline:
                    raise FileNotFoundError(f'Missing cached snapshot: {destination}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{filename}"
                temporary = destination.with_suffix('.part')
                try:
                    with requests.get(url, stream=True, timeout=(15, 120)) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                handle.write(chunk)
                    temporary.replace(destination)
                    receipt.write_text(sha256_file(destination) + '\n')
                finally:
                    temporary.unlink(missing_ok=True)
            digest = sha256_file(destination)
            if not receipt.exists() or receipt.read_text().strip() != digest:
                raise ValueError(f'Cache checksum mismatch or missing receipt: {destination}')
            result.append((source, destination, dict(repo_id=source['repo_id'], revision=source['revision'],
                                                   file=filename, sha256=digest)))
    return result


def documents(snapshot_list):
    """Yield source text unchanged, with stable URL-based grouping IDs."""
    seen = set()
    for source, path, receipt in snapshot_list:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                metadata = row.get('metadata') or {}
                url = metadata.get('url', '')
                host = (urlparse(url).hostname or '').removeprefix('www.')
                if host not in source['domains'] or row.get('source') not in source['source_values']:
                    continue
                license_ = metadata.get('license', '')
                if source['license_contains'] not in license_ or not isinstance(row.get('text'), str):
                    continue
                if len(row['text']) < source.get('min_document_chars', 0):
                    continue
                canonical_url = urlparse(url)._replace(query='', fragment='').geturl().rstrip('/')
                doc_id = hashlib.sha256(canonical_url.encode()).hexdigest()
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                attribution = dict(author=metadata.get('author'))
                if source.get('license_override'):
                    attribution.update(captured_license=license_, license_evidence_url=source['license_evidence_url'])
                    license_ = source['license_override']
                yield dict(document_id=doc_id, source_name=source['name'], source_dataset=source['repo_id'],
                           source_revision=source['revision'], source_file=receipt['file'],
                           source_line=line_number, upstream_id=str(row.get('id', '')), url=url,
                           license=license_, text=row['text'], document_sha256=hashlib.sha256(row['text'].encode()).hexdigest(), **attribution)
