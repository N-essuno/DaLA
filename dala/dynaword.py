"""Pinned DynaWord Parquet documents joined to their quality annotations."""
import hashlib
import json
import re
import random
from pathlib import Path

from .common_pile import sha256_file


def snapshots(config_path, cache_dir='la_output/cache', offline=False):
    from huggingface_hub import hf_hub_download
    result = []
    for source in json.loads(Path(config_path).read_text())['sources']:
        revision = source['revision']
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('DynaWord requires an immutable revision')
        files = {}
        receipts = []
        for kind in ('data', 'metadata'):
            name = source[f'{kind}_file']
            if name is None and kind == 'metadata':
                if source.get('quality_policy') != 'source_and_sentence_screening' or source.get('annotation_filters'):
                    raise ValueError('Missing annotations require explicit source/sentence screening policy')
                continue
            path = Path(hf_hub_download(source['repo_id'], name, repo_type='dataset', revision=revision,
                                       cache_dir=str(Path(cache_dir) / 'dynaword'), local_files_only=offline))
            digest = sha256_file(path)
            if source.get(f'{kind}_sha256') and source[f'{kind}_sha256'] != digest:
                raise ValueError(f'DynaWord checksum mismatch: {name}')
            files[kind] = path
            receipts.append(dict(repo_id=source['repo_id'], revision=revision, file=name, sha256=digest))
        result.append((source, files, dict(repo_id=source['repo_id'], revision=revision,
                                         file=source['data_file'], sha256=receipts[0]['sha256'], annotations=receipts[1] if len(receipts) > 1 else None)))
    return result


def documents(snapshot_list):
    import pyarrow.parquet as pq
    seen = set()
    for source, files, receipt in snapshot_list:
        annotations = pq.read_table(files['metadata']).to_pylist() if 'metadata' in files else []
        meta = {r['id']: r for r in annotations}
        if len(meta) != len(annotations):
            raise ValueError('Duplicate DynaWord annotation IDs')
        parquet=pq.ParquetFile(files['data'])
        sampled=None
        if source.get('pilot_sample_rows'):
            sampled=set(random.Random(source.get('pilot_sample_seed',4242)).sample(
                range(parquet.metadata.num_rows),min(source['pilot_sample_rows'],parquet.metadata.num_rows)))
        row_offset=0
        for batch in parquet.iter_batches():
            count=batch.num_rows
            if sampled is not None:
                indices=[i for i in range(count) if row_offset+i in sampled]
                row_offset+=count
                if not indices:continue
                batch=batch.take(indices)
            for row in batch.to_pylist():
                annotation = meta.get(row['id']) if 'metadata' in files else {}
                if ('metadata' in files and not annotation) or row['source'] != source['source_value']:
                    continue
                if any(annotation.get(key) not in allowed for key, allowed in source['annotation_filters'].items()):
                    continue
                if any(set(annotation.get(key, []) if isinstance(annotation.get(key), list) else [annotation.get(key)]) & set(blocked)
                       for key, blocked in source.get('annotation_exclusions', {}).items()):
                    continue
                if any(re.search(pattern, str(annotation.get(key, '')), re.I)
                       for key, pattern in source.get('annotation_text_exclusions', {}).items()):
                    continue
                text = row.get('text')
                if not isinstance(text, str) or len(text) < source.get('min_document_chars', 0):
                    continue
                year = None
                if source.get('document_year_regex'):
                    found = re.search(source['document_year_regex'], text[:2000], re.I)
                    if not found or int(found.group(1)) < source['min_document_year']:
                        continue
                    year = int(found.group(1))
                digest = hashlib.sha256(text.encode()).hexdigest()
                # Identical text groups together even when repeated under another ID.
                doc_id = hashlib.sha256(('dynaword\0' + digest).encode()).hexdigest()
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield dict(document_id=doc_id, text=text, document_sha256=digest,
                           source_name=source['name'], source_dataset=source['repo_id'], source_revision=source['revision'],
                           source_file=source['data_file'], document_year_from_text=year, upstream_id=row['id'], source_line=None,
                           url=f"https://huggingface.co/datasets/{source['repo_id']}/blob/{source['revision']}/{source['data_file']}",
                           url_kind='upstream_dataset_file_not_original_article', author=row.get('author') or None,
                           license=row.get('license') or source['license'], license_evidence_url=source['license_evidence_url'],
                           source_quality_annotation=annotation, source_added=row.get('added'), source_created=row.get('created'))
