"""Create final-selection exclusions from recorded review and source boilerplate.

This is source curation, not additional corruption. It never edits source text or
labels reviewed examples as representative human validation. Run after screening
and before the final checkpoint-resumed selection.
"""
import hashlib
import json
from pathlib import Path
import re


PATTERNS = {
    'coverage_navigation': re.compile(r'\b(?:this|the) (?:post|story|article) is part of (?:our |a |the )?(?:special coverage|civic media observatory|coverage|series)', re.I),
    'republishing_notice': re.compile(r'\b(?:version|article|story|post)\b.{0,100}\b(?:published|republished|reprinted)\b.{0,120}\b(?:partnership|agreement|permission)\b', re.I),
}


def collect(checkpoints, review_paths):
    excluded = {}
    flagged_documents = {}
    for path in review_paths:
        for row in json.loads(Path(path).read_text())['rows']:
            if row['source_judgment'] != 'accept':
                digest=hashlib.sha256(row['original'].encode()).hexdigest()
                excluded[digest]=dict(reason=row['notes'],review=str(path))
                if row['source_judgment'] in {'reject','uncertain'}:
                    flagged_documents[row['url']]=dict(reason='reviewed_document_quality',review=str(path))
    counts={key:0 for key in PATTERNS}
    counts['reviewed_document_quality']=0
    counts['embedded_line_separator']=0
    for path in sorted(Path(checkpoints).glob('*.screened.jsonl')):
        for line in path.open():
            pair=json.loads(line)
            if any(c in pair['original'] for c in '\u0085\u2028\u2029\v\f'):
                counts['embedded_line_separator']+=1
                digest=hashlib.sha256(pair['original'].encode()).hexdigest()
                excluded.setdefault(digest,dict(reason='embedded_line_separator',source_url=pair['url']))
                continue
            if pair['url'] in flagged_documents:
                counts['reviewed_document_quality']+=1
                digest=hashlib.sha256(pair['original'].encode()).hexdigest()
                excluded.setdefault(digest,dict(flagged_documents[pair['url']],source_url=pair['url']))
                continue
            if pair['source_name'] != 'globalvoices':
                continue
            for reason,pattern in PATTERNS.items():
                if pattern.search(pair['original']):
                    counts[reason]+=1
                    digest=hashlib.sha256(pair['original'].encode()).hexdigest()
                    excluded.setdefault(digest,dict(reason=reason,source_url=pair['url']))
                    break
    return excluded,counts


if __name__=='__main__':
    root=Path('wiki/artifacts')
    paths=[root/'english-scale-first-batch-review.json',root/'english-scale-random-batch-review.json']
    exclusions,counts=collect('la_output/english_scale_checkpoints',paths)
    Path('config/english_scale_review_exclusions.json').write_text(json.dumps(sorted(exclusions),indent=2)+'\n')
    report=dict(method='Recorded agent source flags, other sentences from their source documents, and explicit navigation/republishing patterns',
                exclusions=exclusions,pattern_matches_before_deduplication=counts,
                human_precision=None)
    (root/'english-scale-exclusion-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(excluded_original_hashes=len(exclusions),pattern_matches=counts)))
