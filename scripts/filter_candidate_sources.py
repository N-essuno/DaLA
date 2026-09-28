"""Exclude agent-flagged originals through the shared exporter; retain the parent.

This does not promote candidate data to human-validated status.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from dala.common_pile import sha256_file
from dala.pair_pipeline import export_dataset
from dala.validate_dataset import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('--exclusions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    validate(a.parent)
    manifest=json.loads((a.parent/'manifest.json').read_text())
    if manifest['quality_status']!='candidate_rule_and_dictionary_screened':raise ValueError('Only candidate outputs are supported')
    policy=json.loads(a.exclusions.read_text())
    if policy['language']!=manifest['language']:raise ValueError('Review language mismatch')
    excluded=set(policy['source_sha256'])
    original=[json.loads(line) for f in sorted(a.parent.glob('*/pairs.jsonl')) for line in f.open()]
    pairs=[row for row in original if hashlib.sha256(row['original'].encode()).hexdigest() not in excluded]
    counts=dict(Counter(e['corruption_type'] for row in pairs for e in row['edits']))
    for key in ['selected_by_type','selected_edits_by_type']:
        if key in manifest:manifest[key]=counts
    manifest['postprocessing']=dict(operation='exclude_agent_flagged_source_hashes',parent=str(a.parent.resolve()),
        parent_manifest_sha256=sha256_file(a.parent/'manifest.json'),exclusions=str(a.exclusions.resolve()),
        exclusions_sha256=sha256_file(a.exclusions),pairs_removed=len(original)-len(pairs),
        script_sha256=sha256_file(__file__),native_validation=False)
    book=json.loads((a.parent/'rules.json').read_text())
    documents=[json.loads(line) for line in (a.parent/'documents.jsonl').open()]
    export_dataset(pairs,a.output,manifest,book,documents,manifest['seed'])
    validate(a.output)
    print(manifest['language'],len(original),'->',len(pairs))


if __name__=='__main__':main()
