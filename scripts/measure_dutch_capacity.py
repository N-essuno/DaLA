"""Count paragraph capacity without claiming sentence-level eligibility.

Run from the repository root after preparing the pinned Dutch resources.
"""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from dala.profiles import load_profile, resource
from dala.language_packs.dutch import DutchPack
from dala.dynaword import snapshots, documents


def measure():
    profile = copy.deepcopy(load_profile('nl_validation'))
    profile['curation']['max_paragraphs_per_document'] = None
    adapter = DutchPack(profile)
    counts = defaultdict(Counter)
    for document in documents(snapshots(resource(profile, 'sources_config'), offline=True)):
        n = sum(1 for _ in adapter.paragraphs(document))
        c = counts[document['source_name']]
        c['documents'] += 1
        c['eligible_paragraphs_uncapped'] += n
        for cap in (12, 24, 48):
            c[f'eligible_paragraphs_cap{cap}'] += min(n, cap)
        c['documents_above12'] += n > 12
    result = dict(method='Exact paragraph eligibility under current source and paragraph filters; no sentence/checker eligibility claim. No corpus text rewritten.',
                  counts={k: dict(v) for k, v in counts.items()})
    output = Path('wiki/artifacts/dutch-source-expansion/paragraph-depth-capacity.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    measure()
