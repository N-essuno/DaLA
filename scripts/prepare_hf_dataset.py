"""Package the frozen English corpus as two Hugging Face chat configurations."""
import argparse
import itertools
import json
from pathlib import Path
import platform
import shutil
import zlib

import yaml
from huggingface_hub import DatasetCard
import hf_package_runtime as runtime

REPO_ID = 'schneiderkamplab/dala-english-common-pile'


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def card():
    front = dict(language=['en'], license='other', license_name='source-specific-creative-commons',
                 license_link='LICENSE.md', pretty_name='DaLA English — Common Pile',
                 task_categories=['text-classification', 'text-generation'],
                 tags=['grammar', 'spelling', 'grammatical-error-correction', 'acceptability', 'synthetic-errors'],
                 size_categories=['1M<n<10M'], configs=[dict(config_name=task, data_files=[
                     dict(split=split, path=f'data/{task}/{split}-*.jsonl.gz')
                     for split in runtime.SPLITS]) for task in runtime.TASKS])
    return '---\n' + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + '''---
# DaLA English — Common Pile

English grammatical acceptability and error correction from Common Pile source
sentences with synthetic grammar and spelling corruptions. This is a
**provisional, checker-screened corpus with known source-label errors**, not a
human-validated benchmark. No simplification, paraphrasing or style-transfer task
is included.

## Configurations and size

There are **478,930 distinct original/corrupted pairs**. Each configuration has
**957,860 chat rows**: one original control and one corrupted input per pair.
The two configurations share sentences and document splits; their combined
1,915,720 rows are not independent examples.

| Split | Pairs | Rows per configuration |
| --- | ---: | ---: |
| train | 383,144 | 766,288 |
| validation | 23,946 | 47,892 |
| test | 71,840 | 143,680 |

- `acceptability`: answer `yes` for the original or `no` for the corrupted input.
- `correction`: output the original sentence; clean controls map to themselves.

```python
from datasets import load_dataset

acceptability = load_dataset("schneiderkamplab/dala-english-common-pile", "acceptability")
correction = load_dataset("schneiderkamplab/dala-english-common-pile", "correction")
```

Each row has `messages` (user instruction plus input, assistant target) and
`metadata` (pair/document IDs, split, variant, source author/URL/license, error
count and families). Use **only `messages` as model input**: metadata contains
labels and audit judgments. Pair IDs join both task views to the exact edits in
`provenance/{split}.pairs.jsonl.gz`. `agent_source_judgment` is the latest
100-pair sample judgment or `not_reviewed`; it is not human certification.

For example, the first canonical training pair contains:

- Original: `They are constructing a canal to bring water from a river near the Turkish border to the lake.`
- Corrupted: `They is constructing a canal to bring water from a river near the Turkish border to the lake.`
- Attribution: Tori Egherman, [Global Voices](https://globalvoices.org/2016/05/07/leo-dicaprio-and-lake-urmia/), CC BY 3.0. The second sentence is deliberately modified.

## Construction and source provenance

Sources are selected English Common Pile subsets:

| Publisher | Pairs | Applicable source license |
| --- | ---: | --- |
| Global Voices | 440,818 | CC BY 3.0 |
| 360info | 29,606 | CC BY 4.0 |
| Public Domain Review essays | 8,506 | CC BY-SA 4.0 |

Upstream snapshots:

- [`common-pile/news`](https://huggingface.co/datasets/common-pile/news/tree/13e76bdc8d49ed14d710fac7e3b61186cf74c8d3), revision `13e76bdc8d49ed14d710fac7e3b61186cf74c8d3`.
- [`common-pile/public_domain_review`](https://huggingface.co/datasets/common-pile/public_domain_review/tree/e9c7669206b95871601fe4672a484d8215e6fe6f), revision `e9c7669206b95871601fe4672a484d8215e6fe6f`.

Sentences undergo parser-based guarded corruption and LanguageTool screening.
There are at most two edits per pair and at most one grammar edit. Spelling
combines sourced lexical mistakes with guarded character operations; grammar
includes agreement, modal/auxiliary forms, participles and number. Restricted
deletions and word swaps provide fallback candidates. These are synthetic
errors, not naturally occurring learner corrections. See `metadata/rules.json`
and the generation profile for rule provenance and screening configuration.
Documents receive seeded split assignments before selection. Exact and heuristic
near-duplicate filtering reduces overlap; it is not a proof of zero semantic
leakage or benchmark contamination.

Rows retain author metadata when supplied; missing authors remain null. Bundled
canonical pairs retain source revision, source file/line, document hash, sentence
offsets, original/corrupted edit spans and checker diagnostics. Global Voices'
publisher license is CC BY 3.0, while Common Pile's captured metadata says CC BY
4.0; both values are retained separately. The publisher license is used in the
chat metadata. See [LICENSE.md](LICENSE.md) for attribution and changes.

## Quality findings and limitations

A fresh uniform **100-pair sample from this exact final corpus** received agent
inspection using grammar/spelling criteria:

| Source judgment | Pairs |
| --- | ---: |
| Acceptable original | 80 |
| Clearly erroneous original | 11 |
| Uncertain original | 9 |

All 100 pairs' injected changes were judged error-inducing, but only 80 pairs
had both an accepted source and valid corruption. This is **agent inspection,
not independent human annotation**. It does not establish 100% corruption
precision. The approximate sampling-only 95% Wilson interval for strict usable
pairs is 71.1–86.7%; it does not include annotation bias.

Known source mistakes include `each other weirdness`, `looking his photos` and
`It was 1940s`. Such originals are still labeled correct and still serve as
correction targets. They have **not** been silently removed after this audit.
The full sample and judgments are in `metadata/english-quality-sample.json`.
Earlier pre-curation agent inspection of 130 pairs led to document/boilerplate
exclusions; those earlier results must not be mistaken for an independent
quality estimate of the released corpus.

A same-protocol Danish reference sample yielded 71/100 strict usable pairs,
with 14 clearly erroneous and three uncertain originals. This does not establish
English's overall superiority: the samples are small, not genre/family matched,
and reviewed by the same agent rather than native-speaker adjudicators.

English is heavily spelling-weighted: **506,887 of 691,545 edits (73.3%)** are
spelling-family edits. The corpus realizes 47,073 surface substitutions, but
`that → taht` alone accounts for 81,585 edits (11.8%). Surface variety is not
balanced grammatical coverage or a natural error-frequency estimate. News
sources dominate, particularly Global Voices; regional English, quotations,
extraction artifacts and existing punctuation errors complicate clean labels.
Use this release as provisional synthetic training material and assess it for
your application before treating its validation/test splits as gold benchmarks.

## Validation and reconstruction

`metadata/manifest.json` inventories package files with checksums.
`metadata/validation.json` records mechanical validation separately from
linguistic quality. The original generation manifest is retained as
`metadata/generation-manifest.json`; its historical paths describe the upstream
local build, not additional files in this package.

After downloading the complete repository:

```bash
python recreate_dataset.py
python recreate_dataset.py --recreate-data /path/to/new-data-directory
```

The standalone standard-library script verifies every row, edit round-trip,
document split, source metadata and artifact hash, and can recreate both chat
configurations from the bundled frozen pairs. It does **not** regenerate
corruptions from raw Common Pile. Exact compressed-byte recreation uses the
Python/zlib versions recorded in the package manifest. No custom code is needed
to load the two dataset configurations with `datasets`.
'''


LICENSE = '''# Source-specific licenses and attribution

This release is a collection with heterogeneous source licenses. It does not
relicense all source text under a single blanket license. Original copyrights
remain with their respective authors and publishers.

- **Global Voices**: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/),
  per its [republishing policy](https://globalvoices.org/about/global-voices-attribution-policy/).
  Common Pile's captured BY 4.0 string is retained as `captured_license` rather
  than silently substituted for the publisher's BY 3.0 terms.
- **360info**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  per its [content-use policy](https://360info.org/how-to-use-our-content/).
- **Public Domain Review essays selected here**:
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as recorded in
  the selected Common Pile records. The [publisher's reuse policy](https://publicdomainreview.org/reusing-material)
  notes that some essays have different terms; this release selects the BY-SA
  source records and does not claim every PDR essay is BY-SA or public domain.
  Adaptations of these selected texts retain CC BY-SA 4.0 terms.

**Attribution**: Each chat row includes the supplied author, publisher/source
name, original source URL and source license. Canonical pairs and document
records preserve this attribution plus the pinned upstream dataset revision,
source coordinates and document hash. Missing author metadata is left null,
not replaced with a guessed author; the original publisher page remains linked.
Preserve the supplied attribution, license links and change notices on reuse.

**Changes**: DaLA excerpts sentences from the Common Pile text representation,
selects and splits them, intentionally introduces the grammar/spelling edits
recorded in canonical pairs, and wraps the texts with task instructions and
synthetic labels. Original-control variants have no injected corruption.
Corrupted text is a DaLA modification, not an attributed author's original
wording. Correction targets restore the source sentence, which can itself
contain errors. No publisher or author endorsement is implied.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('la_output/english_common_pile_scaled'))
    parser.add_argument('--output', type=Path, default=Path('export-upload/dala-english-common-pile'))
    args = parser.parse_args()
    source, root = args.source, args.output
    if root.exists():
        raise FileExistsError(root)
    generation = json.loads((source / 'manifest.json').read_text())
    for name, record in generation['artifacts'].items():
        path = source / name
        if runtime.digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Frozen artifact changed: {name}')
    print('Frozen source artifact hashes verified', flush=True)
    root.mkdir(parents=True)
    for folder in ('metadata', 'provenance', 'data'):
        (root / folder).mkdir()
    review = json.loads(Path('wiki/artifacts/language-quality-comparison/en-sample.json').read_text())
    config = dict(tasks=list(runtime.TASKS), prompts=generation['prompts'], seed=generation['seed'], shard_rows=100000,
                  sample_source_judgments={r['pair_id']: r['source_judgment'] for r in review['rows']})
    write_json(root / 'metadata/config.json', config)
    for split in runtime.SPLITS:
        with (source / split / 'pairs.jsonl').open('rb') as reader, runtime.compressed_writer(root / 'provenance' / f'{split}.pairs.jsonl.gz') as writer:
            shutil.copyfileobj(reader, writer)
    with (source / 'documents.jsonl').open('rb') as reader, runtime.compressed_writer(root / 'provenance/documents.jsonl.gz') as writer:
        shutil.copyfileobj(reader, writer)
    for src, dest in [
        (source / 'manifest.json', 'metadata/generation-manifest.json'),
        (source / 'rules.json', 'metadata/rules.json'),
        (Path('config/languages/en_scale.json'), 'metadata/generation-profile.json'),
        (Path('config/english_sources_scale.json'), 'metadata/source-config.json'),
        (Path('wiki/artifacts/language-quality-comparison/en-sample.json'), 'metadata/english-quality-sample.json'),
        (Path('wiki/artifacts/language-quality-comparison/summary.json'), 'metadata/language-quality-comparison.json'),
        (Path('wiki/artifacts/language-quality-comparison/sampling.json'), 'metadata/quality-sampling.json'),
        (Path('scripts/hf_package_runtime.py'), 'recreate_dataset.py'),
    ]:
        shutil.copyfile(src, root / dest)
    (root / 'README.md').write_text(card())
    (root / 'LICENSE.md').write_text(LICENSE)
    DatasetCard.load(root / 'README.md')
    shards = []
    for task in runtime.TASKS:
        shards.extend(runtime.build_data(root, root / 'data' / task, task))
    # Compare all messages and row identities independently to existing task exports.
    compared = {}
    for task in runtime.TASKS:
        count = 0
        for split in runtime.SPLITS:
            paths = [root / s['path'] for s in shards if s['task'] == task and s['split'] == split]
            actual = itertools.chain.from_iterable(runtime.read_gzip(p) for p in paths)
            with (source / split / f'{task}_it.jsonl').open() as reader:
                for chat, line in itertools.zip_longest(actual, reader):
                    if chat is None or line is None:
                        raise ValueError('Task export row count mismatch')
                    old = json.loads(line)
                    expected = [dict(role='user', content=old['direction'] + '\n\n' + old['samples']['content']),
                                dict(role='assistant', content=old['samples']['response'])]
                    if chat['messages'] != expected or chat['metadata']['pair_id'] != old['pair_id']:
                        raise ValueError('Task export content/order changed')
                    count += 1
        compared[task] = count
        print(f'Original export equality: {task} {count:,} rows', flush=True)
    write_json(root / 'metadata/original-export-equivalence.json', dict(status='passed', rows=compared, preserved=['directions', 'inputs', 'responses', 'pair_ids', 'row_order']))
    files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=runtime.digest(p))
             for p in sorted(root.rglob('*')) if p.is_file()}
    manifest = dict(schema_version=1, repo_id=REPO_ID, quality_status='checker_screened_with_known_source_label_errors',
                    source_manifest_sha256=runtime.digest(source / 'manifest.json'),
                    runtime=dict(python=platform.python_version(), zlib=zlib.ZLIB_RUNTIME_VERSION),
                    splits={k: dict(pairs=v['pairs'], rows_per_task=v['acceptability_rows']) for k,v in generation['splits'].items()},
                    shards=shards, files=files)
    write_json(root / 'metadata/manifest.json', manifest)
    write_json(root / 'metadata/validation.json', runtime.validate(root))
    print(f'Package ready: {root}', flush=True)


if __name__ == '__main__':
    main()
