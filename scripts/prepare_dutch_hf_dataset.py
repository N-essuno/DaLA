"""Package the audited Dutch release using the shared standalone chat exporter."""
import itertools
import json
from pathlib import Path
import platform
import shutil
import zlib
import yaml
from huggingface_hub import DatasetCard
import hf_package_runtime as runtime

REPO_ID = 'schneiderkamplab/dala-dutch-dynaword'
SOURCE = Path('la_output/dutch_dynaword_release')
ROOT = Path('export-upload/dala-dutch-dynaword')
AUDIT = Path('wiki/artifacts/dutch-extended-assessment')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def card(generation, summary, review):
    front = dict(language=['nl'], license='other', license_name='source-specific-cc0-and-cc-by-4.0',
        license_link='LICENSE.md', pretty_name='DaLA Dutch — DynaWord',
        task_categories=['text-classification','text-generation'],
        tags=['grammar','spelling','grammatical-error-correction','acceptability','synthetic-errors'],
        size_categories=['1M<n<10M'], configs=[dict(config_name=t,data_files=[dict(split=s,path=f'data/{t}/{s}-*.jsonl.gz') for s in runtime.SPLITS]) for t in runtime.TASKS])
    splits='\n'.join(f"| {s} | {v['pairs']:,} | {2*v['pairs']:,} |" for s,v in generation['splits'].items())
    sources='\n'.join(f"| {s} | {n:,} | {'CC BY 4.0' if s=='eurlex' else 'CC0 1.0'} |" for s,n in summary['sources'].items())
    families='\n'.join(f'| {f} | {n:,} |' for f,n in sorted(summary['families'].items(),key=lambda x:-x[1]))
    return '---\n'+yaml.safe_dump(front,sort_keys=False,allow_unicode=True)+f'''---
# DaLA Dutch — DynaWord

Dutch grammatical acceptability and error correction with synthetic spelling and
grammar errors. **Provisional, checker-screened training data; not a human-validated
gold benchmark.** No simplification, paraphrasing or style-transfer task.

## Configurations

**{summary['pairs']:,} original/corrupted pairs**, **{summary['rows_per_task']:,} chat rows
per configuration**. Every pair contributes a clean control and a corrupted input.
The two configurations share sentences and document splits and are not independent data.

| Split | Pairs | Rows per configuration |
| --- | ---: | ---: |
{splits}

- `acceptability`: answer `yes` for original controls and `no` for corrupted inputs.
- `correction`: return the source sentence, including identity correction for clean controls.

```python
from datasets import load_dataset
acceptability = load_dataset('{REPO_ID}', 'acceptability')
correction = load_dataset('{REPO_ID}', 'correction')
```

Each row has `messages` and `metadata`. Use **only messages as model input**:
metadata contains labels, edit families and audit judgments. Metadata also identifies
the canonical pair, document, split, source and source license. Join pair IDs to
`provenance/*.pairs.jsonl.gz` for exact source/edit spans and checker diagnostics.

## Sources and construction

Selected Dutch DynaWord sources, pinned to revision
`d0158defd949699532e59dea5978c5542afb0400` of
[danish-foundation-models/dutch-dynaword](https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/tree/d0158defd949699532e59dea5978c5542afb0400).

| Source | Pairs | Source-card license |
| --- | ---: | --- |
{sources}

The government subset uses excellent annotations and excludes conversational material.
EUR-Lex selects modern dated documents. The extension uses excellent Rechtspraak
judgments and official announcements with explicit modern publication-year headers.
Annotations are screening signals, not proof of grammatical correctness. Original
article URLs and authors are unavailable: provenance preserves dataset-file URLs,
upstream record IDs, hashes, source revisions and exact offsets; no original URL or
author is invented. Anonymization/substituted names and legal extraction artifacts
can remain. Some judgments describe crime or violence.

One injected edit per pair. Guarded morphology rules, curated spelling mappings and
character-level spelling operators generate candidates. Local LanguageTool screening
requires no relevant source diagnostics and a diagnostic at the injected edit;
spelling additionally uses the pinned OpenTaal wordlist. No family-percentage caps.
Documents receive seeded splits; global exact and heuristic near-duplicate filters
reduce overlap without proving absence of semantic leakage. Parent build manifests,
source configurations, exclusions and the active profile are included under metadata.

## Quality audit and release exclusions

Before release, a fresh uniform **200-pair sample** was drawn from 478,080 previously
unreviewed pairs in the frozen 478,930-pair parent (850 earlier-reviewed originals
excluded). Agent inspection found **187 acceptable originals, 10 erroneous and
3 uncertain**: **93.5% strict source acceptance**. All **200 injected edits** were
judged valid. A separate family-coverage supplement had **24 acceptable originals
and one uncertain**, with all 25 injected edits valid. Do not pool the purposive
supplement into the uniform estimate. These are agent judgments, not independent
native-speaker annotation, and do not establish perfect corruption precision.

The release removes **14 flagged original/corrupted pairs** (13 uniform-sample flags
and one supplementary flag), preserving all remaining records unchanged. The
pre-removal audit is retained in `metadata/quality-sample.json` and related files.
**Removal does not establish a new independently measured precision rate.** Other
source-label errors almost certainly remain: original controls and correction
targets are source excerpts, not guaranteed error-free sentences. Observed defects
include duplicated words, merged headings, broken compounds and name substitutions.

Earlier pilot audits informed source filters, exclusions and the 10:1 document
priority for Rechtspraak versus official announcements in the extension. Earlier
results must not be presented as independent precision measurements for this release.

## Coverage and limitations

| Family | Pairs |
| --- | ---: |
{families}

The release realizes {summary['active_rules']} rules and {summary['distinct_substitutions']:,}
distinct case-folded surface substitutions. This is not a count of independently
observed learner-error patterns. Article changes and spelling dominate; d/dt and
relative-pronoun examples remain sparse. Professional/legal prose dominates, and
synthetic error frequencies do not represent natural learner errors.

## Validation and recreation

All exported task rows, labels, edit reconstruction, document splits and artifact
checksums are validated. Before exclusion, all 478,930 originals were independently
recovered from pinned source offsets and all 241,701 spelling edits passed OpenTaal
checks. The release is a verified unchanged subset. Packaging checks every chat
instruction, input, answer and pair ID against the instruction exports.

```sh
python recreate_dataset.py --help
python recreate_dataset.py --root .
```

The bundled standard-library script validates or recreates task shards from bundled
canonical pairs. It does not recreate upstream source selection or certify linguistic
quality. Package checksums are in `metadata/manifest.json` and mechanical results in
`metadata/validation.json`. Preserve per-row attribution and deliberate-change notices
when reusing data; see [LICENSE.md](LICENSE.md).
'''


def main():
    source,root=SOURCE,ROOT
    if root.exists():raise FileExistsError(root)
    generation=json.loads((source/'manifest.json').read_text())
    for name,record in generation['artifacts'].items():
        p=source/name
        if runtime.digest(p)!=record['sha256'] or p.stat().st_size!=record['bytes']:raise ValueError(f'Changed artifact: {name}')
    summary=json.loads((AUDIT/'release-summary.json').read_text());review=json.loads((AUDIT/'sample.json').read_text())
    root.mkdir(parents=True)
    for folder in ('metadata','provenance','data'):(root/folder).mkdir()
    judgments={r['pair_id']:r['source_judgment'] for name in ('sample.json','family-supplement.json') for r in json.loads((AUDIT/name).read_text())['rows']}
    write_json(root/'metadata/config.json',dict(tasks=list(runtime.TASKS),prompts=generation['prompts'],seed=generation['seed'],shard_rows=100000,sample_source_judgments=judgments))
    for split in runtime.SPLITS:
        with (source/split/'pairs.jsonl').open('rb') as reader,runtime.compressed_writer(root/'provenance'/f'{split}.pairs.jsonl.gz') as writer:shutil.copyfileobj(reader,writer)
    with (source/'documents.jsonl').open('rb') as reader,runtime.compressed_writer(root/'provenance/documents.jsonl.gz') as writer:shutil.copyfileobj(reader,writer)
    copies=[(source/'manifest.json','generation-manifest.json'),(source/'rules.json','rules.json'),
        (Path('config/languages/nl_legal_extended.json'),'generation-profile.json'),(Path('config/dutch_sources_extended.json'),'source-config.json'),
        (Path('config/dutch_extension_production_source_exclusions.json'),'prior-source-exclusions.json'),
        (AUDIT/'sample.json','quality-sample.json'),(AUDIT/'family-supplement.json','quality-family-supplement.json'),
        (AUDIT/'review-summary.json','quality-review-summary.json'),(AUDIT/'summary.json','parent-assessment.json'),
        (AUDIT/'release-summary.json','release-summary.json'),(AUDIT/'release-exclusions.json','release-exclusions.json')]
    for i,parent in enumerate(generation['extension']['parents']):copies.append((Path(parent['path'])/'manifest.json',f'parent-{i}-manifest.json'))
    for src,dest in copies:shutil.copyfile(src,root/'metadata'/dest)
    shutil.copyfile('scripts/hf_package_runtime.py',root/'recreate_dataset.py')
    (root/'README.md').write_text(card(generation,summary,review))
    (root/'LICENSE.md').write_text('''# Source-specific licenses and changes

This release combines CC0 1.0 and CC BY 4.0 source material. It does not replace
source-specific terms with a single blanket license.

- Government web prose, Rechtspraak and Officiële bekendmakingen: CC0 1.0 according
  to their pinned DynaWord source cards: https://creativecommons.org/publicdomain/zero/1.0/
- EUR-Lex: CC BY 4.0 according to its pinned DynaWord source card:
  https://creativecommons.org/licenses/by/4.0/

Per-row metadata and canonical document records retain the source name, license,
license-evidence URL and pinned dataset-file coordinates. Original article URLs
and author names are unavailable and are not invented. Preserve this attribution
and the source-specific license links on redistribution.

Changes: DaLA excerpts sentences, selects and splits records, introduces the
explicitly recorded grammar/spelling errors, and adds task instructions and labels.
Corrupted text is deliberately modified, not original author wording. Clean controls
and correction targets reproduce source excerpts and may themselves contain errors.
No source publisher endorsement is implied.

Rule provenance and notices for referenced lexical resources are retained in the
rulebook and generation profile. OpenTaal and LanguageTool dictionaries are not
redistributed as standalone lexical datasets in this package.
''')
    DatasetCard.load(root/'README.md')
    shards=[]
    for task in runtime.TASKS:shards.extend(runtime.build_data(root,root/'data'/task,task))
    compared={}
    for task in runtime.TASKS:
        count=0
        for split in runtime.SPLITS:
            actual=itertools.chain.from_iterable(runtime.read_gzip(root/s['path']) for s in shards if s['task']==task and s['split']==split)
            with (source/split/f'{task}_it.jsonl').open() as reader:
                for chat,line in itertools.zip_longest(actual,reader):
                    if chat is None or line is None:raise ValueError('Task row count mismatch')
                    old=json.loads(line);expected=[dict(role='user',content=old['direction']+'\n\n'+old['samples']['content']),dict(role='assistant',content=old['samples']['response'])]
                    if chat['messages']!=expected or chat['metadata']['pair_id']!=old['pair_id']:raise ValueError('Task content/order changed')
                    count+=1
        compared[task]=count
    write_json(root/'metadata/original-export-equivalence.json',dict(status='passed',rows=compared))
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=runtime.digest(p)) for p in sorted(root.rglob('*')) if p.is_file()}
    write_json(root/'metadata/manifest.json',dict(schema_version=1,repo_id=REPO_ID,quality_status='checker_screened_review_exclusions_applied',source_manifest_sha256=runtime.digest(source/'manifest.json'),runtime=dict(python=platform.python_version(),zlib=zlib.ZLIB_RUNTIME_VERSION),splits={k:dict(pairs=v['pairs'],rows_per_task=v['acceptability_rows']) for k,v in generation['splits'].items()},shards=shards,files=files))
    write_json(root/'metadata/validation.json',runtime.validate(root))
    print(f'Package ready: {root}',flush=True)


if __name__=='__main__':main()
