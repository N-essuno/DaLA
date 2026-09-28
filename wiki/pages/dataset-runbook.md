---
type: Reference
title: Dataset runbook
description: Reproduce local English dataset generation and review exports.
status: stable
generated:
  by: codex/gpt-6
  at: '2026-09-21T15:13:36+00:00'
---
# Setup

From the repository root, install `requirements.txt` into a virtual environment.
English additionally uses the spaCy `en_core_web_md` 3.8.0 model and the pinned
local LanguageTool 6.6/Temurin 17.0.16 runtime:

```bash
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
python -m dala.language_check --setup
```

The setup command downloads about 300 MB of Java/checker archives to ignored
`la_output/tools`. It targets Linux x86-64. Build commands start a loopback-only
checker server and stop it on completion. No text is sent to a cloud checker.
Sources download once into checksummed `la_output/cache`. Checker responses are
cached in SQLite, keyed by text, dialect and checker software metadata.

# Build

```bash
python -m dala.multilingual --profile config/languages/en.json --output-dir la_output/english_productive
# Equivalent integrated entry point:
python dala/create_dala.py --language en
# Bounded build with cached snapshots:
python -m dala.build_english --offline --max-documents 30 --output-dir la_output/smoke
```

Existing output directories are never mixed or overwritten; choose a fresh path.
`--max-errors` accepts 1, 2 or 3 (default 2). It is an upper bound; the seeded
selection uses fewer errors when no independent spelling edit is available.
Source sentences are preserved exactly. Outputs are staged, mechanically
validated and renamed into place only on success.

# Outputs

Each of `train`, `validation`, `test` contains `pairs.jsonl`, raw acceptability
JSONL/CSV, `acceptability_it.jsonl` and `correction_it.jsonl`. The latter two use
the TV2R instruction schema. The root contains a dataset card, source document
metadata, the evidence rulebook, checksummed build manifest and stratified
`review.csv`. A clean/corrupted pair supplies two rows per task, including
identity correction. Split sizes follow actual source/rule/checker coverage.

# Validation and review

```bash
python -m unittest discover -s tests -v
python -m dala.validate_dataset la_output/english_productive
python -m dala.dataset_review --dataset la_output/english_productive --all --output la_output/review_all.csv
python -m dala.dataset_review --dataset la_output/english_productive --decisions la_output/review_all.csv --output la_output/english_reviewed
python scripts/validate_wiki.py
```

Only explicit positive judgments for original correctness, corrupted incorrectness
and validity of the intended edits admit a pair to the reviewed export. Blank
judgments are not acceptance. A reviewer ID and unchanged reviewed texts are
required. A reviewer string does not itself prove native expertise or independence.

The normal build performs no publication. An explicit `--push-to-hub` on the
integrated CLI uploads task views privately. Source and corruption metadata must
accompany any public release. Human per-rule precision is not established by
passing mechanical checks or by a grammar checker's decisions.

# Evidence regeneration

The checked-in rulebook can be used without downloading learner text. To reproduce
its evidence counts, obtain W&I+LOCNESS v2.1 from the BEA source linked in
[error evidence](/pages/error-evidence.md), extract the A/B/C training M2 files,
and run the documented rulebook command. Training file hashes must match
`config/english_rules.json`. Rebuilding from these inputs was verified byte-identical.

# Danish compatibility and new language packs

Install `da_core_news_md` 3.8.0, then run:

```bash
python -m dala.multilingual --profile config/languages/da.json --output-dir la_output/danish
python -m scripts.check_danish_equivalence
```

The equivalence script caches an immutable DDT snapshot, reuses actual parser
annotations for both implementations, and checks every active targeted rule plus
complete pipelines in both split modes with two seeds. Its receipt is in
`wiki/artifacts/danish-equivalence.json`. Historical source URL behavior is
configurable in the pack; equal inputs and parser versions are prerequisites.

For a new language, supply a profile with `schema_version`, `language`, `mode`,
`parser`, `sources` and `selection`. Compatibility mode uses declarative operator
rules and UD source/filter/split settings. Pair mode supplies an adapter reference,
rulebook/source/exclusion resources, prompts and checker dialects. The adapter
implements `load_rulebook`, `paragraphs`, `sentence_rejection`, `candidates`, and
`select_edits`; candidates use the shared `Corruption` edit record. A productive
rulebook supplies its edit-validation callable. Resource paths are profile-relative.
The supported source adapters are currently UD POS (compatibility) and Common Pile
(pairs); additional source formats require a shared adapter, not a language fork.

Amount and diversity comparison:

```bash
python -m scripts.assess_dataset --baseline la_output/english_common_pile --current la_output/english_productive --output wiki/artifacts/english-comparison.json --review-output la_output/english_productive_agent_review.json
```

Run this sampling command before filling judgments; it regenerates an unreviewed
queue and must not overwrite completed annotations.

# Productive character and token revision

The active English pack includes character generators that require the local
checker; `--checker off` is no longer supported for this rulebook. The historical
corpus-only rulebook can still be used for diagnostics separately. Character
parameters and token guards live in `config/english_productive_rules.json`.

```bash
python -m scripts.evaluate_english_generators
python -m dala.multilingual --language en --offline --output-dir la_output/english_generators
python -m dala.validate_dataset la_output/english_generators
```

The probe command overwrites its automatic probe receipt; preserve completed
agent annotations first. See [generator evaluation](/pages/english-generators.md)
for guard decisions and validation limitations.

The completed generator build is `la_output/english_generators/`; its assessment
and quality limitations are recorded in the generator evaluation page.
