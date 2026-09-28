---
type: Runbook
title: Audited DaLA upload preparation and DFM12 integration
description: Passed-only European HF packages, existing-language inventory, and train-only DFM12 integration.
status: draft
generated: {by: codex, at: '2026-09-28T15:05:00+00:00'}
---
# Selection and scope

The user requested preparation of all audited DaLA-style datasets for HF upload
and integration into DFM12. This authorizes local package preparation and
integration; no upload is performed by this task. The operational selection is
**automated audit passes only**: all four pair criteria must be `yes`, the
stored decision must be `pass`, the job must be `done`, and no explicit release
exclusion may apply. Flagged, uncertain, failed and explicitly excluded pairs
are omitted. This does not assert native linguistic validation or prove that
model-flagged pairs are invalid.

`export-upload/audited-dala-upload-plan.json` inventories 20 language variants:
12 new European language packages and 14 existing DFM12 packages covering
English, Dutch, Polish, Swedish, Bokmål, Nynorsk, Faroese and Icelandic. The
previous packages have locally recorded verified publication receipts and are
reused without duplication. They use their original audit contracts, which
are not relabelled as the newer canonical-pair audit.

# New HF packages

`scripts/prepare_audited_european_hf.py` freezes all terminal pair decisions,
checks source hashes and matches IDs, split ordinals and exact source/corrupt
text before selection. The new packages reuse `hf_package_runtime.py` for chat
construction, deterministic sharding and independent mechanical validation.

Output: `export-upload/european-audited-20260928/`. Only individual
`dala-{language-code}-audited/` directories are uploadable. Parent selection
files, snapshot receipts and logs are local-only. Suggested repository IDs are
`schneiderkamplab/dala-{language-code}-audited`, with lowercase `pt-pt` in the
repository name and explicit `pt-PT`/European Portuguese in data and prompts.
Each package has acceptability and correction configurations, all three original
splits, clean controls, canonical pairs, source-document attribution, rules,
audit dispositions, licensing notes, a card and a standalone validator/recreator.
Only `messages` belongs in model input.

All twelve packages passed full mechanical validation and card parsing. Total:
**4,738,657 retained pairs**, or **18,954,628 chat rows across both tasks and all
splits**. Exact counts and split sizes are in the local package README and
`packages.json`. These views share pairs; they are not independent examples.
The source datasets and all failed/audit-flagged originals remain intact.

# DFM12

`/work/mimir/HRM-Text/dfm12/dala_audited_release.py` consumes the validated
packages, writes train-only additions, reuses existing conversation/screening
and tokenizer code, and protects all raw producer heldout pools, including
pairs omitted by the linguistic audit. It screens exact sentence/document
collisions against those pools and exact chat/text matches against the existing
European reference index. A checksummed read-only SQLite snapshot on local
storage avoids remote random-read contention. Coverage remains partial for
inherited data and excludes fuzzy/semantic decontamination.

Run root: `/work/mimir/HRM-Text/data/dfm12/dala-audited-european-20260928/`.
The final `integration.json` and registered
`data/dfm12/local-audited-dala-additions.json` are published only after screening,
training-only tokenization and row-count verification. Existing sampled DFM12
files remain immutable. The DFM12 build command now supports an explicit
`--local-dala-additions PATH --suffix NAME` for a later isolated corpus build;
existing uploaded source gates are unchanged. No live training is restarted.

# Validation and remaining limitations

The passed-only selection test checks uncertainty, failed jobs and explicit
exclusion precedence. Five DFM12 tests cover clean/corrupt views, heldout rejection,
training-only admission and existing merge/sampling behavior. Full HF validation
checks hashes, standalone artifacts, source provenance, pair IDs, document split
isolation, edit round trips and exact chat recreation. These are automatic checks,
not human linguistic judgments. DFM12 integration completion is recorded in its
final receipt, not inferred from package readiness.


# Completed DFM12 integration

All twelve languages completed screening and tokenization. Exact overlap checks
removed no additional pairs. Final supply: **3,798,535 train pairs**, **15,194,140
chat rows**, **1,265,448,830 tokens** in 24 components and 316 tokenized shards.
No rows were dropped by tokenization. The final integration and registry checksum
were independently rechecked. Receipt: `/work/mimir/HRM-Text/data/dfm12/
dala-audited-european-20260928/completion-summary.json`. Both wiki validators passed;
the HRM-Text bundle reports zero errors and warnings. All parent corruption
families survive the HF audit filter (per-language family counts 7–12).

The upload plan now links the completed DFM12 integration. HF publication and
rebuilding the sampled/live training corpus have not been performed. Local
train-only additions are ready for the documented subsequent isolated build.


# HF publication — 2026-09-28

After explicit user instruction to upload, all twelve new packages were
published publicly under `schneiderkamplab`. All 4,738,657 retained pairs have
the four-yes automated pair-audit decision; both task configurations and all
three splits are published. Verified the exact remote file inventory, sizes,
Git/LFS content hashes and downloaded manifest SHA256 at the recorded commit
for every repository. Publication receipts: `export-upload/european-audited-20260928/
upload-receipts.json`; clickable inventory: the same directory's README.md.
The local upload plan now records all twenty non-Danish language variants as
published (twelve new plus eight earlier). Earlier eight use their established
per-task-row acceptance contracts; they were not reclassified as pair audits.

HF rejected `language: pt-PT` in the Portuguese card. Corrected the card to
`language: pt` with `language_bcp47: pt-PT`; language identity, prompts, pairs and
training tokens remain European Portuguese and unchanged. Revalidated the
complete Portuguese package and updated its DFM12 manifest pins. Old/new
metadata and migration evidence are retained in the local
`portuguese-card-migration/` directory. Prepared-package `upload_performed: false`
fields describe the immutable pre-publication build; upload receipts are the
authoritative publication state. No local-only parent selection files or audit
database were uploaded.
