---
type: Experiment
title: European campaign upgrade and checkpoint continuation
description: Shared early curation, parser prefetch, checker pooling, explicit generation lineage and recovery of previously skipped source sentences.
status: draft
generated: {by: codex, at: '2026-09-26T13:26:13.687017+00:00'}
---
# Implementation

The user authorized the remaining improvements, then stopping and resuming the
main campaign. The tested contraction adapter is now installed in the main
repository. Language-specific rulebooks, lexical resources, parser models and
source bytes are unchanged. The additions are shared code and opt-in inputs:

* A text-only source gate runs before expensive POS/lemma/dependency inference.
  It uses the same necessary text conditions as MorphologyPack; syntactic source
  checks still run afterward. Rejected sentences keep their source spans and
  token positions through placeholders; original text is not rewritten.
* The parser queue has four batches per worker, processed in the same deterministic
  order. This reduces idle workers while downstream batches are screened.
* French uses two existing local checker-server instances and sixteen screening
  clients. Other independent checkers retain eight clients. Hard grammar and
  spelling checks remain enabled with unchanged diagnostic policy.
* Batch receipts and progress now include tokenizer, syntax, candidate-preparation
  and screening timings, selected corruption-family counts, and dictionary-unknown
  examples. The latter aggregate the top twenty words per batch, not an exhaustive
  lexicon-frequency census.

The early-filter paired check passed in all twelve languages: eligible source
sets and grammar-candidate sets were equal. Token indices/spelling selection may
change under the source-span adapter, so this is not a claim of byte-identical
new-generation output. Aggregate syntax time in these small samples decreased
by 26.9%; full-corpus throughput must be measured separately. All **178 unit
tests** passed after installation.

# Preserved work and explicit lineage

The old supervisor/process group was stopped after implementation and tests.
The stop receipt records **1,797,691 retained pairs**, including completed Finnish
and the exhausted first Portuguese source pass. Earlier outputs, source profiles,
checkpoint files and checker caches remain available. The old generation source
is preserved at `/work/mimir/DaLA-before-campaign-upgrade/`.

Only checksum-verified candidate/screened batches were hard-linked into fresh
`european_scale_v2_checkpoints` directories. Uncommitted partial files remain in
the old directory and are regenerated. SQLite caches were backed up after all
old writers stopped. The migration rejects changes to language, rulebooks,
models, source configuration, batch boundaries and seed/order inputs; the
reviewed generator changes are explicit rather than represented as equivalence.

Every exported pair in the resumed run receives the generation signature and
batch number that produced its candidate. The final manifest embeds the legacy
signature, reused-batch inventory and migration receipt. Resuming first rebuilds
the deterministic deduplication/selection state from verified batches. Status
reports label counts from preserved checkpoints while that replay is in progress.
Known agent source exclusions apply during final selection, including reused rows.

When a language exhausts its base source without reaching its split quotas, it
reprocesses the legacy batches with the upgraded generator. Original-sentence
deduplication preserves earlier retained pairs and admits newly recoverable
sentences. These recovery batches have separate IDs/checkpoints and provenance.
This is essential for Portuguese: its v1 pass exhausted the source at 104,712
pairs, so simply continuing beyond the old cursor would recover nothing.

The end-to-end migration fixture preserved all 46 legacy pairs unchanged and
added 26 pairs, with two distinct generation signatures. Independent checks
passed for all 72 resulting pairs: reconstruction, provenance, split isolation,
artifact checksums and task views.

# Resumed campaign

Active run: **european_scale_v2**. Eleven languages resume with **192 CPU parser
workers**, one thread per parser. Allocation: ca 16, cs 16, de 20, el 20, es 16,
et 16, fr 12, it 20, ro 16, uk 20, pt-PT 20. French receives fewer parser workers
because the measured bottleneck was screening. Finnish remains complete at
478,930 pairs using its preserved v1 output.

Caps remain 478,930 pairs per language, with train/validation/test quotas
383,144 / 47,893 / 47,893. Source exhaustion may still yield a shortfall. There
is no new native linguistic validation or upload authorization implied by this
compute migration.

Active pointer: `wiki/artifacts/european-expansion/active-campaign.json`.
Launch receipt: `wiki/artifacts/european-expansion/european_scale_v2/launch.json`.
Migration/test/stop receipts: `wiki/artifacts/european-expansion/campaign-upgrade/`.
Profiles: `la_output/resources/european-expansion/european_scale_v2/`.
Completed outputs: `la_output/european_pilots/european_scale_v2/CODE/`, except fi
at its existing v1 path. Earlier reviewed pilot pointers remain unchanged.

```sh
python -m scripts.report_european_scale_status wiki/artifacts/european-expansion/european_scale_v2/launch.json
```

Post-launch health check confirmed all 192 parser processes alive, all eleven
resumed languages past checkpoint replay and generating, and Portuguese in its
recovery pass (11 recovery batches committed at that snapshot). Total retained
pairs including completed Finnish: 1,820,152. No worker failures were
reported. See `campaign-upgrade/resume-health.json`.
