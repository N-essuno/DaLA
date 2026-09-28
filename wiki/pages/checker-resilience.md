---
type: Experiment
title: Fail-closed checker resilience and Ukrainian continuation
description: Retried local checks, auditable sentence rejection, outage guards and preserved checkpoint lineage.
status: draft
generated: {by: codex, at: '2026-09-27T05:26:40.369389+00:00'}
---

# Failure and shared implementation

Ukrainian stopped at 236,958 retained pairs on another LanguageTool 6.6
`DictionaryMatchFilter` string-offset error involving combining accents. The
failed source is preserved in `../artifacts/european-expansion/checker-resilience/failing-candidate.json`.
It was batch 13447 (zero-based). The user authorized robust handling and a
128-parser resume with corresponding checker capacity.

The shared `LanguageCheck` client now accepts an explicit `failure_policy`.
Production Ukrainian enables three attempts, rotating across local pool
endpoints. Transport errors, HTTP 5xx/408/429, malformed JSON/response structure,
and incomplete results are retried. Other HTTP 4xx and changed checker software
remain hard failures. A complete successful response is required before caching.
Historical profiles without the policy keep their previous error behavior.

After retry exhaustion, an uncached language-specific control sentence is checked
against the attempted endpoints. If a control succeeds, the original candidate
is rejected via `SentenceCheckFailure`; `MorphologyCheck` records
`source_checker_sentence_failure`. Its raw source text, SHA256, attempted
endpoints, errors and checker software identity are appended to an audit JSONL.
No source text is rewritten and a failed response is never converted into a
successful diagnostic. Existing explicit source exclusions remain unchanged.

If health probes fail, screening stops with `CheckerUnavailable`. A shared,
thread-safe circuit also stops on five consecutive failures or twenty failures
within the last thousand uncached checks. These guards prevent silently dropping
large amounts of data during an outage or systematic checker fault. Successes
and failures are measured under the client lock. Cache hits do not hide a stream
of failing new checks. Failure auditing is separate from successful-response
caching.

# Evidence and continuation

Focused tests cover transient retry, persistent sentence rejection, no failed
cache entry, incomplete results, outage and failure-budget stops, HTTP client
errors, and propagation of outage failures through the morphology screen.
A migration regression covers the new policy while preserving the existing
exact-source quarantine and original generator lineage.

A live two-server probe retried the exact production-crashing sentence three
times across both endpoints, recorded a rejection after a healthy control, and
then successfully checked a normal Ukrainian sentence. The cache contained
only that successful check. Evidence: `live-probe.json`, `probe-failures.jsonl`,
and `tests-final.log` under the artifact directory above. These are automatic
checks, not native linguistic validation.

The prior generation source is frozen at
`/work/mimir/DaLA-before-checker-resilience/dala/`. The policy migration permits
only the reviewed `language_check.py` and `morphology_check.py` code changes,
execution counts, destination checkpoint path and explicit checker failure
policy. Rules, lexical inputs, source bytes, selection, parser models and prior
source exclusions remain pinned. All copied batch checksums and previously
screened edits are revalidated; parent migrations, per-batch generation
signatures and recovery ordering are preserved. Old outputs, partial files
and checkpoints remain untouched.

The new profile is
`la_output/resources/european-expansion/checker_resilient_v1/uk/profile.json`,
with **128 parser workers, 16 checker JVMs, and 128 screening clients**.
It uses new `checker_resilient_v1_checkpoints/uk` checkpoints and a separate
`checker_resilient_v1/uk` output. Target and split caps remain 478,930 pairs.
The production failure log is
`../artifacts/european-expansion/checker-resilience/production-failures.jsonl`.
Further parser/checker scaling remains authorized and must preserve this policy.

Release auditing and upload are separate; this change does neither.


Implementation validation completed: **201 tests passed**, including the policy
migration regression. The migration retained **13,736 candidate batches and
248,382 screened rows**, preserving the 236,958 selected pairs. All screened
edits were revalidated; these row counts differ because final deduplication and
split selection happen after screening.

The new generation supervisor and Ukrainian-only capacity controller are
launched. Authoritative run: `../artifacts/european-expansion/checker_resilient_v1/launch.json`.
The active-campaign pointer now refers to this run and its controller receipt.
That receipt includes the shared controller status/history paths. Previous
controllers are exited; all other eleven language builds remain complete.
The new source policy and failure log are retained through any subsequent
execution-only parser/checker resize.

Live startup verification observed all **128 parser workers and 16 checker JVMs**
alive, with checkpoint replay advancing. The preserved selection remains
236,958 pairs until replay catches up. See `checker-resilience/startup-health.json`.

# Completed production run

Completion confirmed at 2026-09-27T10:30:20.772554+00:00: Ukrainian reached 478,930 pairs
(train 383,144; validation 47,893; test 47,893). Export passed recorded
mechanical validation: exact edits and correction round trips, document split
isolation, unique exported text and at most one grammar edit. The resilient
checker recorded one automatic sentence rejection; no additional checker failure
was recorded. Parser and checker processes have exited. All twelve European
candidate builds are now locally complete; final release audits and uploads
remain separate and pending.
