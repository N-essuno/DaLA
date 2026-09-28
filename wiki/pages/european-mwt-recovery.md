---
type: Experiment
title: European contraction recovery and rejection audit
description: Isolated shared-parser recovery, protected source spans, checker triage and production bottleneck evidence.
status: draft
generated: {by: codex, at: '2026-09-26T12:49:23.305165+00:00'}
---
# Scope and isolation

The user authorized implementation following the production-yield diagnosis.
The candidate implementation is in `/work/mimir/DaLA-mwt-recovery`, a snapshot
of the working code, not a clean checkout of HEAD (the working tree has substantial
uncommitted implementation). The running `european_scale_v1` jobs retain their
original files. `active-code-unchanged.json` confirms all twelve checkpoint code
inventories still match the main repository. Existing datasets are preserved.

# Shared implementation

Opt-in `parser_recover_mwt` separates the exact original text from the expanded
syntactic graph. Each ordinary syntactic token has an exact source span; every
word within a multiword expansion is protected from edits. Grammar edits are
rejected if their two-hop syntactic neighborhood touches a protected expansion.
Malformed offsets and missing dependency targets fail closed. The original
sentence, punctuation and contractions are never rewritten. Language-specific
rules and dictionaries remain separate and unchanged. Uncontracted paragraphs
and the default parser path retain the existing adapter.

`MorphologyCheck` also accepts a separate `cache_path` for independent diagnostics
and opt-in `instances` to reuse the existing local checker server pool. Default
single-server behavior is unchanged. No spelling or grammar check is disabled.
There is no wholesale proper-name exemption. Active profiles are unchanged.

# Paired CPU diagnostics

These are the first 300 paragraphs for French/Italian and first 150 for
Greek/pt-PT, not representative random language samples. Both modes processed
identical source paragraphs with the same model and corruption inputs.

| Language | Baseline candidates | Recovery candidates | Additional passing existing checker |
|---|---:|---:|---:|
| French | 139 | 222 | 69 |
| Italian | 28 | 139 | 86 |
| Greek | 7 | 12 | 3 |
| pt-PT | 21 | 78 | 44 |

Candidate totals are before independent screening; the final column counts only
newly recovered contracted sentences. They are not final production counts or
estimated full-corpus yields. Parser/candidate time remained broadly similar on
these small runs. Greek still has many dictionary and morphology rejections:
recovery alone does not resolve its low yield.

Agent inspection of 33 checker-passing recovered examples gave 30 provisional
accepts, two source rejects and one uncertain source. The Italian original has
an apparently missing predicate; a Portuguese original has an extraneous `de`,
and another has uncertain punctuation. Exclusion hashes are recorded in
`config/european-expansion/mwt-recovery-review/` for future profiles. Raw diagnostic
outputs are retained. No native linguistic validation or precision claim is made.

# Independent checker audit

Read-only cached responses were sampled from 80 committed candidate batches per
language. French: 2,498 checked / 890 flagged; Italian: 527 / 247; Spanish:
2,079 / 879; Romanian: 1,151 / 563. Nine Spanish responses were not yet cached
and were omitted. Spelling rules dominate the flags. Agent triage of 30 flagged
sentences per language found mostly names, titles and technical terms, but also
a real French typo and Romanian source extraction defects. Classification of
flagged spans does not certify the whole sentence or the spelling of names.

Therefore retain the current checker policy. A future name exemption needs
independent evidence and a review of source correctness; POS=PROPN or capitalization
alone is insufficient. The read-only audit can be repeated with
`scripts/audit_production_rejections.py --output PATH`.

# Throughput and composition

Two short /proc CPU samples found French parser processes using roughly 2–3
cores collectively while its checker used 7–9 cores. This supports screening
backpressure as a bottleneck; more parser workers alone is unlikely to help.
It is not a sustained throughput benchmark. An opt-in checker pool is prepared
for a separately measured run, not enabled on the active build.

Samples of 80 committed screened batches contain 3,816 Finnish pairs (80.3%
spelling) and 2,166 Estonian pairs (78.3% spelling), with all six configured
grammar families in both. These are before final quota/dedup selection. Their
high total yield is largely spelling and does not establish comparable grammar
precision; both still lack the same independent checker coverage.

# Evidence and reproduction

Artifacts are in `wiki/artifacts/european-expansion/mwt-recovery/`: paired reports,
recovered examples, agent review decisions, cached-checker audit, CPU sample,
active-code equivalence, test log and the implementation patch. The bounded
paired diagnostic runs from the isolated snapshot using
`python -m scripts.diagnose_mwt_recovery LANGUAGE --paragraphs N`.

End-to-end candidate validation uses the existing shared runner, separate
`mwt_recovery_validation` profiles/checkpoints/outputs, two parser processes per
language, and ten source documents each for French, Italian, Greek and pt-PT.
These validation profiles include the agent source exclusions. Full-size jobs
continue independently. No dataset has been uploaded by this experiment.

# Validation results

The isolated implementation passes **172 unit tests**, including source-span
reconstruction, protected expansion edits, dependency-context vetoes, neighboring
sentence offsets, end-to-end pair construction and default/pool checker routing.
`implementation.patch` passes `git apply --check` against the unchanged main
working tree. This is a reviewable patch, not an applied production migration.

A real 40-sentence French checker comparison produced identical hard diagnostics
with one and two local servers. The cold small-batch times were 5.02 versus 3.55
seconds; they do not establish a sustained speedup. Both /proc samples showed
French screening work exceeding parser CPU usage.

The earlier suggestion to skip multiword sentences before dependency inference
is superseded for recovery profiles: those sentences can now yield valid edit
candidates. The existing overlength filter remains. No additional rejection
shortcut was introduced without output-equivalence evidence.

Separate full-cap profiles are prepared at
`la_output/resources/european-expansion/mwt_recovery_next/` for the four tested
languages. They preserve the 478,930 cap, existing split targets and source
exclusions; French opts into two checker instances and 16 screening clients.
These profiles require the isolated code snapshot and fresh `european_mwt_v2`
checkpoints. They have not been launched as full-size jobs. Do not apply the
patch to the main tree while its pinned production runs still need it.

After the three recorded agent source exclusions, the paired diagnostic retains
199 newly recovered checker-passing candidates: French 69, Italian 85, Greek 3,
and pt-PT 42. These remain candidate pairs; 33-row agent inspection is not a
precision estimate and does not replace native review.

All four end-to-end exports passed independent validation: el 1 pairs, it 46 pairs, fr 64 pairs, pt-PT 3,825 pairs. Exact reconstruction, correction round trips, task views, artifact hashes and document split isolation passed. These ten-document smoke tests are not balanced production datasets; the Portuguese documents all hash to the training split. Full-size profiles retain the original split quotas. See `export-validation.json` for exact split counts.

# Superseding deployment

The user subsequently authorized installation and a campaign restart. The
[European campaign upgrade](european-campaign-upgrade.md) records the installed
changes, preserved legacy checkpoints and active v2 run. Earlier statements
above describing an isolated-only implementation are historical.
