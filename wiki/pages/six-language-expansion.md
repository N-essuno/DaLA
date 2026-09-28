---
type: Plan
title: Six separate DynaWord language datasets
description: Accepted size and coverage contract, pinned resources, candidate implementation and release gates for Polish, Swedish, Bokmål, Nynorsk, Faroese and Icelandic.
status: draft
generated: {by: codex/gpt-6, at: '2026-09-24T11:02:51.221719+00:00'}
sources:
  - id: lt
    resource: https://dev.languagetool.org/languages
    title: LanguageTool language support
  - id: giella
    resource: https://github.com/giellalt/lang-fao/tree/f60c642e3ab6a5ada26306c959c1eb9c7a743845
    title: Pinned Faroese linguistic resources and normative paradigm tests
  - id: dict
    resource: https://github.com/wooorm/dictionaries/tree/8cfea406b505e4d7df52d5a19bce525df98c54ab
    title: Pinned Hunspell dictionaries and upstream license notices
  - id: iceec
    resource: https://repository.clarin.is/repository/xmlui/handle/20.500.12537/73
    title: Icelandic Error Corpus
---
# Accepted contract

The user requested six separate datasets. Polish (`pl`), Swedish (`sv`) and
Norwegian Bokmål (`nb`) strive for 478,930 distinct original/corrupted pairs each.
Faroese (`fo`) should exhaust the reasonable clean source supply. Icelandic
(`is`) initially used that policy; the user subsequently capped it at 478,930
pairs, matching Polish, Swedish and Bokmål. Nynorsk (`nn`) initially used
the same policy; the user subsequently requested a 478,930-pair selected dataset
while preserving the complete 871,199-pair generated pool. In all
six, grammar and spelling coverage is an independent deliverable. No repeated
clean sentences or weakened quality thresholds to pad counts.

`config/multilingual_targets.json` records the contract and proposed dataset
names. Each standard has separate prompts, source configuration, rules,
dictionaries and outputs. Instructions name the written standard and preserve
meaning/style. Correct-to-correct transformations, conversions between standards,
paraphrasing, simplification and style editing remain out of scope.

# Current implementation and status

Six profiles live at `config/languages/{pl,sv,nb,nn,fo,is}.json`. They use the
existing pair export pipeline, exact offsets, document splits and deduplication.
`dala/parsing.py` adds Stanza parsing while leaving existing spaCy packs on their
previous backend. Unalignable multiword expansions are rejected, never rewritten.
`dala/language_packs/morphology.py` supplies candidate grammatical constraints;
`dala/character_operations.py` now supports explicitly opted-in Unicode letters,
internal deletions and configured substitutions. Existing rules retain their
ASCII behavior by default.

The resources are now independently curated paradigms rather than UD spelling
attestations. UD training vocabulary seeds the Polish/Icelandic lookups; UD
syntax models provide parsing, not error-frequency evidence.

| Standard | Inflection evidence | Analyses | Unique forms where recorded |
|---|---|---:|---:|
| Polish | SGJP / Morfeusz2 | 567,179 | 437,684 |
| Swedish | SALDO morphology | 998,333 | 855,950 |
| Bokmål | current normative Ordbank forms | 788,337 | 600,831 |
| Nynorsk | post-2012 normative Ordbank forms | 529,653 | 401,803 |
| Faroese | GiellaLT published normative generator | 2,699,850 | 947,705 |
| Icelandic | BÍN, normal forms only | 689,602 | 232,728 |

These are lexical analyses/word forms, **not correction-pair counts**. Distinct
Nynorsk tags (`eint`, `bu`) are explicitly converted. All allowed analyses are
retained, including syncretism and accepted alternatives. The Faroese generator
release is `grammar-fao/v1.0.1-beta.1`; pinned lexical stems supply query seeds,
and returned forms supply the paradigms. Its outputs replace the initial 6,988
analyses from example tests. GPL-3.0 source licensing is retained alongside
resource provenance; dataset-source licenses remain separate.

Scripts: `prepare_multilingual_resources.py`, `prepare_multilingual_parsers.py`,
`prepare_nordic_lexicons.py --fetch`, `prepare_inflection_lexicons.py`,
`prepare_faroese_paradigms.py`, `prepare_faroese_generator.py --fetch`, then
`configure_six_languages.py`. Run these in that order when rebuilding from
scratch: the initial resource stage writes the initial UD morphology.
All model/resource bytes have checksum receipts. Faroese generation preparation
uses Python 3.11 with `hfst==3.16.0.1`; exported resource consumption works in
the ordinary Python 3.13 pipeline. Dictionaries remain pinned independently.[^dict]

`scripts/configure_six_languages.py` compiles configuration and records configured
versus active rules in `wiki/artifacts/six-language-expansion/configured-coverage.json`.
Empty families remain visible. Initial grammar candidates cover agreement,
governed verb forms, pronoun case and language-appropriate preposition case.
Spelling includes internal swaps/deletions, consonant doubling, undoubling, and
language-specific letter/diacritic confusions. These productive spelling rules
are **synthetic nonword generators**, not claimed observed substitutions.

The new outputs are explicitly **candidate_rule_and_dictionary_screened**.
`MorphologyCheck` verifies dictionary recognition/nonrecognition; it is not an
independent grammar checker and must not be presented as one. Unlike Dutch,
LanguageTool has no full support for these six standards uniformly.[^lt]
Polish and Swedish additionally run local LanguageTool over the **original**
sentences; this is not a claim that every injected error is checker-detected.
GreynirCorrect was tested for Icelandic but missed inspected agreement errors,
so it is not used as a certification gate.

No production-quality or publication claim is warranted by resource availability
or successful reconstruction checks.

# Source selection

Immutable HF inventories and source cards are archived under
`wiki/artifacts/six-language-expansion/`. Initial selected pools:

- Polish: gov.pl, local-government articles, EUR-Lex, Biblioteka Nauki.
- Swedish: Akademiliv, Forskning & Framsteg, EU legal texts.
- Bokmål: explicitly labelled government-nob; expand after measured pilot yield.
- Nynorsk: government-nno and wikipedia-nno; mixed Målfrid needs standard filtering.
- Faroese: BLARK Small, Sosialurin, Wikipedia. Exclude ASR transcripts initially.
- Icelandic: court decisions, laws, state gazette, scholarly journals, Wikipedia.

Missing quality annotations are allowed only through an explicit
`source_and_sentence_screening` policy; an absent metadata file is never silently
treated as an excellent-quality annotation. Retain per-record licenses and authors
where available, particularly Polish academic articles. Sentence-level source
quality and source-specific extraction defects still require sampling.

# Remaining production gates

1. Inspect pilot originals and corruptions independently, stratified by family,
   rule, standard and source, plus a random sample. Record agent judgment separately
   from native-speaker validation.
2. Resolve morphology ambiguity and allowed variants. UD attestation alone is
   insufficient evidence that an unattested alternative is ungrammatical.
3. Add observed error evidence where available (e.g. IceEC), carefully excluding
   style/meaning edits; corroborate productive rules with authoritative resources.[^iceec]
4. Measure configured, eligible, selected and audited coverage separately. A large
   spelling yield cannot satisfy the grammar coverage requirement.
5. Validate language/standard assignment and cross-dataset train/test collisions.
6. Run resumable production only after the corresponding rules/source filters
   pass audit; report source exhaustion and all shortfalls without padding.
7. Audit completed datasets and prepare separate HF packages. No new publication
   has been performed as part of this expansion.

[^lt]: LanguageTool 6.6 lists substantial Polish support, limited Swedish grammar rules, and Norwegian browser spelling-only support; it does not establish equivalent coverage for Nynorsk, Faroese or Icelandic.
[^giella]: GiellaLT Faroese revision f60c642e3ab6a5ada26306c959c1eb9c7a743845, normative inflection tests and license archived locally.
[^dict]: Separate dictionaries for all six codes, including distinct nb and nn.
[^iceec]: IceEC contains categorized spelling/grammar errors; no automatic extraction has yet been validated for this expansion.

# Pilot findings and current work

Earlier `pilot_v2`–`pilot_v9` outputs are diagnostic and superseded by the
`audit_v10` builds, which include all selected source pools. Counts from these
small samples are not production totals. Full candidate generation started on 2026-09-24 after the audit-driven fixes.
`audit_v10` and `audit_v13` remain diagnostic pilots, not final releases. `report_six_language_coverage.py` records realized
families, operators, distinct substitutions, source mix and reproducible samples.

Agent inspection found unmarked UD errors, syncretic noun readings, incomplete
references (e.g. Norwegian sentences ending at `Meld.`), Icelandic anonymization
placeholders, a Polish surname mistagged as a determiner, and source agreement
errors even in government texts. Changes reject ambiguous replacement readings,
protect interior capitals, reject source extraction risks, and check source
adjective/noun compatibility. A singular/plural parser choice alone cannot
license an error in a noun such as Swedish `hus` or `resultat`. Unknown lexical
features never count as evidence of incompatibility. These checks sacrifice
some yield deliberately; remaining source/context errors still require review.

The Nynorsk pilot after tag correction realized eight grammar families plus
spelling, compared with three grammar families beforehand. This is coverage
evidence, not a precision result. The Faroese audit realized twelve grammar families plus all five spelling
operators. The final regression suite passed 122 tests. All six semantic
conformance suites passed, including explicit guards for valid alternatives.

The host's existing shared-filesystem Python/runtime files stalled on reads.
The matching runtime was reconstructed locally at `/tmp/dala-six-venv`; Java
and LanguageTool archives were extracted at `/tmp/dala-six-tools`. Archive
checksums match the existing pinned runtime. This is an execution workaround,
not a changed linguistic model. A separate `/tmp/dala-fst-venv` holds the
Python 3.11 preparation toolchain. No user environment was replaced.

# Full candidate run: scale_v1

Six resumable builds run under `scripts.run_six_language_scale` with 100 parser
workers (pl24, sv16, nb16, nn12, fo8, is24). Their profiles and generation inputs
are frozen. Checkpoints live in `la_output/{language}_scale_v1_checkpoints`;
logs and status receipts are in `wiki/artifacts/six-language-expansion/scale_v1`.
Large languages target 478,930 pairs; smaller languages exhaust selected sources.
Parallel fragments preserve every eligible paragraph and original offsets.

Timestamped snapshot (2026-09-24T06:40:48.770004+00:00):

| Language | Retained candidate pairs | Completed batches |
|---|---:|---:|
| pl | 3,464 | 91 |
| sv | 3,239 | 61 |
| nb | 123,685 | 666 |
| nn | 75,930 | 571 |
| fo | 5,208 | 214 |
| is | 1,961 | 111 |

These are intermediate candidate counts, before final cross-dataset isolation
and linguistic review. No six-language dataset has been uploaded.

`scripts.finish_six_language_scale` waits for completion, validates artifacts
and pinned inputs, isolates exact/near duplicate originals across datasets
(priority nn, fo, is, pl, sv, nb), and reports realized coverage and fresh samples.
Its state is `scale_v1/finalization.json`. Final candidate destinations are
`la_output/six_language_candidates_scale_v1/{language}`. A successful automatic
finalization still requires linguistic review; it is not human validation.

# Verification and audit evidence

- Regression: 122 tests passed (`/tmp/dala-six-regression-final.log`).
- Danish equivalence: 5,501 sentences, 77,014 per-rule comparisons; exact rows,
  CSV and RNG state across both split modes/two seeds (`danish-equivalence.json`).
- Real Nynorsk ordinary/fragmented/resumed builds: 206 identical canonical pairs
  and valid source offsets/checker evidence (`fragment-equivalence.json`).
- Semantic conformance: all six current language receipts pass
  (`conformance/summary.json`). Icelandic compound paradigm gaps remain documented.
- Cross-language isolation smoke: all six bounded pilots passed; no collisions
  in this sample, which is not evidence of no collisions at full scale.
- Agent diagnostic inspection: 96 earlier examples gave 89 provisional accepts,
  five rejects, two uncertain; 22 additional examples gave 20 provisional accepts,
  one reject, one uncertain. These purposive samples are not precision estimates
  or native-speaker gold. Flagged sources were excluded before launch.

Final fixes include union-valued morphological feature compatibility, unique
compatible dictionary lemma normalization, Faroese finite-person paradigms,
Polish modal controller inflections, and stronger source/extraction checks.
The Faroese Wikipedia raw source tag is `wiki`, not its folder name `wikipedia`;
the source configuration was corrected before launch. A separate corrected
Wikipedia pilot produced 24 candidate pairs.

# Early scale audit and replacement runs

A fresh 36-pair agent sample from the first 50 screened batches per language
(seed 240924) provisionally accepted 34 and flagged two. This small early-source
sample is diagnostic, not a final precision estimate. Full rows and judgments
are in `scale_v1/early-review`. The Swedish source flag is a likely missing
subordinator; final selection excludes its original by hash.

The Polish edit `ówczesnych słodyczy → ówczesnej słodyczy` exposed a systematic
resource gap: generating paradigms only from UD-seeded lemmas omitted singular
`słodycz` analyses of `słodyczy`. A missing alternative cannot license corruption.
`scripts.close_surface_analyses` now queries every retained surface against the
native lexicon and merges all supported normative readings, including unseeded
homonyms. Preparation invokes this step automatically. Polish now has 567,179
analysis groups; Icelandic 689,602. This adds readings, not new word surfaces.

Polish and Icelandic `scale_v1` runs were stopped and superseded by `scale_v2`
profiles using the completed surface analyses. The Polish conformance suite
includes the discovered example as a forbidden transformation and passes.
Bokmål, Nynorsk and Faroese continue unchanged. Swedish is resuming verified
`scale_v1` checkpoints after a local checker connection reset; the run supervisor
retries transport failures only, with a bounded retry count.

Finalization supports explicit per-language run IDs and late agent-review source
exclusions, preserving immutable generation inputs. The late-exclusion smoke
test removed exactly one test fixture and independently validated the remaining
245 pairs. Neither the fixture nor the smoke output is a linguistic rejection
or a release dataset.

Replacement PL/IS builds launched successfully at 06:47 UTC; all six worker
parents are alive. The finalizer is waiting on PL/IS `scale_v2` and the other
four `scale_v1` statuses. Current snapshot: `scale_v1/progress-snapshot.json`;
launch receipts: `scale_v1/replacement-builds-launch.json` and
`scale_v1/replacement-finalizer-launch.json`. Icelandic conformance also passed
after closure. Swedish resumed beyond its failed checkpoint.

# 2026-09-24 checkpoint recovery

At 08:10 UTC the builds had retained nb404,747, nn398,013, fo60,328,
is39,201, pl10,709 and sv5,357 candidate pairs. Only Icelandic was still
running. The nb/nn/fo parents raced a worker writing its candidate receipt;
pl/sv exhausted whole-process retries for local checker connection resets.
Automatic finalization correctly stopped after a build failure.

The operational fix publishes receipt/progress JSON atomically, waits for
submitted futures before accessing their receipts, releases completed futures,
and retries an identical local checker request up to four times on connection
reset/timeout. Exhausted retries still fail and never enter the checker cache.
No grammar rules, dictionaries, source filters, thresholds or selection order
changed. All 125 tests pass, including new failure/retry/publication regressions.

Icelandic was stopped at preserved checkpoints so all six can resume with the
same fixed code. `scripts.recover_six_language_checkpoints` verifies each reused
batch checksum and creates separate `*_recovery_v1_checkpoints` directories.
Original checkpoints/signatures are retained; migration receipts record both
code revisions and the full verified batch inventory. Only the two operational
code files and checkpoint destination may differ. The export carries this
migration history instead of claiming all reused pairs used the new code.

Evidence and archived previous code: `wiki/artifacts/six-language-expansion/
recovery-20260924`. Recovery is in progress; final linguistic review and
publication remain pending.

Recovery validation completed: all six checkpoint sets verified; 206 canonical
pairs match the archived pre-fix output exactly, and fresh ordinary/fragmented/
resumed builds agree. All six `recovery_v1` builds and their finalizer launched.
The authoritative current-run pointer is `wiki/artifacts/six-language-expansion/
current-run.json`. Logs/statuses are under `recovery_v1/`; final candidate
outputs will be `la_output/six_language_candidates_recovery_v1/{language}`.
Old failure statuses are preserved as history, not current run status.

Recovery live audit: all six screened checkpoint pools now realize grammar
and spelling (pl11, sv8, nb8, nn8, fo12, is12 grammatical families). These are
before-global-selection coverage counts, not final releases. A 65-example
family-stratified agent inspection provisionally accepted 58, rejected five,
and marked two uncertain. Flagged originals are included in late final-selection
exclusions. Most flags are source defects; one Icelandic infinitive was edited
incorrectly but mislabeled as adjective agreement. This diagnostic sample is
not a precision estimate. Receipts, full samples and judgments: `recovery_v1/
live-audit`. Generation continues with unchanged linguistic inputs.

# Completed Bokmål and Faroese candidate builds

Bokmål reached its 478,930-pair target; Faroese exhausted all 6,163 selected
batches and retained 88,237 pairs. Both passed independent artifact checks,
source provenance, task-view reconstruction, balanced labels and split checks.
Receipts and coverage: `recovery_v1/completed-audit`. Cross-dataset isolation and
late review exclusions have not yet been applied; these are candidate counts.

Bokmål has 358,821 grammar and 120,109 spelling pairs across eight grammatical
families. Faroese has 46,527 grammar and 41,710 spelling pairs across twelve
grammatical families; sources contribute BLARK70,421, Wikipedia16,966 and
Sosialurin850 pairs. Source exhaustion does not establish linguistic precision.

Fresh uniform 20-pair samples per completed dataset (seed240926) yielded 39
provisional agent accepts and one uncertain Faroese source, queued for exclusion.
An initial concern about `pabba` was resolved by the pinned normative inventory,
which attests nominative singular; this accepted variant was not excluded.
This is a small agent audit, not native-speaker validation or final certification.
Polish, Swedish, Nynorsk and Icelandic generation continues.

# Nynorsk cap with the full pool preserved

The user explicitly requested no deletion and a Nynorsk cap matching the
478,930-pair large-language target. The complete 871,199-pair dataset remains
untouched at `la_output/nn_dynaword_recovery_v1`, together with all checkpoints.
A separate selected export is complete at
`la_output/nynorsk_capped_478930/nn`: train383,144, validation47,893, test47,893.
Selection uses stable pair-ID order within existing splits, after review
exclusions and deduplication; no sentence, prompt or corruption is rewritten.

The final six-language isolation also receives `--pair-caps nn=478930` and uses
the full preserved pool as input. Capped-out rows do not enter the shared
duplicate index. Late audit exclusions can therefore be replaced from the
remaining pool without deleting or regenerating the original dataset. The cap
is a selection policy, not a new claim of linguistic validation.

The capped export passed independent artifact/task/provenance validation at
2026-09-24T10:40:52+00:00. Exact split counts and unchanged full-pool manifest
were checked; all eight grammar families and spelling remain represented.
Receipts: `recovery_v1/nynorsk-cap-validation.json` and
`recovery_v1/nynorsk-cap-selection.json`. All 19 full-pool artifacts remain
present at their original sizes. No original pairs or checkpoints were deleted.

# Icelandic target and 64-worker runs

The user requested an Icelandic cap of478,930 and64 workers per language.
The active Polish, Swedish and Icelandic jobs now use `scaled64_v1` profiles
with64 parser processes each (192 total), two CPU threads per parser, and
383,144/47,893/47,893 pair split targets. Completed datasets remain unchanged;
the runner now defaults to64 parser workers for every language on future runs.
Checker thread counts are separate from parser workers.

`scripts.migrate_execution_profile` verified and reused PL3,024 candidate/3,016
screened batches, SV2,423/2,417 and IS7,077/7,059. Original checkpoint directories
remain untouched. Code, rules, source filters, batch composition, seed, models
and all linguistic options are unchanged. Only worker count, stopping targets,
release-target metadata and the checkpoint destination may differ. Migration
receipts retain the complete earlier migration history.

Build and finalizer launch receipts are under `scaled64_v1/`. Finalization uses
the new PL/SV/IS runs and the completed `recovery_v1` runs for NB/NN/FO, with
explicit nn/is caps. The authoritative mapping is `current-run.json`.

# Explicit startup of all 64 workers

Live inspection of `scaled64_v1` found46–58 processes despite the64-worker
limit. On checkpoint-heavy resumes, the initial scheduling window included
already cached batches; Python's lazy pool then reused its smaller worker set.
`scripts.worker_pool.PrestartedProcessPool` now uses public executor APIs and
an initialization event to start every configured process before generation.
The parser initializer, batch tasks and deterministic selection are unchanged.
Its SHA256 is pinned in each profile and verified at startup and finalization.

A multiprocessing test confirms all requested processes exist before submitting
generation work. All126 tests pass after updating the now-obsolete uncapped
Icelandic/Nynorsk contract assertion. A fresh206-pair ordinary/fragmented/resumed
comparison matches the previous canonical output exactly. Evidence lives in
`scaled64_v2/verification.json` and `fragment-equivalence_prestarted.json`.

The active jobs and finalizer now use `scaled64_v2` for PL/SV/IS; earlier runs
and all their checkpoints remain preserved. The current-run pointer reflects
this replacement. NB/NN/FO remain completed under their recorded run IDs.
Both Icelandic and the selected Nynorsk dataset are capped at478,930 pairs.

Live process verification completed:64 Polish,64 Swedish and64 Icelandic
parser processes (192 total), recorded with PIDs and timestamp in
`scaled64_v2/worker-verification.json`. All three profiles enforce478,930 pairs
and383,144/47,893/47,893 split targets. Existing checkpoints are preserved.

# User-requested pause

Polish, Swedish and Icelandic `scaled64_v2` builds and the waiting finalizer
are paused. All202 processes in their two detached process groups were stopped;
verification found no live processes remaining. No dataset, checkpoint or
partial file was deleted. Last committed candidate/screened batch checksums
were checked independently. Preserved selected counts: PL123,655, SV164,392,
IS128,744. Completed NB/NN/FO datasets remain unchanged.

The durable pause receipt is `scaled64_v2/pause.json`; it contains original
launch commands, split counts and checksum verification. On a later explicit
resume request, launch its `resume_launches.builds.command` and
`resume_launches.finalizer.command` from the repository using
`/tmp/dala-six-venv/bin/python -u`, with detached sessions and the saved log paths.
Preserve/archive any stale PL/SV/IS terminal status files before starting the
finalizer, and refresh launch/PID receipts. Keep the current code, profiles,
resources and pinned pool implementation unchanged for direct checkpoint reuse.
The runner verifies completed batch receipts and reconstructs selection state;
unfinished batches may be recomputed. Each resumed language remains configured
for64 parser workers and478,930 pairs. Do not resume automatically while paused.

User subsequently requested resume. At 2026-09-24T11:38:05.903093+00:00 all pinned
inputs were verified unchanged, and the three builds plus finalizer were
restarted. The pause is no longer active; previous pause and launch receipts
remain archived under `scaled64_v2/resume-*`. All worker counts and caps
remain unchanged.

# Checker-session exhaustion and bounded reuse

At14:04 UTC PL/SV were found stopped at225,852/336,166 retained pairs. Local
HTTP connections failed with errno99 (Cannot assign requested address); the
finalizer stopped as intended. Icelandic continued with64 parser workers.

The screening function created a new thread pool per batch. LanguageCheck
retained each thread-local HTTP session until the end of the build, allowing
connections to grow with batch count. A pinned `PersistentScreeningPool` now
reuses screening threads across batches, drains outstanding tasks before each
batch exits (also on errors), preserves input/result order, and explicitly closes
its pool when the build ends. Requests, diagnostics, cache keys and linguistic
filters are unchanged. It does not suppress checker failures.

All128 tests pass. A live Polish LanguageTool comparison made160 identical
checks per mode and returned byte-equivalent parsed JSON responses; original
batch-scoped threads retained161 sessions versus9 with8 persistent threads.
See `screening_v1/live-checker-equivalence.json`. The production limit is32
screening threads per PL/SV checker, independent of64 parser processes each.

PL/SV use new `screening_v1` profiles and checksum-verified reused checkpoints;
IS continues under `scaled64_v2`, and completed datasets are untouched. Earlier
checkpoints and failure receipts remain preserved. The screening-pool code is
pinned by SHA256 in profiles and checked again during finalization.

PL/SV screening_v1 supervisors and the replacement finalizer have been launched.
Icelandic scaled64_v2 continues without interruption. Current-run metadata now
records both active build supervisors and the finalizer launch receipt so a
future pause can stop both groups safely. Startup worker counts are timestamped;
PL/SV are configured for64 parser processes and32 persistent screening threads.

# Increase remaining builds to 128 parser workers

The user requested 128 workers per remaining language. Swedish completed at
478,930 pairs; Polish and Icelandic move to `scaled128_v1`, each with128
parser workers. Only process count and checkpoint location differ from their
previous profiles. Target and split caps, linguistic settings, pinned worker
helpers, and Polish persistent screening settings remain identical.

Old process groups were stopped and all old checkpoint files preserved.
`scripts.migrate_execution_profile` checks committed batch hashes before
reusing them in separate checkpoint directories. The runner now accepts
`--parser-workers` so the override is explicit and reproducible. Three worker
and screening pool tests passed. These are execution checks, not linguistic
validation. Launch, migration, and live process evidence belongs under
`wiki/artifacts/six-language-expansion/scaled128_v1/`.

At 2026-09-24T19:21:57.350690+00:00, live process inspection confirmed128 Polish and128 Icelandic parser workers (256 total). Both retain478,930-pair caps. Migration verified PL14,805 candidate/14,770 screened batches and IS31,040 candidate/31,021 screened batches. Separate build launch receipts and the restarted finalizer are registered in current-run.json. Old checkpoints remain preserved.

# Pending Polish-to-Icelandic worker handoff

User requested that Icelandic increase from128 to256 parser workers when Polish
finishes. `scripts/handoff_icelandic_workers.py` runs as a detached watcher,
registered in current-run.json active_launch_receipts so a pause includes it.
It requires Polish's successful terminal build receipt and its process group
to exit, then stops the Icelandic128-worker group and waiting finalizer using
captured process-start identities. It preserves old checkpoints, checksums
committed batches into `is_scaled256_v1_checkpoints`, and restarts Icelandic
and the finalizer with the replacement run mapping. The478,930 cap and all
linguistic settings remain identical. If Icelandic is already terminal, no
restart occurs. Restarting includes checkpoint migration and replay overhead.

Evidence lives under `scaled256_v1`: handoff-launch.json, handoff-status.json,
handoff-inputs.json, preserved-progress.json, and eventual worker-verification.json.
The watcher verifies256 live parser processes before reporting completion.
Three automatic tests cover the successful-build gate, refusal to signal a
changed process identity, and stopping an owned detached process group.
No linguistic or human validation is implied.
