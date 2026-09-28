---
type: Experiment
title: European grammar coverage and CPU preparation
description: Opt-in language constraints, targeted source pilots, full-source preparation and measured remaining coverage and quality limits.
status: draft
generated: {by: codex, at: '2026-09-26T08:15:00+00:00'}
sources:
  - id: lefff
    resource: https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer/tree/bc0ebd0135a6cc78f48ddf184069b4c0b9c017d8
    title: Pinned Lefff morphological lexicon and LGPL-LR license
  - id: freeling
    resource: https://github.com/TALP-UPC/FreeLing/tree/0bae6b7f6d1b405e67658895de54b59bfb3b6338/data/pt
    title: Pinned Portuguese FreeLing lexical data
  - id: freeling-tags
    resource: https://freeling-user-manual.readthedocs.io/en/latest/tagsets/tagset-pt/
    title: Portuguese morphological tag definitions
  - id: europarl
    resource: https://www.statmt.org/europarl/
    title: Europarl provenance, corpus terms and held-out quarter
  - id: ep-terms
    resource: https://www.europarl.europa.eu/legal-notice/en/home
    title: European Parliament reuse and attribution conditions
---
# Scope

Continues [the CPU expansion](european-cpu-expansion.md) and its grammar-gap
diagnosis. All inference and resource preparation use CPUs. Earlier resources,
source pools and datasets are retained. Candidate data are not a release or a
claim of native linguistic precision. The final selected paths and independent
export checks are in `wiki/artifacts/european-expansion/coverage_final/`.

# Shared implementation and isolated linguistic inputs

`dala.rule_compiler` now supports opt-in set-valued edit features, required
features, and conditional feature applicability. All-analysis syncretism vetoes
remain in force, including unknown and veto-only analyses. The runtime uses
consistent set semantics for nominal agreement and records rejection reasons
per family: lemma, POS, individual feature, head ambiguity, controller, subject
and dictionary checks. Batch receipts retain these counters across restarts.

Only `config/european-expansion/grammar/CODE.json` supplies the language choices:

* Finnish licenses demonstratives `tämä`, `tuo`, `se` as PRON with PronType=Dem
  specifically in a `det` dependency to a noun. Local Finnish-TDT training data
  contains 814, 70 and 333 such attributive occurrences respectively. The compiler
  now produces 38 determiner-number and 330 determiner-case substitutions.
* Romanian retains case sets such as Acc,Nom and Dat,Gen. Only disjoint case sets
  can form an edit; the noun's alternative readings must also exclude the
  replacement. There are 3,570 adjective-case and 27 determiner-case substitutions.
* Ukrainian permits missing Gender only in plural number endpoints. Case,
  animacy and possessor distinctions remain protected. This produces 513
  adjective-number and 122 determiner-number substitutions, fewer than the
  earlier unsafe blanket-relaxation diagnostic.
* Finite-number rules can admit explicit, unambiguous count-noun subjects from
  separate small language-owned lemma inventories. Complex noun phrases,
  ambiguous number, coordination and expletives remain excluded. Person edits
  still require explicit allowlisted personal-pronoun evidence. Independently
  specified personal-pronoun features may fill missing parser features in a
  subject dependency; contradictions reject the source. No subjectless verb
  supplies evidence for its own intended person.

Optional coverage priorities apply only to targeted pilots. Full-source profiles
remove those priorities and retain ordinary deterministic grammar/spelling
selection. No fixed per-family production percentage is imposed.

# Nominal resources

The language-neutral `prepare_nominal_inputs` converter reads exact tag tables
from separate French and pt-PT recipes. It expands only existing UD training
lemma/POS combinations, preserves unknown tags as veto-only, and does not add
lexical forms to normative spelling inventories. Runtime dictionary checks still
apply. Resource URLs, immutable revisions, hashes and license files are pinned
in the resource lock and compiled receipts.

Lefff adds 4,568 eligible French analyses and 128 veto-only analyses. French
adjective-gender mappings increase from 22 to 1,092; determiner number from 3 to
23 and determiner gender from 14 to 26.[^lefff]

FreeLing adds 1,167 eligible Portuguese adjective analyses and 932 veto-only
analyses. Its upstream mixes Portuguese varieties: these inputs are explicitly
restricted to pt-PT training adjective lemmas plus the independent pt-PT
dictionary. No Brazilian corpus text or grammar patterns were imported. This
restriction is candidate evidence, not native proof of every form. Tag mappings
are explicit; common gender and invariable number retain multiple values.
The upstream COPYING exception identifies the Portuguese dictionary as GPLv3,
not the general FreeLing code license.[^freeling][^freeling-tags]

# Source preparation and scaling

Targeted retrieval selects complete original documents containing candidate
forms; person-family retrieval additionally requires an explicit subject word.
It never manufactures clean sentences or bypasses syntax/source screening.
Wikipedia document IDs remain stable across pools. Portuguese retrieval retains
sitting-date grouping and excludes Q4 2000, the upstream held-out convention.
The later Portuguese pilot admits complete chapters up to 30,000 characters;
the earlier 6,000-character chapter window was a substantial source limitation.

All 87 pinned Wikipedia shards for the eleven applicable languages (23.23 GB
compressed) and the complete Portuguese Europarl archive have been prepared as
canonical full-source JSONL. The full-source Portuguese adapter includes all
chapters per sitting, instead of one bounded pilot chapter. Original sources and
prepared outputs remain separate. `scale-source-lock.json` pins every Wikipedia
shard digest; each language's preparation receipt pins the canonical bytes.

`la_output/resources/european-expansion/scale_inputs_v1/` contains independent
CPU profiles, each capped at 478,930 pairs with train/validation/test targets of
383,144 / 47,893 / 47,893. They use eight parser processes and one parser thread
per process, shared checkpointing, and explicit shortfall reporting. Source
availability is not a promise of that many acceptable pairs. Full-size generation
has not been launched by this preparation step.

Europarl's archive delegates its license to the original sources. The corpus
site supplies a permissive research-use statement, while the Parliament's legal
notice specifies attribution and conditions for partial reproduction. This is
not a new CC license grant for our modified sentence pairs. Publication still
needs the final attribution/terms decision; corpus research preparation does
not remove that gate.[^europarl][^ep-terms]

# Concurrency correction

The first checkpointed Finnish diagnostic (`coverage_cpu_v5/fi`) incorrectly
rejected thousands of original words. Sequential replay accepted the same words;
unserialized eight-thread replay caused a native segmentation fault. Voikko
handles must not be concurrently called by screening threads. A shared
`SerializedDictionary` now locks each native handle while leaving parser
processes parallel. A 363-pair replay gives identical serial and eight-thread
results, all accepted at lexical screening. The rebuilt Finnish pilot has
6,772 pairs before agent exclusions; the faulty 434-pair version is retained
only as diagnostic evidence. Receipts are `voikko-thread-diagnosis.json` and
`voikko-thread-fix.json` under the expansion artifacts.

# Validation and review

Automatic tests cover disjoint case sets, accepted-alternative vetoes, conditional
feature applicability, possessor/animacy preservation, Finnish attributive-only
licensing, lexical noun ambiguity, explicit pronoun completion and serialized
native calls. Nynorsk's complete compiled mapping inventory remains identical.
The files covered by the earlier complete Danish row/CSV/RNG-equivalence run
have unchanged hashes; this is a compatibility check, not a newly rerun Danish
experiment. Receipts explicitly distinguish these forms of evidence.

Agent inspection is recorded per pair. Source errors and uncertain examples are
excluded through the existing exporter into new versions, leaving parents intact.
This includes a Czech magazine title masquerading as a declarative sentence:
removing it can reopen a coverage gap, which must be reported rather than hidden.
No native speaker has validated these pilots. Blank per-family review sheets and
independent local-checker diagnostics are included; checker silence does not
certify correctness. Czech, Estonian and Finnish have no supported local
LanguageTool backend in this environment.

# Reproduction and handoff

Use `/tmp/dala-six-venv/bin/python`. Inputs remain in the existing shared pipeline.
For fresh run IDs, prepare the pinned downloads, nominal conversions, source
pools, compiled resources, and audited profiles in that order:

```sh
python -m scripts.fetch_european_expansion
python -m scripts.prepare_nominal_inputs
python -m scripts.prepare_european_sources --run-id NEW_SOURCES --limit 1000
python -m scripts.prepare_european_expansion --run-id NEW_PACK --source-run NEW_SOURCES
python -m scripts.apply_european_audit_policy --input la_output/resources/european-expansion/NEW_PACK --output la_output/resources/european-expansion/NEW_AUDITED --coverage-targets wiki/artifacts/european-expansion/gap-diagnosis.json --parser-workers 8 --checkpoint-root la_output/european_pilots/NEW_CHECKPOINTS --review-directory config/european-expansion/final-source-review --review-directory config/european-expansion/coverage-review
python -m scripts.run_european_pilots --run-id NEW_PILOT --profile-root la_output/resources/european-expansion/NEW_AUDITED --workers 12 --max-documents 1000
python -m scripts.prepare_european_scale_inputs --profile-root la_output/resources/european-expansion/NEW_AUDITED --run-id NEW_SCALE_INPUTS
```

For targeted retrieval, pass `--target-profile-root` and `--gap-report` to source
preparation. `merge_candidate_pilots` uses the original exporter and deduplicator,
checks language/rule/split identity, and rejects conflicting document contents.
Rare-family priority resolves alternative edits of the same original. Parent
manifests retain generation diagnostics; overlapping parent counts are not
misrepresented as unique merged processing counts.

A later full-source candidate build uses the same runner with `--max-documents 0`
(the complete pool and configured split quotas), a fresh output run ID and the
prepared scale profile root. It does not publish. Repeating the identical command
resumes verified checkpoints; changing generation inputs requires a fresh run.

[^lefff]: Immutable mirror includes the morphology and original LGPL-LR license.
[^freeling]: Immutable lexical input and COPYING exception 17; resources retain their own licenses.
[^freeling-tags]: Only exact licensed tags are interpreted; unhandled tags remain veto-only.
[^europarl]: Research use, document markup, and the Q4 2000 test convention.
[^ep-terms]: General permission does not settle every derived-release attribution condition.

# Final measured selection

The final selected pilots contain **30,897 pairs**, including **11,687 grammar**
and **19,210 spelling** pairs. Splits total **24,978 train / 2,861 validation /
3,058 test**. All twelve languages have nonempty splits. Counts are distinct
canonical pairs; each task also exports a clean control for each pair.

| Language | Pairs | Train | Validation | Test | Grammar families |
|---|---:|---:|---:|---:|---:|
| ca | 1,197 | 973 | 126 | 98 | 8/8 |
| cs | 4,759 | 3,860 | 494 | 405 | 9/10 |
| de | 3,849 | 3,220 | 288 | 341 | 11/11 |
| el | 413 | 337 | 38 | 38 | 8/8 |
| es | 1,846 | 1,484 | 170 | 192 | 8/8 |
| et | 4,306 | 3,357 | 432 | 517 | 6/6 |
| fi | 6,762 | 5,425 | 660 | 677 | 6/6 |
| fr | 2,473 | 2,098 | 196 | 179 | 8/8 |
| it | 591 | 469 | 70 | 52 | 8/8 |
| pt-PT | 801 | 700 | 47 | 54 | 8/8 |
| ro | 1,203 | 935 | 113 | 155 | 8/8 |
| uk | 2,697 | 2,120 | 227 | 350 | 9/9 |

Coverage is **97/98 configured language/family cells**, compared with 70/98 in
the earlier selected expansion. All six zero-mapping gaps are fixed. Every
configured family is represented in eleven languages. Czech verb-person
agreement remains missing after excluding unmarked film/magazine titles and
an uncertain source punctuation construction. The productive mappings remain
available; this is a retained-clean-context gap, not a missing implementation.
A targeted pronoun-source pass recovered both Greek verb families without
weakening grammar licensing. Sparse families (sometimes only one example)
remain particularly important targets for native review and additional sources.

Recorded agent inspection covers 140 rows: 126 provisional accepts, 8 uncertain
and 6 rejects. These stratified, targeted inspections are **not a precision
estimate**. All flagged sources were excluded. A separate automatic source
boundary rule excludes internal unquoted question/exclamation punctuation,
which exposed titles and reference fragments; its removals are recorded
separately from agent judgments. Final native-review sheets contain 518 rows
with blank judgments. Earlier review packets remain pinned to their parents.

Automatic evidence: 165 tests passed; all twelve final exports passed independent
artifact hashes, edit reconstruction, task views, provenance and split-isolation
checks. Completed Finnish checkpoint replay produced byte-identical canonical
pairs in every split, without reparsing. The full-source preparation contains
13,123,130 documents and 36,443,406,118 text characters across all twelve languages.
The full-size candidate profiles are prepared, not launched. All source input
hashes and resource locks were verified. No GPU was used and no dataset uploaded.

Remaining work is linguistic/release work rather than a GPU prerequisite: secure
clean Czech person-agreement contexts; obtain native per-family judgments;
measure precision beyond this targeted inspection; and finalize source/lexicon
license and attribution handling before publication. Full-size acceptance rates
and attainable totals are unmeasured; configured targets are not achieved counts.

All 98 configured families now have active compiled mappings. The final independent
LanguageTool diagnostic sampled 450 pairs in nine supported languages and found
no hard original-sentence flags; its manifests match the final selections. This
is checker evidence only. The historical `perfect_verb_form` label can also cover
required participles in passive auxiliary constructions; it does not establish
coverage of every perfect tense or of all grammatical constructions in a language.
Detailed technical completion flags are in `coverage_final/technical-checks.json`.


# Full-size CPU launch

On 2026-09-26, the user authorized full-size generation with 128–192 CPU workers.
Run `european_scale_v1` was launched with **192 parser processes total: 16 for
each of the twelve languages**, one parser thread per process. Eight persistent
screening threads per language handle downstream checking. Native Voikko calls
remain serialized per handle. GPU visibility and inference are disabled.

Fresh run profiles preserve the prepared inputs and all earlier outputs. Each
language targets 478,930 pairs (383,144 train, 47,893 validation, 47,893 test),
with no document limit and shortfalls reported if its source pool is exhausted.
The existing worker-pool and persistent screening-pool helpers are reused and
checksum-pinned; their three focused tests passed. Completed pilots remain the
current selected datasets until the new candidates finish and are assessed.

Launch receipt and supervisor PID: `wiki/artifacts/european-expansion/european_scale_v1/launch.json`.
Per-language logs: `wiki/artifacts/european-pilots/european_scale_v1/`.
Checkpoints and progress: `la_output/european_pilots/european_scale_v1_checkpoints/CODE/`.
Candidate outputs: `la_output/european_pilots/european_scale_v1/CODE/`.
The launch is detached from the interactive session. No upload is performed.


Status snapshots can be refreshed without affecting the running jobs:

```sh
python -m scripts.report_european_scale_status wiki/artifacts/european-expansion/european_scale_v1/launch.json
```

The reporter records actual parser-process counts, retained pairs, splits and
interval throughput. Initial inspection confirmed all 192 parser processes alive
and retained checkpoints in eleven languages while the last finished startup.
Do not extrapolate a production ETA from model startup or early batches.

# Production yield diagnosis (2026-09-26)

Agent code inspection found that `StanzaParser` rejects every sentence containing
a multiword-token expansion, even when other words have exact source offsets.
It still runs POS/lemma/dependency inference before representing that sentence
as one opaque X token. This is an implementation limitation, not evidence that
the source sentence is incorrect. The opaque sentence also enters the sentence
counter and usually fails the minimum-word check as `length`. Thus parser
alignment and length counters overlap; previously reported sentence rates do
include these placeholders and are not rates of successfully aligned sentences.

Recommended follow-up: a separately versioned shared parser adapter that retains
original surface spans and expanded syntactic nodes, forbids edits to ambiguous
spans, and only permits grammar edits with fully supported controllers. Validate
exact reconstruction, dependency references and contraction cases before a new
run. Do not change the SHA-pinned active generation code. While retaining the
current rejection policy, skipping expensive inference for already-ineligible
sentences is another candidate optimization, subject to equivalence testing.

Independent checker rejection samples and grammar/spelling family distributions
should be audited before relaxing any filters. Finnish/Estonian lack the same
independent checker coverage and their high yield is not a quality comparison.
CPU-stage profiling is still needed to attribute French processing cost.

# Current campaign

The v1 run was stopped and continued as `european_scale_v2` with explicit legacy
batch provenance and a recovery pass for skipped contractions. See
[European campaign upgrade](european-campaign-upgrade.md) for the current status
command and receipts. Finnish retains its completed v1 output.
