---
type: Experiment
title: CPU expansion of the twelve European pilots
description: Document-preserving clean-source candidates, descriptive lexical evidence, training-only correction mining and independent diagnostics.
status: draft
generated: {by: codex, at: '2026-09-26T07:39:39.896503+00:00'}
sources:
  - id: wikipedia
    resource: https://huggingface.co/datasets/wikimedia/wikipedia
    title: Wikimedia Wikipedia snapshots
  - id: europarl
    resource: https://www.statmt.org/europarl/
    title: European Parliament corpus and source conventions
  - id: unimorph
    resource: https://unimorph.github.io/doc/unimorph-schema.pdf
    title: UniMorph morphological feature schema
  - id: boyd
    resource: https://github.com/adrianeboyd/boyd-wnut2018
    title: Falko-MERLIN training corrections
  - id: geccc
    resource: https://github.com/ufal/GECCC
    title: Czech GECCC corpus with predefined splits
  - id: uagec
    resource: https://github.com/grammarly/ua-gec
    title: Ukrainian GEC-only annotations and training split
---
# Scope and retained baselines

The user authorized all non-GPU work. All parsing in this experiment explicitly
uses Stanza `use_gpu=False`; no GPU generation, inference or training is required.
This continues [the twelve pilots](european-pilots.md). Their `audited_v2` outputs
and language profiles remain intact. New resources and profiles live in
`la_output/resources/european-expansion/`; new datasets use separate run IDs.
The new candidates are not an uploaded or production-certified release.

# Shared code and language isolation

`dala.canonical_source` accepts checksum-pinned, document-preserving JSONL. The
existing pipeline still handles generation, deduplication, splitting and export.
Article identity excludes the snapshot revision so an article retains its split
across revisions. Duplicate identities and source/manifest language mismatches
fail; the shared document ordering also rejects source/profile language mismatch.

`prepare_european_sources` normalizes source formats from input recipes. Language
choices are in `config/european-expansion/sources.json`. The expanded parser
profiles explicitly run multiword expansion. An unalignable contracted sentence
is represented as an opaque X root, which the existing source screen rejects;
neighboring sentences retain their original offsets and dependency trees. Earlier
profiles retain their parser behavior. Exact-offset and sentence-isolation tests
cover this change.

`prepare_unimorph_inputs` converts lexical tags from separate language recipes in
`config/european-expansion/morphology/`. Grammar controllers and definitions remain
in the original language inputs. Unknown tags cannot generate substitutions but
retain their known analyses for ambiguity vetoes. Portuguese personal infinitives
remain nonfinite; Slavic ESS-to-locative and Finnic case conventions are explicit
inputs. Possessor features remain separate. Ambiguous comma-separated analyses
now veto each licensed feature value in the shared compiler. Lexical evidence is
opt-in and does not promote one-off UD attestations.

`prepare_european_expansion` assembles versioned inputs for the original
`MorphologyPack`. The same pack now also accepts reviewed observed nonword maps;
each application still requires a recognized original, an unrecognized replacement
and an eligible non-name token. No language-specific Python generator was added.

# Sources and resource status

All twelve CPU parser model packages were downloaded and checksummed. Settings
are in `la_output/resources/european/{code}/live-parser-settings.json`.

Eleven languages use a downloaded first shard of Wikimedia's November 2023
Wikipedia snapshot at immutable repository revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. The bounded source diagnostic samples
100 complete articles per language, 600–6,000 characters each. Article IDs,
URLs, raw text and hashes survive preparation. Wikipedia is a candidate source,
not clean gold. Text extraction defects and mixed-language passages were found
in inspection.[^wikipedia]

pt-PT instead uses the OPUS Europarl v8 Portuguese raw archive. Its source is
European Parliament Portuguese, with historical spelling retained. One complete
bounded chapter per sitting date is sampled; chapters from a sitting cannot be
split across train/validation/test. Q4 2000 is excluded following the original
corpus's held-out convention. The first bounded pool has 100 sitting groups from
252 dates with eligible bounded XML files. This is not an estimate of full corpus
capacity. Generic Portuguese Wikipedia is not silently labeled pt-PT. The OPUS
license redirects to original source terms; redistribution review is still open.[^europarl]

UniMorph resources were pinned and downloaded separately for all twelve languages.
Only paradigms of lemmas already present in that language's UD training inventory
are converted. Many resources originate in Wiktionary and are **descriptive lexical
evidence, not normative dictionaries**. No form is added to the spell-checker's
normative word list. French/Italian/Portuguese resources are principally verbal;
Estonian has little adjective coverage. The generic Portuguese morphology does
not certify pt-PT: the European dictionary and existing standard-specific syntax
constraints still apply. Czech MorfFlex and Ukrainian VESUM compressed extensions
were not imported. Multiword forms and unknown-tag generation are excluded.[^unimorph]

# Correction evidence and inspection

Training-only mining uses the existing M2 reader and a shared preparation wrapper.
German Falko-MERLIN and Czech GECCC corrections have automatic ERRANT categories
over human-corrected text. Ukrainian uses the upstream annotation reader on the
GEC-only training directory; fluency annotations fail closed and documents, rather
than annotators, determine support. No held-out correction text is mined.[^boyd]
[^geccc] [^uagec]

| Language | Supported correction patterns | Dictionary-screened spelling candidates | Agent-provisionally accepted maps |
|---|---:|---:|---:|
| German | 2,191 | 255 | 47 |
| Czech | 5,958 | 2,333 | 50 |
| Ukrainian | 486 | 12 | 2 |

Support requires at least three distinct source sentences for M2, or three distinct
source documents for Ukrainian. These support units are not interchangeable.
Only the explicitly reviewed subset becomes rules. Candidate lists contained valid
German umlaut transliterations such as `fuer` and `Maenner`; these are not activated.
Names, colloquial forms and uncertain variants also remain outside the reviewed
subset. Word-pair agent inspection is not native validation or a precision estimate.
Raw grammar patterns remain contextual evidence, not unconditional substitutions.

A separate extraction from pinned local LanguageTool 6.6 XML inventories explicitly
grammar/spelling-typed expert-authored examples. These are **not observed mistakes**
and are not activated as context-free rules. Shared Portuguese examples still need
pt-PT applicability review. Untyped/style/complex alternative examples are excluded;
zero extracted fixtures does not establish absence of useful rules in a language.
Receipts include entity-resource checksums. Local rule code is LGPL-2.1-or-later;
this does not establish a license for any unrelated correction corpus.

# Assessment and artifacts

The initial source-only diagnostic `sources_cpu_v1` contains 2,568 pairs from 100
source documents per language, using the original limited morphology. All twelve
exports passed checksums, exact reconstruction, task-view consistency and split
isolation. This diagnostic predates sentence-level multiword isolation and lexical
expansion. Its counts should not be presented as final expanded capacity.

The expanded diagnostic `lexical_cpu_v1` contains **3,443 pairs** and has nonempty train/validation/test splits for every language. Agent inspection sampled 112 pairs (one per selected family plus three uniform examples per language): 99 provisional accepts, 6 rejects and 7 uncertain. All 13 non-accepted originals were excluded from the next build. This stratified inspection is not a precision estimate. An additional Estonian proper-name example was excluded.

The resulting run is `audited_cpu_v1`. It adds general rejection of empty parentheses, repeated full stops, doubled ASCII spaces and letter-number extraction labels; it protects capitalized lemmas outside German and checks originals with local LanguageTool in nine supported languages. Earlier diagnostic datasets remain intact. Measurements and final audit outcomes
are recorded in `wiki/artifacts/european-expansion/assessment.json` and the run's
coverage/validation receipts. A local CPU LanguageTool audit is an independent
triage signal: dictionary/name alerts are often false positives, and silence does
not certify grammaticality. Czech, Finnish and Estonian have no supported local
LanguageTool backend in this runtime. No native-speaker validation has occurred.

Source inspection found empty parentheses, leftover labels and mixed-language
material. Source cleanliness therefore remains a separate requirement from good
corruption rules. Grammar and spelling breadth must be assessed per language and
per family; a high count of compiled substitutions is not evidence of correctness.

# Reproduction

Use `/tmp/dala-six-venv/bin/python`. `lxml==6.1.3` is needed only for XML fixture
extraction; normal dataset generation retains the existing dependencies.
Do not reuse output IDs. Pinned download receipts are beside each resource; the source/morphology download lock is in `config/european-expansion/resource-lock.json`. Original UD/dictionary inputs are prepared by the earlier pilot runbook. Correction-repository revisions, training file hashes and the German release archive digest are in the per-language evidence recipes; materialize those pinned repositories/files before mining.

```sh
python -m scripts.fetch_european_expansion
python -m scripts.prepare_european_sources --run-id NEW_SOURCES
python -m scripts.prepare_unimorph_inputs
python -m scripts.mine_multilingual_evidence
python -m scripts.extract_rule_examples
python -m scripts.prepare_european_expansion --run-id NEW_PACK --source-run NEW_SOURCES
python -m scripts.apply_european_audit_policy --input la_output/resources/european-expansion/NEW_PACK --output la_output/resources/european-expansion/NEW_AUDITED_PACK
python -m scripts.run_european_pilots --run-id NEW_RUN --profile-root la_output/resources/european-expansion/NEW_AUDITED_PACK --workers 4
python -m scripts.report_six_language_coverage la_output/european_pilots/NEW_RUN/* --output wiki/artifacts/european-pilots/NEW_RUN/coverage
python -m scripts.audit_multilingual_checker la_output/european_pilots/NEW_RUN --output wiki/artifacts/european-pilots/NEW_RUN/checker
python -m unittest discover -s tests
python -m scripts.validate_wiki
```

Further scale-up can use the existing resumable batch pipeline. Do not weaken
variant vetoes, document grouping or source quality merely to fill a pair quota.
Shared-compiler recompilation preserves the complete existing Nynorsk mapping inventory; receipt: `wiki/artifacts/european-expansion/compiler-compatibility.json`. Danish legacy generation was not changed.

Remaining corpus-access, annotation-scope and per-family linguistic review gaps
are recorded rather than attributed to GPU availability.

[^wikipedia]: Snapshot schema and per-language immutable download receipts.
[^europarl]: Parliamentary provenance and original held-out convention; source terms need separate export review.
[^unimorph]: Universal feature schema; individual language README files describe differing coverage and provenance.
[^boyd]: Falko and MERLIN have separate licenses; Wikipedia edits from the same archive were not mined.
[^geccc]: Only `data/train/sentence.m2` was mined, with its byte hash and repository revision retained.
[^uagec]: Only GEC-only training annotations; test data and fluency corrections were not mined.

# Final selected CPU pilots

The selected outputs contain **3,032 distinct pairs: **1,081 grammar** and **1,951 spelling**. Every language has nonempty train, validation and test. These are additional versioned source-expansion pilots; the original 9,346-pair annotated-prose pilots remain available and are not being replaced or added to this count.

| Language | Pairs | Grammar | Spelling | Train | Validation | Test | Grammar families |
|---|---:|---:|---:|---:|---:|---:|---:|
| ca | 125 | 80 | 45 | 89 | 22 | 14 | 6/8 |
| cs | 184 | 75 | 109 | 170 | 6 | 8 | 7/10 |
| de | 415 | 165 | 250 | 346 | 24 | 45 | 9/11 |
| el | 185 | 78 | 107 | 141 | 23 | 21 | 5/8 |
| es | 194 | 170 | 24 | 150 | 17 | 27 | 6/8 |
| et | 411 | 89 | 322 | 345 | 39 | 27 | 6/6 |
| fi | 589 | 121 | 468 | 434 | 46 | 109 | 4/6 |
| fr | 250 | 67 | 183 | 212 | 27 | 11 | 5/8 |
| it | 55 | 36 | 19 | 41 | 5 | 9 | 6/8 |
| pt-PT | 140 | 84 | 56 | 117 | 15 | 8 | 5/8 |
| ro | 104 | 24 | 80 | 92 | 2 | 10 | 5/8 |
| uk | 380 | 92 | 288 | 305 | 27 | 48 | 6/9 |

Grammar families means selected/configured, including configured-but-inactive families in the denominator. The complete missing-family and distinct-substitution inventories are in `coverage/coverage.json`; counts do not establish precision.

Greek was extended to 1,000 complete Wikipedia articles and pt-PT to all 252 eligible bounded sitting groups after stricter screening emptied the first Greek test split. Both extensions exercised the existing resumable batch pipeline with eight CPU parser workers per language. Other pools still contain 100 source documents each. The final selection is given by `wiki/artifacts/european-expansion/current.json`, not by combining overlapping run directories.

Observed nonword maps now take precedence over synthetic character edits **within the spelling family** when applicable; grammar priority is unchanged. A first attempt exposed capitalized-map validation and mapping-only export bugs, both fixed with an end-to-end regression test. Failed `audited_cpu_observed_v2` checkpoints are retained, not selected. After the final source exclusions, the German and Czech datasets realize five and seven observed-map applications respectively; Ukrainian has two configured maps but no realized observed-map example in this source pool.

A fresh 24-pair Greek/Portuguese sample gave 23 provisional accepts and one uncertain Greek punctuation construction. Inspection of the 14 observed-map applications also found an unmarked German film title and an uncertain Czech source construction. Those three originals were excluded through the shared exporter into new candidate versions; parents remain intact, and manifests record parent/exclusion hashes. Final manifests keep candidate status: no native validation is implied. Policies are in `config/european-expansion/final-source-review/`.

Automatic evidence: **158 tests passed**, all twelve selected exports passed independent artifact/task/reconstruction/provenance/split checks, resource-lock verification passed, and Nynorsk mapping recompilation remained identical. All Stanza inference was CPU-only. No dataset was uploaded.

Remaining pre-production work: fill the documented missing grammar families; validate accepted variants and lexical/tag conversions per language; broaden real-error evidence beyond de/cs/uk where access permits; improve source cleanliness and title/name detection; resolve Europarl redistribution terms; and obtain native linguistic validation. None of those requirements is a GPU bottleneck.

# Follow-up coverage diagnosis

Read-only inspection found 28 missing language/family cells, of which six have
zero compiled mappings: Finnish determiner case/number, Romanian adjective/determiner
case, and Ukrainian adjective/determiner number. The other 22 have compiled rules
but no retained final examples. Detailed mapping/eligibility counts are in
`wiki/artifacts/european-expansion/gap-diagnosis.json`.

Concrete representation gaps: Finnish resources contain zero DET analyses but do
contain demonstrative PRON paradigms; the configured determiner families need
language-specific syntactic/POS licensing. Romanian input case values frequently
combine Acc,Nom or Dat,Gen; the compiler excludes multi-valued features as edit
endpoints, and nominal context matching also needs consistent set semantics.
Ukrainian number compilation preserves gender/animacy signatures too rigidly across
singular/plural paradigms. A read-only counterfactual ignoring both features yields
769 adjective and 184 determiner substitutions, demonstrating a compatibility
bottleneck, not proving those substitutions safe. The remedy is conditional,
language-specific feature applicability, not blanket feature removal.

French nominal evidence remains tiny (22 adjective-gender, 3 determiner-number and
14 determiner-gender mappings); pt-PT adjective number has only two mappings.
Broader nominal paradigms and reviewed closed-class inventories are needed.

Finite agreement currently requires an explicit, allowlisted PRON subject with
matching parser features. This excludes ordinary noun subjects and omitted
subjects by design. Zero eligible candidates for several languages could reflect
source composition, parser/lexicon feature mismatch, or these context restrictions;
per-gate counters are needed before attributing all gaps to any one cause.
Greek adjective number had four eligible sentences, pt-PT verb number four and
perfect-form one, and Ukrainian verb person one, despite no final examples. These
are downstream selection/screening/audit gaps, not missing generators.

Next priorities: diagnose rejection gates; fix the six zero-mapping families with
negative/valid-variant tests; strengthen French/Portuguese nominal resources; add
unambiguous lexical-noun subjects for verb number while retaining stricter person
licensing; then retrieve family-targeted clean contexts and require audited examples
per family. Do not infer the intended subject from the verb being corrupted in a
subjectless clause. The configured inventory itself is not exhaustive grammar
coverage. No dataset or rulebook was changed by this follow-up diagnosis.

# Coverage follow-up

The fixes, expanded pilots, full-source preparation and remaining review gates are recorded in [European coverage preparation](european-coverage-preparation.md). Earlier counts above describe retained historical runs.
