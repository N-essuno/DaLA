---
type: Implementation
title: Dutch DaLA from Dutch DynaWord
description: Selected next language, pinned source investigation and Dutch-specific corruption implementation.
status: draft
generated: {by: codex/gpt-6, at: 2026-09-21T21:14:14+00:00}
sources:
  - id: dynaword
    resource: https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/tree/d0158defd949699532e59dea5978c5542afb0400
    title: Dutch DynaWord pinned source snapshot
  - id: gender
    resource: https://taaladvies.net/woordgeslacht-algemeen/
    title: Taaladvies — Dutch grammatical gender and accepted variation
  - id: lt
    resource: https://github.com/languagetool-org/languagetool/tree/v6.6/languagetool-language-modules/nl
    title: LanguageTool 6.6 Dutch rules and examples
  - id: lexicon
    resource: https://github.com/OpenTaal/opentaal-wordlist/tree/b250510dda431785f962019167d1415198ff3905
    title: Pinned OpenTaal word list
---
# Scope

After authorizing the English upload, the user asked to choose Norwegian,
Swedish or Dutch, using DynaWord from danish-foundation-models for clean text.
The user then explicitly selected Dutch. The earlier Danish mention was
corrected to Norwegian and is not an active language request.

Implement a Dutch language pack in the shared pair pipeline, a pinned DynaWord
Parquet adapter and a reviewed pilot before scaling. Do not copy Danish
linguistic rules or reuse English grammatical forms. No Dutch upload is yet
requested. Keep original source text and stable offsets; exclude problematic
text instead of silently editing it into a supposedly clean target.

# Source investigation

The current Dutch snapshot has 20 data Parquets and 20 annotation Parquets;
its card reports 37.89B tokens, unlike stale search results showing just 560
rows. This is the whole mixed corpus, not usable clean-sentence volume.[^dynaword]
Pinned inventories and cards for the considered languages are retained under
`wiki/artifacts/dynaword-language-selection/`.

Initial candidates were government web text (`dienst_publiek_en_communicatie`),
PBL reports and Naturalis publications. Downloaded actual data and annotations.
PBL's first example has visibly split words, PDF extraction debris and altered
personal details despite an `excellent` source-quality annotation. Defer it.
Start with government web text: 127,715 records, joined to annotation records
by ID. Filtering and text deduplication leave 12,916 eligible documents
(140,656,324 characters). The pilot draws 150; upstream quality labels are
125 `good` and 25 `excellent`. Require complete, substantive, good/excellent content, then perform
sentence screening and a fresh linguistic audit. Annotations are machine-made
selection signals, not grammatical gold.

Data records expose ID, text, source, dates and token count; annotation records
provide quality/category signals. Original article URL and author are absent.
Do not invent them: preserve an explicitly labeled pinned dataset-file URL,
upstream row ID, source publisher and recorded license (government web: CC0).
The lack of original-article attribution is a limitation. Stable document
identity groups identical full texts across IDs.

# Corruption and checker design

Use guarded subject agreement and d/dt errors, article/number/gender and
demonstrative agreement, sourced lexical misspellings and productive spelling.
Restrict singular gender rules to unambiguous noun inventories; Dutch allows
both articles for some nouns, so parser gender alone is insufficient.[^gender]
Exclude ambiguous zij/ze, polite u, inversion and coordinated subjects from
initial subject-agreement rules. Preserve regional alternatives.

LanguageTool 6.6 returns several genuine Dutch grammar diagnostics as
`uncategorized`: `JIJ_LOOP`, `HET_RESULTATEN`, `EEN_LELIJKE_MEISJE`, etc.[^lt]
The old hard-type filter silently ignored them. Add an explicit per-profile
rule-ID allowlist; English defaults remain unchanged. Do not indiscriminately
promote all style/uncategorized diagnostics to grammar errors.

The Dutch carrier `Het woord is ...` can also flag valid isolated verb forms
like `word` and `ben`. Therefore LanguageTool carrier results alone cannot
establish nonword status. Add pinned OpenTaal vocabulary as a second veto on
all lexical and productive spelling replacements.[^lexicon] Curated checker
examples are not observed learner frequencies, and a dictionary screen still
needs contextual audit. Probe receipts are kept as diagnosis, including
rejected false-positive candidates, not as an accepted rulebook.

# Implemented resources and safeguards

The language pack is `dala/language_packs/dutch.py`, with all inventories in
`config/languages/nl.json` and `config/dutch_rules.json`. The source adapter is
`dala/dynaword.py`. The active rulebook has 50 rules: guarded agreement rules,
six sourced lexical misspellings, and three productive character operators.
These are normative/curated examples and synthetic extensions, not measured
learner-error frequencies. The lexical misspellings cover *hogedrukgebied*,
*onmiddellijk*, *kinderlijk*, *openlijk*, *adellijk* and *sowieso*.

The productive operators transpose internal letters, duplicate a consonant or
delete a doubled consonant. Original words must occur in pinned OpenTaal;
replacement words must not. Contextual diagnostics and isolated lexical checks
are also mandatory. The build does not yet implement Dutch random word
swapping/deletion: a Dutch validity analysis is required before enabling those.

Removed both `die → dat` rules after finding a grammatical counterexample:
*Ik hoor die mensen in de tuin zingen.* and *Ik hoor dat mensen in de tuin
zingen.* can both be correct. Merely changing the determiner's apparent gender
is therefore insufficient. A regression test preserves this abstention. Also
excluded proposed spelling targets *gemeentehuize* and *platvorm* because the
independent dictionary recognizes them; conservative exclusion is preferable
to asserting these are universally invalid spellings.

Limit each document to 12 deterministic hash-selected paragraphs, retaining
unchanged source offsets and original order. This keeps very long government
pages from dominating the pilot. The 150 pilot documents are selected by a
seeded document hash, not manually chosen for clean examples. Near-duplicate
sentences and exact text collisions are excluded by the shared pipeline.

# Reproduction and assessment

```bash
python -m pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_md-3.8.0/nl_core_news_md-3.8.0-py3-none-any.whl
python -m scripts.prepare_dutch_resources
python -m dala.pair_pipeline --profile nl --max-documents 150 --max-errors 1 --output-dir la_output/dutch_dynaword_pilot_final
python -m dala.validate_dataset la_output/dutch_dynaword_pilot_final
python -m scripts.assess_dutch la_output/dutch_dynaword_pilot_final wiki/artifacts/dutch-final-assessment --seed dala-dutch-dynaword-audit-v2
python -m unittest discover -s tests
```

Use one error per pilot pair for clear attribution during assessment; the pack
supports up to two errors per pair by default, with at most one grammar edit.
The two rows per task are the correct original and its corrupted counterpart;
this is deliberate label balancing, not two distinct corruption examples.
Dataset manifests freeze source revisions/hashes, parser, rules, profile,
checker and code hashes. The assessment independently checks every original
against its pinned source document and every spelling replacement against
OpenTaal, then creates a seeded uniform sample for agent linguistic review.
Do not overwrite the annotated sample by rerunning assessment into its folder.

87 tests passed after Dutch ambiguity, source-filter, checker-category and paragraph-cap regressions.
This verifies implementation invariants, not native-speaker linguistic quality.

# Pilot audit and corrective changes

The initial frozen build (`la_output/dutch_dynaword_pilot`) contains 2,495 pairs
from 149 contributing documents. It has 1,938 distinct substitutions, but 2,427
of 2,495 edits (97.3%) are spelling. Only 15 of 50 configured rules occur;
configured rule counts must not be presented as realized coverage.

A seeded uniform 100-pair agent review found 90 acceptable sources, eight
erroneous/incomplete sources and two uncertain sources. All 100 intended edits
introduced errors. Eleven supplementary examples cover the rarer families;
one has a source typo and an incorrect family label: the parser misread singular
*strekdam* as plural, so a real gender error was labeled an article-number error.
Keep these initial judgments in `wiki/artifacts/dutch-pilot-assessment/`.
They are not native-speaker gold or a measurement of the later corrected build.

Corrections made after this audit:

- Reject standalone subordinate clauses, standalone quotation tokens and
  missing spaces after punctuation. Retain normal apostrophes in *collega's*
  and *'s avonds*. Source text is never silently repaired.
- Explicitly exclude the eleven reviewed erroneous/uncertain source sentences,
  with links to the judgments. Preserve the original build and audit rather
  than overwriting evidence with post-filter results.
- Require a changed noun lemma for plural determiner rules, avoiding the
  observed singular-noun misclassification.
- Also require that a plural determiner's noun is a subject governed by a
  plural finite verb. Constructed counterexample: *Wij genieten van de reizen
  naar andere landen.* can become the grammatical *Wij genieten van het reizen
  naar andere landen.* Nominalization makes a mere noun POS/number test unsafe.
  Restricting the syntactic frame is conservative and reduces coverage.
- Review exact Dutch checker IDs that are tagged `misspelling` but describe
  agreement (notably *ik vindt*). Only explicitly configured IDs are reclassified
  as grammar. Add omitted article/demonstrative grammar IDs. English's default
  category handling is unchanged. The raw diagnostic coverage investigation is
  preserved in `wiki/artifacts/dutch-checker-coverage.json`.

The intermediate checker/source-filter build (`dutch_dynaword_pilot_v2`) has
2,297 pairs and realizes all six families, including eight d/dt examples and
one demonstrative example. It predates the final nominalization guard and is
not the final pilot. The initial investigation found hundreds of grammatically guarded candidates
with no usable checker diagnostic; coverage remains a substantial bottleneck. Do not
silently accept these candidates or claim balanced grammar coverage.

# Final pilot: amount, diversity and quality

The final frozen pilot is `la_output/dutch_dynaword_pilot_final/`, manifest SHA256
`e2374ee4055f538c52425962e01345c39396ec14484fc369e3689e458e329413`.
It contains **2,552 pairs / 5,104 rows per task**, from 149 contributing documents.
Splits contain 2,034 training, 213 validation and 305 test pairs, grouped by
source document. Maximum contribution is 42 pairs from one document.

| Corruption family | Pairs |
| --- | ---: |
| Productive and lexical spelling | 2,475 |
| Subject–verb agreement | 25 |
| Article gender | 24 |
| Article number | 19 |
| d/dt | 8 |
| Demonstrative agreement | 1 |

There are 1,976 distinct observed surface substitutions in the **generated
output**, not 1,976 learner-attested mappings. Twenty configured rules occur;
spelling remains 97.0% of pairs. The largest substitution, `de → het`, accounts
for 33 pairs (1.3%). Surface diversity is good but grammatical coverage is not:
some families are absent from validation/test, and demonstrative agreement has
only one example. Do not extrapolate this into a balanced general Dutch grammar
benchmark or estimate production-scale volume from this pilot.

A new seeded uniform 100-pair agent review found **91 acceptable sources,
7 erroneous/incomplete sources and 2 uncertain sources**. All 100 injected
changes introduce errors. Five originals had been inspected in the earlier
uniform/supplementary audit, so this is not a completely independent sample.
A separate 17-pair family supplement has acceptable sources and valid edits
in all 17 cases. It must not be pooled with the uniform sample to estimate
population precision. Judgments and full measurements are in
`wiki/artifacts/dutch-final-assessment/`.

Remaining failures include a missing subordinate verb, a malformed noun phrase,
a coordinated plural subject with singular *helpt*, incomplete transcripts,
and punctuation damage. Some conversational source documents have upstream
`good`/`complete` labels despite these errors. Thus metadata and LanguageTool
are useful filters but insufficient certification. The 90/100 initial and
91/100 final judgments do **not** establish a statistically meaningful quality
improvement. The final pilot is preserved with these flags, not advertised as
clean gold data. No native-speaker validation has been performed.

For immediate inspection, `la_output/dutch_dynaword_agent_reviewed/` contains
only the **108 explicitly accepted pairs / 216 rows per task** from the final
uniform sample and coverage supplement, across 80 documents. This subset is
selected by agent judgment, is not representative of the full population and
is not human gold. Recreate it from the frozen pilot and `decisions.csv` using
`dala.dataset_review --decisions`; blank/uncertain decisions are not accepted.

Automatic validation passed for both datasets: checksums, exact edit and
correction roundtrips, task-view consistency, balanced labels and document split
isolation. The full pilot also passed 2,552 independent original-text/source
roundtrips and 2,475 independent OpenTaal spelling-veto checks. The final
implementation passes **87 tests**, with its receipt in
`wiki/artifacts/dutch-test-receipt.json`.

The next scaling work should prioritize cleaner non-conversational source
selection, better complete-sentence screening, broader independently validated
grammatical coverage and native Dutch review. Simply scaling the present mix
would mainly produce a larger spelling dataset. No Dutch HF upload was made.

[^dynaword]: Snapshot/card read through the Hub API; source licenses vary by subset.
[^gender]: Normative evidence and explicit warning about dual-gender words.
[^lt]: Local pinned distribution inspected alongside live Dutch checker probes.
[^lexicon]: Independent real-word veto, with its license preserved alongside downloaded data.
