---
type: Implementation
title: Dutch source and corruption expansion for scaling
description: Evidence-driven source selection, productive morphology and validation of broader Dutch coverage.
status: draft
generated: {by: codex/gpt-6, at: 2026-09-22T02:46:10.701486+00:00}
sources:
  - id: source
    resource: https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/tree/d0158defd949699532e59dea5978c5542afb0400
    title: Pinned Dutch DynaWord subsets
  - id: morphology
    resource: https://dev.languagetool.org/developing-a-tagger-dictionary.html
    title: LanguageTool morphology export documentation
  - id: adjective
    resource: https://taaladvies.net/wel-of-geen-e-achter-een-bijvoeglijk-naamwoord-algemeen/
    title: Dutch adjective inflection rules and exceptions
  - id: replacement
    resource: https://github.com/languagetool-org/languagetool/blob/v6.6/languagetool-language-modules/nl/src/main/resources/org/languagetool/rules/nl/replace.txt
    title: Pinned Dutch correction inventory
  - id: research
    resource: https://backoffice.biblio.ugent.be/download/01HTFDVQ3Y3J7F79KGHEXQ3RZ1/01HTFE2THKJVWAYGHBJWGKPNDW
    title: Exploring LLMs' Capabilities for Error Detection in Dutch L1 and L2 Writing Products (2024)
---
# Objective and scope

The user requested sufficient Dutch source quality and corruption coverage to
scale beyond the 150-document pilot. They also requested a comparison of rule
counts with Danish and English. Preserve all earlier pilot/audit artifacts and
the original `nl` profile; develop `nl_expanded` for measured improvements.
No Dutch upload is authorized by this request.

# Source investigation

Downloaded and inspected actual data and quality annotations for EUR-Lex,
Wikiwijs, C5 Filtered and Auditdienst Rijk. Also range-read samples from the
beginning, middle and end of the 918,634-record Rechtspraak file. Cards and
inspection samples are in `wiki/artifacts/dutch-source-expansion/`.[^source]

- Select **modern EUR-Lex** (CC BY 4.0): professionally drafted legislation.
  The supplied `created` field is only a collection-wide range, not a document
  date. Require the first Dutch title-date match within the first 2,000 source
  characters to be 2000 or later; record the extracted year. This excludes much
  older spelling without inventing exact dates or article URLs. Titles, recitals,
  fragments and overly long sentences still require sentence filtering.
- Retain **government web prose** (CC0), now restricted to `excellent` quality
  annotations and excluding conversational content and descriptions indicating
  transcripts, speeches, podcasts, interviews or dialogue. Annotation labels
  alone are not certification; the earlier audit demonstrated their limitations.
- Defer **Wikiwijs**: the inspected educational material contains malformed
  sentences, learner-facing exercises and deliberately corrected text, so it
  is unsuitable as an undifferentiated source of correct targets.
- Defer **Auditdienst Rijk/PBL**: PDF line fragmentation, headers, extraction
  damage and substituted personal details are prominent in inspected material.
- Defer **C5 Filtered**: mixed web prose, visibly erroneous passages and modified
  names; no strong quality advantage over selected government text.
- **Rechtspraak** is a large potential future source, but range samples include
  joined headings, placeholders and missing spaces inside words. Do not equate
  its professional origin with clean extraction or count all rows as usable.

Both selected sources retain pinned data/annotation hashes. Original article
URLs remain unavailable, so provenance retains dataset-file coordinates and
upstream IDs. Round-robin document selection prevents the larger source from
swamping a bounded pilot; document hashes still determine split membership.

# Corruption expansion

Export the Dutch POS dictionary from the existing pinned LanguageTool 6.6
runtime, preserving the bundled dual CC BY 3.0 / BSD notice. Intersect with the
pinned OpenTaal vocabulary and derive **154,001 unambiguous singular noun forms,
8,643 verb paradigms and 12,112 adjective inflections**. These are lexical
resources, not learner-observed errors or numbers of independent rules.[^morphology]

The derived resource and its checksum are created by
`scripts/prepare_dutch_morphology.py`. Dictionary noun gender must agree with
contextual parser gender. Exclude dual-gender/number readings and unsupported
parts of speech. Allow up to two attached adjectives between determiner and noun.

Replace 30 fixed subject-agreement entries with three productive dictionary
rules (first person, singular and plural), retaining unambiguous pronouns and
abstaining on `zij/ze`, polite `u`, inversion and coordination. Add possessive
agreement (`ons/onze`), guarded indefinite-neuter adjective inflection, and
relative-pronoun agreement for an explicit inanimate-neuter noun inventory.
Keep LanguageTool diagnostics mandatory; do not accept candidates merely because
the generator believes them incorrect.

Adjective corruption only adds inflection in `een + uninflected adjective +
unambiguous singular neuter noun`. Do not remove inflection indiscriminately:
fixed expressions, function titles, rhythm and regional usage admit alternatives.
Do not alter relative pronouns with human antecedents.[^adjective]

An additional adversarial example is *De suiker toevoegen kost tijd* versus
*Het suiker toevoegen kost tijd*: nominalization can preserve correctness.
The expanded determiner guard therefore abstains when the apparent noun is
attached to an infinitival/clausal-subject governor. The regression also covers
the parser incorrectly calling *toevoegen* finite in this frame.

Extracted and individually inspected 44 additional one-word spelling mappings
from the pinned curated correction file. Exclude real-word targets, translations,
archaic alternatives and questionable mappings. Alongside the original six this
makes **50 lexical mappings**; all 50 pass Dutch carrier recognition/nonword
probes and independent OpenTaal vetoes. Published correction inventories are not
observed learner frequencies; support counts remain null.[^replacement]

The expanded inventory contains **71 entries in nine families**: 50 lexical
spellings, three productive character operators and 18 grammar entries. Compare
this with the previous Dutch 50 entries/six families, English 86 entries/nine
families (72 lexical spellings), and Danish 15 top-level operators (45 lexical
spellings nested in one operator). Entry counts encode different levels of
abstraction and are not a comparable measure of linguistic coverage.

# Source screening and pipeline changes

Add optional Dutch source checks for missing finite subordinate verbs, obvious
plural-subject/singular-verb disagreement, fragment-prone beginnings, and spaces
before punctuation. Carry forward all known erroneous/uncertain sentences from
both earlier audits as explicit, attributed exclusions. No source rewriting.

The shared checkpoint pipeline now dispatches its source adapter from the
profile, so Dutch can use the same resumable builder as English. English's
default document ordering and source adapter remain unchanged. The expanded
profile uses a deterministic round-robin source order only when requested.

# Research limits

The 2024 Dutch L1/L2 error-detection study describes its datasets as proprietary;
we did not obtain a reusable correction-pair corpus from it. Do not imply that
our dictionary rules or curated spelling entries were extracted from those
learner annotations.[^research]

[^source]: Pinned cards, downloaded Parquets and range samples, not search-index counts.
[^morphology]: Official export format; actual inventories derived from the locally pinned dictionary.
[^adjective]: Normative rules with explicit exceptions informed the narrow corruption direction.
[^replacement]: Curated checker replacements, not a frequency-ranked list of learner errors.
[^research]: The paper's abstract explicitly describes proprietary L1 and L2 data.

# Expanded pilot and production readiness

The recovered expanded pilot is frozen at
`la_output/dutch_curated_expansion_pilot`: 200 documents (100 per source),
7,547 parsed sentences, 2,297 checker candidates and **1,448 retained pairs**
(2,896 rows per task), from 176 contributing documents. All artifact, exact
source-offset, split-isolation and OpenTaal checks pass. Export originally
failed because the shared validator omitted `dictionary_mapping` from its
operator dispatch; the fix adds that existing validator, with a regression
covering acceptance and rejection through `validate_pairs`. **96 tests pass**.
No acceptance checks were relaxed to recover the build.

| Family | Retained pairs |
| --- | ---: |
| Spelling | 709 |
| Article gender | 627 |
| Adjective inflection | 42 |
| Subject–verb agreement | 39 |
| Article number | 11 |
| Demonstrative agreement | 9 |
| Possessive agreement | 9 |
| Verb d/dt | 1 |
| Relative-pronoun agreement | 1 |

45 of 71 entries realized; 649 distinct substitutions. Spelling falls from
97.0% in the original pilot to 49.0%, but article gender/number changes together
constitute 44.1%. Merely counting nine families conceals extremely sparse
relative-pronoun and d/dt examples. The most frequent exact substitution,
`de→het`, is 457/1,448 (31.6%). Government supplies 1,239 pairs; EUR-Lex 209.

A seeded uniform **200-pair agent audit** found **187 acceptable sources,
nine erroneous sources and four uncertain sources**. All 200 intended edits
were judged valid. All 16 supplementary rare-family pairs were judged usable.
Government: 160 acceptable / eight erroneous / three uncertain out of 171.
EUR-Lex: 27 acceptable / one erroneous / one uncertain out of 29. This is agent
inspection, not native-speaker gold; 93.5% strict usability is not evidence of a
statistically established improvement over the earlier 91/100 result. Source
errors include joined headings, broken word hyphenation, agreement errors and
malformed noun phrases. Frozen judgments, proposed exclusions and automatic
checks are in `wiki/artifacts/dutch-curated-assessment/`. No sampled rows were
removed from the measured pilot; proposed exclusions are not yet applied.

The actual filtered source pool contains **2,969 excellent government documents
and 7,304 modern EUR-Lex documents** (10,273 total). With the current
12-paragraph cap and source-specific pilot yields, a crude extrapolation is
`2,969 × 12.39 + 7,304 × 2.09 ≈ 52,053 pairs` before full-corpus duplicate losses.
This is not a quota guarantee or a confidence interval. It does not support
promising approximately 479,000 pairs, the English release size. More paragraphs
and additional clean sources require new measurement; lowering quality filters
would not establish capacity at the same quality.

The bounded checkpoint profile `nl_scale_probe` uses two parser processes and
50-document batches. Its 200-document build at
`la_output/dutch_checkpoint_probe` produces **exactly the same 1,448 canonical
pair records** as the ordinary builder, and artifact validation passes.
A second build reusing completed checkpoints at
`la_output/dutch_checkpoint_resume_probe` also validates and reproduces exactly
the same pair records. Checkpoint/resume evidence is stored in
`wiki/artifacts/dutch-curated-assessment/checkpoint-validation.json`.

**Decision:** scalable construction is technically demonstrated, but a full
production-quality release or English-sized quota is not yet justified. First
address source defects and article concentration, measure more documents and
source depth, and audit the resulting candidate independently. A bounded
larger build can serve this validation; no full production run or Dutch upload
has been launched in response to the readiness question.

# Follow-up

The user approved the larger validation experiment. Its completed 1,000-document
results, fresh audit, balance tradeoffs and final 2,312-pair curated artifact are
recorded in [Larger Dutch validation](dutch-larger-validation.md). The earlier
pilot and its judgments above remain unchanged.
