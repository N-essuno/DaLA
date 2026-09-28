---
type: Implementation
title: Larger Dutch validation build and balance experiment
description: A 1000-document checkpointed build with stricter source filtering, explicit family caps and a fresh held-out-from-review audit.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-22T03:03:04.079898+00:00}
sources:
  - id: previous
    resource: ./dutch-scale-readiness.md
    title: Expanded Dutch pilot and production readiness
  - id: sources
    resource: https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/tree/d0158defd949699532e59dea5978c5542afb0400
    title: Immutable Dutch DynaWord source snapshot
---
# Scope

The user approved the proposed larger validation build, source filtering and
family balancing after the 200-document readiness assessment. This is not
approval to publish a Dutch dataset. Preserve previous pilot and audit artifacts.

# Construction

`config/languages/nl_validation.json` retains the expanded 71-entry rulebook and
pinned source/morphology resources. The build uses 1,000 deterministically ordered
documents (500 government and 500 EUR-Lex), 12 eligible paragraphs per document,
one edit per pair, four parser processes and 100-document checkpoint batches.
All original checker and exact-edit requirements remain in force.[^previous]

```sh
.venv/bin/python -m dala.pair_pipeline --profile nl_validation --max-documents 1000 --max-errors 1 --offline --output-dir la_output/dutch_validation_raw
```

Source changes are opt-in profile inputs:

- Carry forward 13 erroneous/uncertain originals from the prior 200-pair audit,
  with evidence links, alongside the existing exclusions (33 unique originals).
- Reject internal alphabetic hyphenation as a conservative extraction-risk
  filter. This intentionally also drops legitimate compounds; it does not label
  those sentences ungrammatical and does not rewrite source text.
- Reject obvious initial list markers and selected joined heading/sentence forms.
- Check unambiguous personal-pronoun number against a single finite predicate,
  including past tense. Abstain on coordinated/multiple-subject parses; a
  regression caught the parser treating coordinated `Mijn broer en ik` as two
  subjects of one verb.

Source filtering and checkpoint identity remain frozen during construction.
Raw output will remain available separately from any balanced subset.

# Balance experiment

`config/dutch_validation_balance.json` specifies **at most 25% article errors**
(gender and number combined) and **at most 60% spelling** in each split.
`scripts/balance_pairs.py` finds the largest feasible integer subset per split,
retains all uncapped grammar families first, prioritizes rarer article-number
examples within the article group, then uses stable pair hashes for selection.
No duplicate examples, sentence modifications or split moves are introduced.
These are deliberate training-mixture caps, not empirical learner frequencies.
The current full suite passes 102 tests; receipt:
`wiki/artifacts/dutch-source-expansion/validation-tests.txt`.

```sh
.venv/bin/python -m scripts.balance_pairs la_output/dutch_validation_raw la_output/dutch_validation_balanced --config config/dutch_validation_balance.json
```

The subset manifest records the parent manifest hash and validation, balancing
configuration and implementation hashes, exact per-split quotas and exclusions.
Independent exhaustive small-inventory tests verify maximality and fraction
bounds; separate tests cover order independence, rare-family retention and
failure on impossible inventories/multiple edits. Full artifact validation is
also exercised on a real earlier pilot export.

# Capacity measurement

Exact paragraph counts under the selected sources' paragraph filters:[^sources]

| Source | Documents | Cap 12 | Cap 24 | Cap 48 | Uncapped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent government | 2,969 | 32,862 | 53,318 | 70,463 | 101,185 |
| Modern EUR-Lex | 7,304 | 87,639 | 169,753 | 277,777 | 632,390 |

These counts demonstrate remaining source depth, not accepted sentence/pair
capacity. The fixed 12-paragraph limit uses 120,501 of 733,575 eligible paragraphs.
Additional paragraphs must pass the same sentence and corruption checks; older
52k-pair extrapolation applies only to the 12-paragraph setting. Evidence:
`wiki/artifacts/dutch-source-expansion/paragraph-depth-capacity.json`. Reproduce
with `.venv/bin/python -m scripts.measure_dutch_capacity`.

# Audit protocol

Create a fresh seeded sample from the frozen balanced output, excluding originals
already present in earlier review samples and supplements. This is uniform over
the remaining, previously unreviewed pair population, not over the entire output.
Record the population size and hashes of prior review inputs. Keep a separate
rare-family supplement; do not pool it into the uniform precision estimate.
Agent inspection and automatic checks do not constitute native-speaker gold.

A development-set replay with explicit sentence exclusions disabled found the
general filters catch four of the previous nine clear source errors, while
18/187 previously accepted originals trigger the new conservative extraction
filter. This illustrates a recall/retention tradeoff; it is not a fresh quality
estimate. Evidence: `wiki/artifacts/dutch-source-expansion/source-guard-replay.json`.

# Completed results

The run parsed **37,280 sentences** from 11,595 paragraphs, generated 10,262
candidates, and retained **6,474 pairs** after screening and duplicate filtering.
All stages preserve the pinned input texts and canonical pair records.

| Stage | Pairs | Rows per task | Purpose |
| --- | ---: | ---: | --- |
| `la_output/dutch_validation_raw` | 6,474 | 12,948 | Full screened output; no family downsampling |
| `la_output/dutch_validation_balanced` | 2,325 | 4,650 | Frozen population used for the fresh audit |
| `la_output/dutch_validation_curated` | 2,312 | 4,624 | Final review-informed subset, with 13 flagged originals excluded |

The balancing caps remove 4,149 pairs from the raw output. This is a substantial
size tradeoff, not extra data generation. In the curated subset, spelling is
60.0% and articles collectively 25.0%; the remaining 15.0% covers other grammar.
The largest exact substitution, `de→het`, falls to **18.5%** (428/2,312).

| Family | Raw | Final curated |
| --- | ---: | ---: |
| Spelling | 3,446 | 1,387 |
| Article gender | 2,608 | 507 |
| Article number | 71 | 71 |
| Adjective inflection | 162 | 161 |
| Subject–verb agreement | 111 | 111 |
| Demonstrative agreement | 37 | 37 |
| Possessive agreement | 37 | 36 |
| Relative-pronoun agreement | 1 | 1 |
| Verb d/dt | 1 | 1 |

The final subset contains 680 contributing documents, 50 realized rule entries
and 1,210 distinct substitutions. A descriptive reparse identifies **60 verb
lemmas** in subject agreement, **93 adjective lemmas**, 369 head-noun lemmas in
article-gender edits, and 44 in article-number edits. One article-gender head was
unresolved by the isolated-sentence reparse; this reparse is a diversity proxy,
not an additional licensing or precision judgment. Evidence:
`wiki/artifacts/dutch-validation-assessment/lexical-coverage.json`.

Curated splits: train 1,820 / validation 232 / test 260 pairs. Source counts:
government 1,932 / EUR-Lex 380. Existing document split assignments are preserved;
no rare example is copied or moved to fill evaluation splits.

## Fresh audit

The uniform sample contains **200 previously unreviewed originals**, drawn from
2,248 eligible pairs after excluding 77 previously reviewed originals from the
2,325-pair balanced dataset. Agent judgments: **188 acceptable / seven erroneous /
five uncertain sources**; all 200 injected edits judged valid. The six-pair
family supplement has five acceptable sources and one source error; all six
injected edits were judged valid. Do not combine the supplement with the
uniform sample to estimate precision.

The source-specific uniform counts are government **148/160 acceptable** (seven
errors, five uncertain) and EUR-Lex **40/40 acceptable**. The EUR-Lex sample is
small; this does not establish perfect source quality. The 94.0% usable rate
also does not establish an improvement over the preceding 93.5% sample.
No native-speaker validation or factual verification is claimed.

Errors include duplicated `dit`, `zo vaak dan`, missing `n` in `kunnen zij`,
incorrect spacing in `achteruit gegaan`, and `menig ... studie` where the
non-person de-word requires `menige`. The latter was checked against
[Taaladvies](https://taaladvies.net/menig-of-menige-politicus/). The supplemental
source switches `Onze infrastructuur` from object to an omitted subject;
[Onze Taal's coordination rules](https://onzetaal.nl/taalloket/samentrekking-algemeen)
helped adjudicate it. These are source defects, not failed injected errors.

Inspection of the cached **full LanguageTool responses** for all 13 flagged
sources found **no diagnostics at all**. They were not discarded merely by our
severity/category configuration. See `flagged-source-checker-diagnostics.json`.
Enabling additional existing diagnostic categories would not fix these cases.

All 13 flagged originals are recorded in
`config/dutch_validation_review_exclusions.json` with per-sentence evidence.
The final curated subset is created by rerunning balancing over the raw export
with those exclusions:

```sh
.venv/bin/python -m scripts.balance_pairs la_output/dutch_validation_raw la_output/dutch_validation_curated --config config/dutch_validation_balance.json --exclusions config/dutch_validation_review_exclusions.json
```

The 94.0% audit belongs to the frozen **pre-removal** balanced dataset's previously
unreviewed population. It is not a fresh independent measurement of the final
2,312-pair subset. Removing known defects does not certify the remaining data.

## Verification and scale implications

All dataset artifact/task/offset/split checks pass. Independently recovered all
6,474 raw originals from the pinned Parquets, verified all 3,446 spelling edits
against OpenTaal, checked every curated pair is unchanged from its raw parent,
verified the 13 review exclusions are absent, and checked both caps in every
split. The full test suite passes **102 tests**. The final curated manifest SHA256
is `1afbe0a0765c7ff197d2098bb741e9e5748fc6523addc0a143cca7875faed80b`.

Evidence, frozen judgments, build logs and verification receipts are under
`wiki/artifacts/dutch-validation-assessment/`; `curated-summary.json` is the final
machine-readable receipt.

At the measured per-source yields, the selected corpus with the 12-paragraph
cap projects to approximately **48k raw pairs**, or **16k balanced pairs** under
the experimental mixture. This source-stratified extrapolation ignores further
whole-corpus duplicate losses, heterogeneous documents and split rounding; it
is not a guaranteed quota. See `capacity-projection.json`. Higher paragraph
caps require their own quality/yield measurement.

**Conclusion:** the scalable pipeline and explicit balancing work; productive
coverage now realizes substantial verb/adjective/noun variety. Source quality
remains around 94% in agent inspection, and d/dt and relative-pronoun coverage
are still only one example each. The outputs are usable as disclosed synthetic
training candidates, but not a clean linguistic benchmark or evidence of
English-sized production capacity. Deeper sampling of the selected sources,
especially EUR-Lex, is a concrete next volume experiment; a larger independent
source audit and targeted rare-family collection remain necessary for stronger
quality/coverage claims. No Dutch publication or full-corpus build occurred.


[^previous]: Previous frozen data, 200-pair agent audit, export regression and checkpoint/resume equivalence.
[^sources]: Pinned downloaded Parquets, annotation filters and actual paragraph eligibility counts; no extrapolation from dataset-card record counts.

# Superseding production settings

The user subsequently requested no paragraph cap and priority-based article
reduction instead of percentage downsampling. Use `nl_scale` for new production
runs; see [uncapped Dutch generation](dutch-uncapped.md). The capped profiles,
artifacts and measurements on this page remain frozen historical evidence.
