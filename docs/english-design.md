# English DaLA: empirical grammar and spelling design

Updated 2026-09-21 following the project owner's clarification. The implemented design and current results are maintained in the
[OKF wiki](../wiki/index.md); the EWT generator remains an engineering pilot. This document supersedes the earlier
proposal to restrict evidence to single-error sentences or to exclude spelling.

## Scope

Include grammatical errors and spelling mistakes. Exclude simplification,
paraphrasing, stylistic preference and style transfer. A correction dataset's
"before" text is not necessarily unacceptable: annotation labels and edit
frequency alone cannot establish that reversing an edit will introduce an error.

Keep the two original DaLA principles: start from real, clean text, and introduce
errors guided by observed mistakes. Use correction corpora to establish the error
patterns and curated Common Pile text to supply independent clean sentences.

## The TV2R reference and DFM11

Inspected live dataset metadata and the first 100 training rows per instruction
variant on 2026-09-21:

| Dataset | Revision | Shape |
| --- | --- | --- |
| `giannor/dala_tv2r` | `b2deeb25200996ccedf61858df9d4620ecb3a039` | `text`, `corruption_type`, `label` |
| `giannor/dala_tv2r_it` | `9976093cd4a6d6ee126d6389a267019e361a841c` | `direction`, nested `samples.content`, `samples.corruption_type`, `samples.response`; yes/no judgment in Danish |
| `giannor/gec_dala_tv2r_it` | `308570cc643b1d4d2ab9ba2771a0093764838003` | Same instruction structure plus affected tokens; erroneous input → corrected text, clean input → unchanged text |

`/work/mimir/HRM-Text/data_io/prefix_config_dfm11.yaml` includes both instruction
variants at repeat 1. Its converter is
`scripts/convert_dfm8_giannor_tv2r.py`: user = direction + content, assistant =
response, with corruption metadata retained separately. The HRM-Text knowledge
page `wiki/pages/dfm8-plan/danish-linguistic-acceptability-and-gec-data.md` records
984,126 acceptability rows and 930,565 correction rows, derived from 492,063 unique
clean TV2R news sentences. These totals are repository records, not newly
recomputed full-Hub counts.

The samples contain `corrupt_spelling` in both instruction variants. The Danish
implementation already has an explicit correct-word → common-misspelling
lexicon. English should support the same class of evidence-based spelling rule.
A useful reference structure does not constitute an independent precision audit
of every TV2R row.

English should produce both acceptability and correction views, with clean
identity examples in correction data to teach preservation. Prompts should
explicitly mention **grammar and spelling**. Retain a canonical paired record
with source dataset/revision, document ID/URL, sentence offsets, original text,
corrupted text, and an `edits` list containing rule IDs, spans and evidence IDs.
Keep the diagnostic metadata out of the model prompt. An edits list supports
multiple errors without overloading the existing scalar corruption label.

## Evidence extraction now implemented

`dala/error_evidence.py` reads standard M2 files, extracts all non-identity
annotated correction patterns, and reports support by file. It:

- Includes edits from sentences containing multiple errors.
- Deduplicates support by tokenized source sentence, including across files and
  annotators. Support is not a count of independent writers/documents.
- Records multi-error and touching/overlapping-edit support. Non-overlapping
  edits can still interact through agreement, reference, or interpretation.
- Preserves case, exact token spans, file checksums and example locations.
- Separately identifies single-word `R:SPELL` candidates.
- Labels every result `unreviewed_evidence_not_a_rule`. No mined pattern is
  automatically added to the corruption generator. Raw inventory includes all
  M2 categories; lexical/style-like categories are not thereby admitted.

Initial run on the [BEA-2019 W&I+LOCNESS v2.1 release](https://www.cl.cam.ac.uk/research/nl/bea2019st/):

| Partition used | Sentences actually parsed | Patterns with support ≥3 | Single-word spelling candidates among these |
| --- | ---: | ---: | ---: |
| W&I A/B/C training, separate files | 34,308 | 2,393 | 29 |
| LOCNESS native development | 988 | 28 | 1 |

The native development partition is used for exploratory error evidence, so it
must not later be presented as untouched evaluation data for these rules. No
BEA test corrections were mined. Reports and downloaded evidence remain in
ignored local `la_output/evidence/`. The original research corpus text is not
being republished as part of this branch.

Examples of observed spelling corrections, with distinct-sentence support:

| Observed error → correction | Support | Sentences also containing other retained edits |
| --- | ---: | ---: |
| confortable → comfortable | 5 | 5 |
| becasue → because | 4 | 3 |
| polution → pollution | 4 | 3 |
| technolgy → technology | 4 | 4 |
| goverment → government | 3 | 3 |

These are measurements in this release, not population-wide frequency claims.
Some automatically typed spelling candidates are debatable lexical substitutions
or names, which is another reason to review candidates before enabling them.
ERRANT types are automatic even when the underlying corrections are human.

Reproduce after obtaining the M2 files from BEA:

```bash
python -m dala.error_evidence --output la_output/evidence/wi_train_patterns.json \
  la_output/evidence/A.train.gold.bea19.m2 \
  la_output/evidence/B.train.gold.bea19.m2 \
  la_output/evidence/C.train.gold.bea19.m2
```

## Turning evidence into safe corruptions

1. Review the underlying correction contexts. Separate native and learner
   evidence; do not conflate repeated sentences with independent authors.
2. For grammar, specify the positive eligibility conditions **and** grammatical
   counterexamples. Reversing an observed *in → on* correction, for instance,
   is not a license to swap those prepositions globally.
3. For spelling, begin with reviewed, exact word → attested misspelling entries.
   Check recognized spelling variants, dialect forms, names, technical words,
   quotations and foreign-language spans. Dictionary absence alone is not proof
   of a misspelling. Do not generalize a doubled-letter error into permission to
   delete any doubled letter.
4. Treat real-word spelling confusions separately: the replacement can itself
   be a valid English word, so syntax/meaning must rule out the alternative
   reading in the new sentence. When that cannot be established, abstain.
5. Test one error at a time during rule calibration to attribute failures.
   This is an evaluation technique, not a restriction on evidence mining or
   final dataset size/error count. Later combine validated edits only when
   their effects do not cancel or create a different valid interpretation;
   validate the combined result too.
6. Preserve source text exactly apart from the registered edits. Include clean
   examples in both output tasks. Split by original document, keeping all
   variants and task views together, and check near duplicates across splits.

## Clean source selection

Prioritize candidates within Common Pile's news, Public Domain Review, Pressbooks
and LibreTexts collections. Initial first-three-row spot-checks found contemporary
prose and usable provenance fields, but also near-empty news pages, table-heavy
Pressbooks pages, and mathematics exercises. This is a shortlist for curation,
not an assertion of collection-wide cleanliness.

Select publication/book and paragraph types explicitly; exclude exercises,
quoted historical passages, deliberately incorrect examples, tables, navigation,
references, fragments and extraction artifacts. For textbooks, retain explanatory
prose from vetted books. Preserve the article/book source ID and URL. Review
original sentence acceptability as well as the introduced error. Automatic
parsing of Common Pile will replace EWT's gold annotations, so parser uncertainty
must be reflected in rule eligibility and validation.

## Implementation status

Common Pile loading and curation, guarded grammar and spelling mappings,
independent multi-error generation, local checker screening, and TV2R-compatible
instruction exports are implemented. The pipeline now consumes language input packs, preserves Danish behavior under
frozen-reference equivalence tests, and uses productive English grammar rules.
See the [assessment](../wiki/pages/multilingual-assessment.md) and
[runbook](../wiki/pages/dataset-runbook.md)
and [measured results](../wiki/pages/build-results.md). Human precision review is
still outstanding; mined patterns are never automatically enabled.
