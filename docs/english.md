# English DaLA: initial corruption policy

**Historical pilot:** the [revised empirical design](english-design.md) supersedes
this document as the production plan. The implemented Common Pile pipeline is
documented in the [OKF wiki](../wiki/index.md). Grammar and spelling are both in scope;
EWT remains a pilot source, and single-error mining is not required. The code and
measurements below describe the existing pilot, not a completed production dataset.

This extends DaLA's method: real source sentences, one injected error, balanced
acceptable/unacceptable pairs, and error-level quality evaluation. It does not
replace the source sentences with generated grammar templates. The Danish rules
are unchanged. English output is an **unvalidated candidate dataset**, not a
validated English benchmark.

## Why these errors?

The empirical starting point is the learner-writing error inventory in
[CoNLL-2014 / NUCLE, Table 1](https://aclanthology.org/W14-1701.pdf): subject–verb
agreement (SVA), verb form (Vform), article/determiner (ArtOrDet), and pronoun form
(Pform). [ERRANT](https://aclanthology.org/P17-1074/) provides a complementary
error-analysis taxonomy. These support the broad error families; they do **not**
establish the frequency or precision of our particular substitution rules.
Learner errors are also not necessarily representative of native writers' errors.
An English release should state its intended population and validate the chosen
subtypes against that population's writing.

[BLiMP](https://aclanthology.org/2020.tacl-1.25/) is a secondary reference for
minimal-pair grammatical contrasts, not the empirical source of this error
inventory. Its reported human agreement does not transfer to this implementation.

Examples below are illustrative, not quotations from the error corpora.

| Rule | Example (source → corruption) | Evidence family | Scope and exclusions |
| --- | --- | --- | --- |
| Subject–verb agreement | She works → She work | SVA | Overt adjacent personal-pronoun subject and indicative finite verb; a finite lexicon. Excludes collective nouns, coordinated subjects, subjunctives and marked subordinate constructions. Keeps singular *they* agreement plural. |
| Demonstrative number | these books → this books | ArtOrDet | Gold determiner dependency and matching noun number, adjacent common noun. Excludes coordination, numerical modifiers, and compounds. |
| Modal complement form | will go → will goes | Vform | Full modal preceding a gold base-form verb; curated inflections. Not modal meaning substitutions. |
| Do-support form | did go → did went | Vform | Full auxiliary *do/does/did* preceding a base verb; excludes lexical *do*. |
| Perfect participle | has eaten → has ate | Vform | Gold auxiliary *have* with VBN and a curated distinct past form. Excludes regular past/participle syncretism and lexical *have*. |
| Subject-pronoun case | She works → Her works | Pform | Bare pronoun in an overt finite subject construction; excludes coordination, fragments, and comparative clauses with a marker. This narrow subtype needs separate empirical validation. |

Auxiliary rules allow intervening adverbs attached to the same verb, such as
*did not go* and *has already eaten*. They exclude intervening clauses and verbs.
Agreement does not change *was* to *were*: embedded irrealis readings can license
both even when indicative morphology is annotated.

Order is fixed rare-first, following Danish DaLA, using initial EWT development
coverage: perfect participle, do-support, modal form, demonstrative number,
agreement, pronoun case. Within a rule, a seeded hash chooses one eligible span.
The rule order is frozen rather than tuned to the test set. This distribution
is not claimed to reproduce natural error frequencies.

## Deliberately deferred

Arbitrary preposition swaps, article deletion, *a/an* chosen from spelling alone,
*some/any* swaps, tense changes, and homophone swaps without syntactic checks can
leave a grammatical sentence or change only its meaning. Reflexives require
binding analysis; *who/whom*, collective agreement, and negative concord involve
register or dialect differences. Generic spelling noise is distinct from the attested spelling errors now included
in the revised design. The pilot does not yet implement the reviewed English
spelling lexicon. These families need separately evaluated English rules.

## Data and implementation

- Source: [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html),
  pinned to `r2.17`. Observe the source treebank's attribution and license when
  redistributing derived data (EWT lists CC BY-SA 4.0).
- Use the existing UD annotation rather than reparsing with a downloaded model.
  CoNLL-U multiword tokens and empty nodes are handled separately. Components
  without an independent surface span cannot be edited.
- Exactly one source character span is replaced. Whitespace, contractions,
  punctuation, and all other source characters are preserved.
- Skip sources with failed surface alignment, `Typo=Yes`, `Foreign=Yes`, or
  `CorrectForm` annotations. These checks cannot detect all existing errors.
- Keep sentences with more than five tokens and 2–5,000 characters. English does
  not apply the Danish-specific `SLUTORD` or POS-diversity filters.
- Preserve official EWT train/dev/test partitions (`dev` is exported as `val`).
  Remove duplicate sources and pair-text collisions, with test then validation
  taking priority. Both members stay together; collisions remove whole pairs.
- Unlike the Danish path, English abstains if no targeted rule applies. There is
  no random deletion/shuffling fallback. Sizes therefore follow coverage rather
  than the Danish fixed sizes. Reports expose this selection bias.
- Candidate CSVs retain `text`, `corruption_type`, `label`. Audit CSVs carry
  sentence/pair IDs, source and changed text, offsets, and blank review fields.
  The review queue samples up to 30 pairs per rule per split, deterministically.

## Running and reviewing

```bash
python -m dala.multilingual --language en --output-dir la_output
# Offline, with en_ewt-ud-{train,dev,test}.conllu in a directory:
python -m dala.multilingual --language en --data-dir /path/to/ewt
python -m unittest discover -s tests -v
```

`dala_en_report.json` records input, abstention, eligible-type and selected-type
counts, removed collisions, and exported pair counts. `dala_en_review.csv` is the
stratified review queue. Full per-split audit files permit exhaustive review.

Review **both** sides: EWT is real web text and includes unmarked misspellings,
fragments, and pre-existing grammatical errors. A correct dependency annotation
does not certify source acceptability. In the initial development inspection,
existing spelling and auxiliary errors survived annotation-based filtering.
Do not infer corruption precision from successful generation, model preference,
or a grammar checker's silence.

Suggested release procedure:

1. English-proficient reviewers independently mark `original_acceptable`,
   `corrupted_unacceptable`, and `single_error`; record ambiguity/register issues
   in `review_notes`. Use the target written-English variety agreed for the study.
2. Reject a pair unless all three judgments hold. Adjudicate disagreements;
   inspect all rare-rule examples and a random sample of common-rule examples.
3. Report counts and precision with confidence intervals separately for each
   corruption. Revise or remove unreliable rules and rerun the review. A useful
   proposed criterion is at least 95% accepted pairs per type, but this is a
   proposed release criterion, not an achieved result or an existing DaLA rule.
4. Review source acceptability for all released pairs; a sampled corruption audit
   alone cannot clean the full set. Freeze code, source revision, selection seed,
   and policy before final evaluation.

The automatic tests check edit mechanics, grammatical guard cases, balanced
labels, deterministic selection, and split isolation. They do not replace
linguistic validation. No human precision result is claimed in this branch.

## Extending to another language

Add a configuration in `dala/languages.py`, implement language-specific candidate
rules with explicit abstention, and wire the new policy into the CLI. Keep the
source language's error evidence and quality analysis alongside its rules.
Do not silently reuse Danish or English substitutions for unsupported languages.

## Initial implementation check (2026-09-21)

On the pinned EWT release with seed 4242, the candidate run yielded:

| Selected rule | Train pairs | Validation pairs | Test pairs |
| --- | ---: | ---: | ---: |
| Perfect participle | 73 | 6 | 10 |
| Do-support form | 252 | 21 | 26 |
| Modal complement form | 502 | 61 | 55 |
| Demonstrative number | 640 | 89 | 87 |
| Subject–verb agreement | 612 | 97 | 105 |
| Subject-pronoun case | 963 | 111 | 123 |
| Total | 3,042 | 385 | 406 |

All 3,833 pairs passed exact single-span reconstruction and label-balance checks;
exported texts were unique and split-disjoint. Seventeen automated tests passed.
These are engineering checks, **not** a measured linguistic precision result.
An assistant spot-check exposed pre-existing source errors and a capitalization
artifact (fixed), and motivated excluding *was → were* ambiguity. No human
review has been completed. Rare-rule sample sizes, source cleanliness, and the
empirical representativeness of the narrow subtypes remain release limitations.
