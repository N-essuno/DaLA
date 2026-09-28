---
type: Evaluation
title: Productive multilingual pipeline assessment
description: Danish equivalence and measured English amount, diversity and agent-assessed quality.
status: stable
generated: {by: codex/gpt-6, at: '2026-09-21T17:07:20+00:00'}
---
# Delivered inputs and outputs

This assessment records the first productive build, before the subsequent
[spelling expansion](/pages/english-spelling.md). Its counts and receipts remain
historical comparisons; the active spelling inventory now has 72 entries.

Danish input: `config/languages/da.json`. English input:
`config/languages/en.json`, referencing productive rules, empirical evidence,
source snapshots and exclusions. Common API: `dala.pipeline.build`; CLI:
`python -m dala.multilingual --profile PATH`. See [architecture](/pages/pipeline-architecture.md).

Final English output: `la_output/english_productive/`. Baseline output remains
`la_output/english_common_pile/`. Both use the same 1,939-document pool and
81,749 parsed sentences. The final run uses seed 4242, maximum two independent
errors and the same parser/checker versions. Only the current pipeline and
productive rules should be used for new builds; the baseline is a preserved
comparison artifact.

# Danish equivalence

The refactor moved active Danish rules and source/split settings to a declarative
pack. The frozen reference and configured pipeline matched on 77,014 per-rule
comparisons over 5,501 real DDT sentences, plus four full builds (two split modes,
two seeds). Rows, CSV bytes and final RNG states match. All active rules had
positive cases. This establishes compatibility for the tested inputs and versions,
not correctness of the historical linguistic rules. [Receipt](../artifacts/danish-equivalence.json).

# Amount

Pairs increased from **14,103 to 18,958** (34.4%).
The final dataset has **37,916 rows per task**, including unchanged
clean controls, from 1,659 source documents.

| Split | Pairs | Rows per task |
| --- | ---: | ---: |
| train | 15,021 | 30,042 |
| validation | 1,866 | 3,732 |
| test | 2,071 | 4,142 |

There are 2,132 two-error pairs. 5,764
originals are newly retained, while 909 baseline
originals are no longer retained. The result is not a strict superset: changed
priority choices, additional ambiguity guards and checker acceptance change
selection. The source pool was deliberately held fixed to measure generalization;
this remains much smaller than the TV2R source corpus.

# Diversity

Distinct realized, case-insensitive surface substitutions increased from
**86 to 2,038**.
Distinct original word forms increased from 66
to 1,584. These are measured output counts,
not rulebook sizes. The current pack has 13 lexical spelling entries, four lexical
demonstrative entries, and five productive grammar rules. Unused lexical entries
do not inflate the reported realized diversity.

| Family | Earlier edits | Current edits | Distinct substitutions, earlier → current |
| --- | ---: | ---: | ---: |
| demonstrative_number | 2,195 | 1,271 | 3 → 3 |
| do_support_form | 123 | 791 | 7 → 302 |
| modal_verb_form | 1,181 | 5,872 | 14 → 967 |
| noun_number | 92 | 298 | 13 → 140 |
| perfect_participle | 78 | 1,329 | 2 → 44 |
| spelling | 9,864 | 8,727 | 13 → 13 |
| subject_verb_agreement | 1,880 | 2,802 | 34 → 575 |

The top substitution remains `that→taht`, but its share of all edits fell from
40.4% to 26.0%.
Entropy increased from 3.65 to
6.73 bits. The equivalent number of equally frequent
substitutions is 12.6 versus
106.1; thus the gain is not only a longer
list of singleton edits. Nevertheless, spelling still has only 13 mappings,
common spellings and auxiliaries dominate, and family counts remain uneven.
There are 147 test edit instances
whose family/surface pair does not occur in training (baseline: 0).
That is lexical novelty, not proof of semantic independence.

# Quality assessment

All 52 tests pass. Independent export validation checks hashes, exact source/edit
reconstruction, inverse correction, task views, label balance and document split
isolation. Every final source span matches its pinned original document. Source,
profile and implementation hashes match the final receipts.

The final default checker requires no relevant original diagnostics and a diagnostic
for every injected edit. The final run rejected 1,319
candidates for source diagnostics and 4,032
for missing edit diagnostics. These counts are screening outcomes, not precision.

A fixed seed selected ten unique pairs per error family, 70 pairs total, from the
completed pre-audit build. Agent inspection accepted all injected edits in those
70 pairs, but identified **two malformed originals and one uncertain original**:
a malformed clause transition, a missing noun after “a common”, and questionable
“where … native to”. All three were added to the auditable source exclusions and
removed by a fresh full build. The remaining 67 inspected pairs are retained.
The complete judgments and source/license attribution are in the
[agent review](../artifacts/english-agent-review.json).

This is a family-stratified, non-blinded **agent** assessment, not independent
human review. The 3/70 source flags are an observed sample count, not an unbiased
corpus-wide error-rate estimate. Removing known cases does not establish precision
on the other sentences. Final status remains **checker_screened**; human precision
is unmeasured. Source cleanliness is still a demonstrated weakness. Broader blinded
linguistic review, especially of clean originals and productive rules, is needed
before presenting this as a validated benchmark.

During development, tests also identified two important guard requirements:
reverse dictionary lookup is incomplete for some valid forms (e.g. lectures),
and changing demonstratives with unchanged singular/plural nouns (fish/series/means)
can preserve grammaticality. Bidirectional dictionary-table validation and explicit
number-ambiguity abstention now cover these cases. Syncretic verb forms, guessed
OOV inflection, collective agreement and interacting grammar edits are excluded.

# Conclusion and limits

The refactor preserves Danish behavior and materially broadens English grammatical
coverage without using arbitrary corruption fallback. The retained corpus is
larger and lexically more diverse on identical sources. It is still constrained
by the small source pool, narrow spelling inventory, uneven error-family coverage,
checker dependence and unmeasured human precision. Expanding sources alone would
not resolve those quality limitations.

# Receipts

- [Amount and diversity metrics](../artifacts/english-comparison.json)
- [Final build manifest](../artifacts/english-productive-manifest.json)
- [Final mechanical validation](../artifacts/english-productive-validation.json)
- Final manifest SHA-256: `aaf99ebe123a54f5d54be00d1e18c7ad1527fcd903232f228f3c8d6d4005f472`.

No Hub publication or Git commit was performed. Generated sentence data and local
runtime caches remain under ignored `la_output/`; code, inputs, knowledge and
lightweight receipts remain in the repository working tree.
