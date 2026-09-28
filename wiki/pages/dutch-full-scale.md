---
type: Implementation
title: Full-source Dutch DynaWord production build
description: Checkpointed full-corpus generation with uncapped paragraphs, lower article priority, and identical parallel local screening.
status: draft
generated: {by: codex/gpt-6, at: 2026-09-22T04:44:48.847589+00:00}
sources:
  - id: configuration
    resource: ./dutch-uncapped.md
    title: User-approved uncapped Dutch configuration
  - id: pilot
    resource: ./dutch-larger-validation.md
    title: Prior validation and known source-quality limits
---
# Scope and frozen inputs

The user explicitly requested a full-scale build following the uncapped probe.
Run **all 10,273 selected source documents** (2,969 excellent government and 7,304
modern EUR-Lex) in 103 deterministic batches. No document limit, paragraph cap,
family-percentage downsampling or fixed pair quota. One injected error per pair;
other grammar precedes articles, with spelling fallback. All 46 audited source
exclusions and original checker requirements remain active.[^configuration]

```sh
.venv/bin/python -m dala.pair_pipeline --profile nl_scale --max-errors 1 --offline --output-dir la_output/dutch_dynaword_uncapped
```

Checkpoints: `la_output/dutch_scale_checkpoints/`. Live log:
`/tmp/dutch_full_scale.log`. Final output is created atomically after all batches
and dataset validation. No Dutch Hugging Face upload is requested.

# Screening throughput and equivalence

The initial single-checker attempt was interrupted before export to address its
approximately 4–5 uncached candidates/second throughput. Its partial checkpoints
are archived at `la_output/dutch_scale_single_checker_attempt/`; its log is in
`wiki/artifacts/dutch-full-scale/single-checker-attempt.log`. Completed HTTP
responses remain available in the versioned SQLite cache; partial checkpoint
files are not treated as completed data.

Added optional identical local LanguageTool pools. Default operation remains one
server. Each instance has its own log/config, every endpoint must be local, and
all servers must report exactly the same software identity. Cache keys remain
software/language/text based. Thread-local HTTP sessions distribute cache misses
across instances; diagnostic classification and screening logic are unchanged.

The test suite passes **104 tests**, including rejection of remote pool members,
rejection of software-version mismatches and reuse of single-server cache entries
by a pool. On 200 real Dutch candidates, a 16-instance cold-cache check produced
**exactly the same evidence and rejection reasons** as the single-checker cached
baseline: 131 accepted, 69 rejected. Measured throughput was 66.2 candidates/s.
Receipt: `wiki/artifacts/dutch-full-scale/checker-pool-equivalence.json`; benchmark
source and test log are alongside it.

The production host exposes 384 CPU cores. Production uses **128 identical local
instances**, two checker threads and a 2 GiB heap ceiling each, with 128 client
workers, as explicitly requested by the user. The interrupted 32-instance
attempt is archived in `la_output/dutch_scale_32_checker_attempt/`. Only operational checker settings changed from the uncapped profile;
linguistic/source settings did not. The run gets a fresh checkpoint identity.
Configuration and `dala/` code remain frozen while the build is active.

# Completion checks

After construction: verify artifacts/task views, recover all originals from the
pinned source documents, check spelling substitutions against OpenTaal, measure
realized families/rules/substitutions/source contributions, and draw a fresh
seeded audit excluding originals in prior review samples. Report agent linguistic
inspection separately from automatic checks; retain the pre-review output and
judgments. Earlier source-noise estimates do not certify this larger output.[^pilot]

# Completed build and independent assessment

The full run completed on 2026-09-22: **188,182 pairs**, **376,364 rows per task**,
from 9,537 contributing documents. All 10,273 selected documents and 733,575
eligible paragraphs were processed; 1,818,573 sentences yielded 378,653 screened
candidates. Train/validation/test contain 151,067 / 19,552 / 17,563 pairs.
Government contributes 81,145 pairs and EUR-Lex 107,037. Maximum contribution
from one document is 564 pairs. No paragraph or percentage caps were applied.

Processing from checkpoint initialization to the last screened batch took
30m54s, excluding checker startup, final export and subsequent assessment.
The last ten batches averaged 221.5 candidates/s. Logs and timings are retained
in `wiki/artifacts/dutch-full-scale/{build.log,build-timing.json}`.

Independent verification passed for artifact checksums, all task views and
balanced clean/corrupt labels, edit reconstruction/correction round trips,
document split isolation, unique exported texts, and source provenance.
All **188,182 originals** were recovered by offsets from the pinned source
snapshots; all **100,269 spelling edits** passed the independent OpenTaal
membership/nonmembership test. All 46 prior source exclusions are absent, and
all 342 pairs from the earlier 20-document single-checker probe match the
128-instance production output exactly, including checker evidence. Receipts
are in `additional-checks.json`; the frozen manifest hash and main checks are in
`wiki/artifacts/dutch-full-assessment/summary.json`.

## Amount and diversity

**64 of 71 configured rules** fired, with **19,892 distinct case-folded
original/replacement strings**. This count includes productive spelling and is
not a count of independently attested learner-error patterns.

| Error family | Pairs | Share |
| --- | ---: | ---: |
| spelling | 100,269 | 53.28% |
| article_gender | 71,156 | 37.81% |
| article_number | 6,573 | 3.49% |
| adjective_inflection | 6,165 | 3.28% |
| demonstrative_agreement | 2,293 | 1.22% |
| subject_verb_agreement | 1,126 | 0.60% |
| possessive_agreement | 567 | 0.30% |
| relative_pronoun_agreement | 24 | 0.01% |
| verb_dt | 9 | 0.00% |

Article gender/number changes together are 41.3%; spelling is 53.3%.
The single most common substitution, `de → het`, occurs 58,051 times (30.8%).
Productive spelling operators supply 83,650 pairs; fixed spelling mappings
supply 16,619. Nine d/dt and 24 relative-pronoun examples remain too few for
substantial family-specific evaluation. Corpus size therefore does not imply
uniform grammar coverage. This release candidate is about 39% of the prior
478,930-pair English release, with roughly 376k rather than 958k rows per task.

## Fresh quality audit

Seed `dala-dutch-full-audit-20260922-v1` selected 200 uniform pairs from 187,807
previously unreviewed originals; 375 originals appearing in earlier audit samples
were excluded before sampling. All originals and recorded substitutions were
inspected, including full grammar-corrupted sentences. No factual validation of
claims was attempted. This is **agent inspection, not native-speaker gold**.

* Sources: **190 acceptable (95%), six erroneous (3%), four uncertain (2%)**.
* Injected edits: **200/200 valid**. This is a sample result, not a guarantee.
* Government: 81/87 acceptable, three erroneous, three uncertain.
* EUR-Lex: 109/113 acceptable, three erroneous, one uncertain.
* Separate coverage supplement: **22/22 acceptable sources and valid edits**;
  do not pool these purposively selected examples into the uniform estimate.

Errors include plural antidepressiva with singular werd, an infinitival list
item, a heading joined to a sentence, an ill-formed complement after beweerden,
a missing determiner, and allen referring to companies. The latter judgment
was checked against [Team Taaladvies](https://www.vlaanderen.be/team-taaladvies/taaladviezen/allen-alle),
which distinguishes references to persons from other entities. Uncertain cases
concern a possible institutional-name hyphen omission, translation/word-choice
problems, punctuation, and parenthetical adjective variants.

The 95% strict source acceptance is similar to the earlier capped validation's
94% under a different sample and source mixture; it does not establish a
statistically meaningful improvement. Source cleanliness remains the main
observed limitation. The modest sample also cannot certify rare-error precision.

The frozen full output remains `la_output/dutch_dynaword_uncapped/` with manifest
SHA256 `96f9606e0b5e9a0ad0ce183b1f4a2303bec277a12710c10480b1c2e702e81e76`.
Per-row judgments, the separate supplement, review summary and ten flagged
original hashes are preserved under `wiki/artifacts/dutch-full-assessment/`.
These flags **have not been removed from the frozen output**; exclude them in a
subsequent release preparation, without treating that removal as an independent
re-audit of the remaining corpus. No Dutch dataset has been uploaded.


[^configuration]: The user removed the paragraph cap and replaced percentage balancing with priority ordering.
[^pilot]: Prior 200-pair fresh audit found 188 usable sources and all injected edits valid; two error families remained extremely sparse.
