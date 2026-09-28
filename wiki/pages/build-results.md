---
type: Reference
title: Build results and limitations
description: Measured English Common Pile build and verification, distinct from linguistic precision.
status: stable
generated:
  by: codex/gpt-6
  at: '2026-09-21T15:34:50+00:00'
---
# Initial exact-substitution build

This is the preserved first-build baseline. The [current productive build and assessment](/pages/multilingual-assessment.md) supersede it for new English generation.

Local output: `la_output/english_common_pile/` on branch `multilingual`.
Command: `python -m dala.build_english --offline --output-dir la_output/english_common_pile`.
Seed 4242, maximum two independent errors, spaCy en_core_web_md 3.8.0,
LanguageTool 6.6 with local Temurin 17.0.16. No Hub publication was performed.

The source pool contains 1,939 documents, 38,107 selected paragraphs and 81,749
parsed sentences. The completed export contains 14,103 pairs from
1,642 documents. Each task has 28,206 rows,
half clean controls and half corrupted inputs. Raw and instruction acceptability
are two representations of the same task, not additional independent examples.

| Split | Pairs | Acceptability rows | Correction rows |
| --- | ---: | ---: | ---: |
| train | 11,212 | 22,424 | 22,424 |
| validation | 1,367 | 2,734 | 2,734 |
| test | 1,524 | 3,048 | 3,048 |

# Error coverage

Counts below are edits, so a two-error pair contributes twice.

| Family | Edits |
| --- | ---: |
| demonstrative_number | 2,195 |
| do_support_form | 123 |
| modal_verb_form | 1,181 |
| noun_number | 92 |
| perfect_participle | 78 |
| spelling | 9,864 |
| subject_verb_agreement | 1,880 |

There are 1,310 two-error pairs. The common
`that → taht` mapping contributes 6,232 edits. The
inventory and parser/checker selection are narrow and skewed; aggregate scores
must not be interpreted as representative English error coverage or natural
error frequencies. Some families have small held-out counts.

# Verification

37 tests pass, covering guarded grammar, attested spelling, multi-edit
independence, evidence counting, source integrity, Unicode offsets, review
acceptance and export checks. The seven OKF concepts and internal links validate.
All exported source spans were compared against the pinned cached documents.
Artifact hashes, exact edits and inverse correction, labels, task-view alignment,
source provenance and document split isolation pass independent artifact checks.
The source-code fingerprints match the implementation used for this run.

The 92-rule evidence file rebuilds byte-identically from the recorded A/B/C
training M2 inputs. A parallel 30-document run reproduces all 225 retained pairs
and checker evidence from the earlier sequential run after audited exclusions.
This is a bounded reproducibility check, not a second full corpus rebuild.
Both instruction schemas load in Hugging Face and pass TV2R content conversion;
see [integration limitations](/pages/tv2r-and-history.md).

# Quality findings

The first unchecked 30-document smoke build had 301 pairs; an existing source
error motivated independent checker screening. The first checked smoke had 227
pairs. Agent spot-checks of 25 smoke and 21 full-build examples found three
malformed or suspect originals missed by the checker; exact exclusions and
reasons are recorded in `config/english_source_exclusions.json`. The final smoke
has 225 pairs. These were development inspections, not a random blinded study
and not a human precision estimate.

In the final full run, the checker rejected 1,031
candidates for source diagnostics and 3,710 for
missing edit diagnostics. A further 119 exact text
collisions and 89 near duplicates were excluded.
Three known source issues were excluded before checker screening.
The final status is **checker_screened**, not human-validated. Residual source
errors, parser errors, checker blind spots and related-article leakage remain
possible. Use the explicit [review workflow](/pages/dataset-runbook.md) before
claiming human-validated benchmark precision.

# Receipts

- [Complete build manifest](../artifacts/english-build-manifest.json)
- [Final verification receipt](../artifacts/english-build-validation.json)
- Local dataset manifest SHA-256: `db8295d5cf93aa23c9806d93132756445c4002e7d16c917a20a47b17818f1b08`.

The generated sentence data and runtime caches stay under ignored `la_output/`;
implementation, evidence registry, knowledge and lightweight receipts are kept
in the repository working tree. Changes have not been committed.
