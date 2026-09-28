---
type: Implementation
title: Uncapped Dutch generation with article errors at lower priority
description: Production configuration without paragraph or family-percentage caps, retaining source checks and reviewed exclusions.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-22T03:59:33.284985+00:00}
sources:
  - id: previous
    resource: ./dutch-larger-validation.md
    title: Frozen capped validation and balancing experiment
---
# Requested change

The user requested removal of the paragraph cap and less aggressive balancing
through article-rule priority alone. The active production profile is now
`config/languages/nl_scale.json`. Historical profiles and exports remain frozen
so their checksums and audit results remain reproducible.[^previous]

# Configuration

- Remove `max_paragraphs_per_document`: process every paragraph that passes the
  existing paragraph eligibility checks. This is not a relaxation of the source
  quality, paragraph-format or sentence-length checks.
- Priority: d/dt, subject–verb agreement, relative-pronoun agreement, possessive
  agreement, adjective inflection, demonstrative agreement, article number,
  article gender, then spelling.
- Other grammatical errors therefore precede article errors. Article number
  precedes the much more frequent article gender. Spelling remains the fallback;
  placing it before articles would suppress articles in almost every sentence.
- No article/spelling percentage caps and no post-build balancing step. Retain
  screened pairs subject to the existing source exclusions and duplicate checks.
- Merge the 33 prior source exclusions with the 13 most recent review exclusions
  into `config/dutch_scale_source_exclusions.json` (46 unique originals).
- Use a fresh checkpoint directory, `la_output/dutch_scale_checkpoints`, because
  changed inputs invalidate the previous run's candidate/checker receipts.

Full-source command, with no document or paragraph limit:

```sh
.venv/bin/python -m dala.pair_pipeline --profile nl_scale --max-errors 1 --offline --output-dir la_output/dutch_dynaword_uncapped
```

This command exports the screened pairs directly. Do not run the old
`scripts.balance_pairs` percentage-cap experiment on this production output.
The full-source build has not been started as part of this configuration change.

# Verification

The bounded `nl_uncapped_probe` profile has the same linguistic/source settings
and a separate checkpoint directory. It processes **all eligible paragraphs**
from the first 20 deterministic source documents:

```sh
.venv/bin/python -m dala.pair_pipeline --profile nl_uncapped_probe --max-documents 20 --max-errors 1 --offline --output-dir la_output/dutch_uncapped_probe
```

Targeted checks confirm all 30 eligible synthetic paragraphs are returned rather
than the previous 12, with exact original offsets. Actual parsed examples verify
that demonstrative agreement and article-number changes now beat article-gender
changes; article edits remain available ahead of spelling, and subject agreement
retains its priority. All 46 audited exclusions are loaded. These checks are
recorded in `wiki/artifacts/dutch-uncapped-checks/profile-checks.json`.

The completed probe processed **1,019 paragraphs and 2,370 sentences**, yielding
557 checker candidates and **342 retained pairs** (684 rows per task). The same
20 documents previously supplied 240 paragraphs and 136 retained pairs under
the capped configuration (with current audit exclusions applied for comparison).
This is a 2.51× pair-yield increase on this bounded sample; it measures the joint
paragraph/priority change, not the isolated effect of either setting.

Article errors account for 126/342 pairs (36.8%), versus 65/136 (47.8%) in the
capped comparison. No percentage caps were used. Spelling contributes 197 pairs;
other realized grammar includes 11 adjective, five demonstrative, one subject
agreement, one possessive and one d/dt example. These are small-sample counts,
not a full-corpus composition or linguistic-precision estimate.

Full artifact/task/split validation passes. All 342 originals round-trip to the
pinned source texts, all 197 spelling substitutions pass independent OpenTaal
checks, and all 46 reviewed exclusions remain active. The production and probe
profiles are identical except for names and operational checkpoint settings.
Evidence: `wiki/artifacts/dutch-uncapped-checks/build-comparison.json` and `build.log`.
The probe manifest SHA256 is
`7ae939ba911bafc8c0c864b13e83f78ce7cfbcfe2f1c04e227570fd13e089b31`.
No fresh linguistic precision estimate is implied by this configuration or smoke test.

[^previous]: The 1,000-document capped experiment and its 25% article / 60% spelling subset are historical measurements, not requirements of DaLA.

# Full-source run

The user subsequently authorized full-scale construction. See
[the full-source run record](dutch-full-scale.md). Operational checker pooling
was added after the probe; the paragraph, priority and source settings above
remain unchanged.
