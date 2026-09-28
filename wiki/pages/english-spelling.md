---
type: Reference
title: Expanded English spelling inventory
description: Published common-error lists, lexical screening, and the distinction between spelling and random corruption.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-21T17:28:20+00:00}
sources:
  - id: cambridge
    resource: https://dictionary.cambridge.org/grammar/british-grammar/spelling-top-50-spelling-mistakes-in-english
    title: Cambridge English Grammar Today top 50 spelling mistakes
  - id: mw
    resource: https://www.merriam-webster.com/grammar/commonly-misspelled-words
    title: Merriam-Webster commonly misspelled words
  - id: idp
    resource: https://ielts.idp.com/indonesia/about/news-and-articles/article-ielts-writing-tips-for-mastering-spelling/en-gb
    title: IELTS IDP spelling advice
---
# Inventory and evidence

This page records the lexical expansion. The subsequent
[generator revision](/pages/english-generators.md) adds productive spelling and
guarded token fallback; the earlier absence described below is historical.

The active `config/english_productive_rules.json` expands from 13 to **72 spelling
mappings covering 66 distinct correct word forms**. The original W&I evidence is
preserved. Published additions retain exact listed incorrect spellings and source
IDs; they do not infer new errors by analogy. Cambridge describes learner exam
errors, while Merriam-Webster and IELTS IDP publish editorial spelling advice.
These are not interchangeable population frequency estimates.[^cambridge][^mw][^idp]

For new list-only entries, `distinct_sentence_support` is null, not an invented
corpus count. Multiple sources for an identical pair are merged. The original
92-entry `config/english_rules.json` remains the historical corpus-only inventory.
The active inventory is the reproducible, curated input; rebuilding the historical
inventory does not recreate this expansion.

Agent screening excluded real-word outputs such as `customers → costumers` and
`whether → wether`, and avoided ambiguous/archaic forms, regional alternatives and
an inconsistent Cambridge example. `business → bussines` was also excluded because
the checker failed to flag its exact token. The published pair is retained in the
rulebook's exclusion metadata rather than silently relabelled as a valid spelling.

Run `python -m scripts.screen_english_spelling` to screen each active mapping using
local LanguageTool 6.6, in both en-US and en-GB. The correct word must escape the
misspelling diagnostic and the incorrect word must receive an exact-span diagnostic.
See [screening receipt](/artifacts/english-spelling-screening.json). This is a
mechanical dictionary screen, not human validation. Full dataset generation also
checks original sentences and each injected error in context, and avoids named
entities and proper nouns.

# Spelling rules versus random corruption

English currently has lexical spelling rules plus productive **grammatical**
inflection rules. It does not yet generalize letter doubling, silent-e deletion,
or ie/ei swaps to arbitrary words. The expanded inventory contains examples of
several spelling mechanisms, but those mechanisms are not enabled as generators.

Danish has both lexical spellings and productive orthographic/morphological rules,
such as r-related endings and genitives. Its final `token_fallback` rule randomly
chooses word deletion or neighbouring-word swapping when no earlier targeted
rule succeeds. Deletion excludes several POS categories and adjacent noun groups;
swapping uses POS-based restrictions and case repair. These are heuristics, not a
proof that the resulting sentence is ungrammatical. These token operations are
not random character typos. English does not currently enable this fallback.

[^cambridge]: Publisher list retrieved through a search-index rendering of its localized page; canonical direct access returned HTTP 403. Only spelling facts were transcribed, not example sentences.
[^mw]: Publisher page inspected directly; explicit correct and incorrect word forms.
[^idp]: Publisher spelling advice retrieved through search-index rendering; canonical direct access returned HTTP 403.

# Rebuilt dataset

`la_output/english_spelling_expanded/` uses the same pinned source pool, seed and
checker settings as `la_output/english_productive/`. Run:
`python -m dala.multilingual --language en --offline --output-dir NEW_DIRECTORY`.

Pairs increased from 18,958 to **21,090** (+11.2%), yielding **42,180 rows per task**.
There are 16,696 training, 2,097 validation and 2,297 test pairs. The output realizes
**67 spelling mappings across 61 words**, compared with 13 mappings across 11 words
previously. Five inventory entries did not appear in retained outputs. Across all
families, distinct surface substitutions increased from 2,038 to 2,088.

The dominant `that → taht` mapping fell from 26.0% to **22.0% of all edits** and
from 62.9% to **45.3% of spelling edits**. Spelling effective substitution diversity
(2 to the power of Shannon entropy) rose from 3.90 to 11.97. This improves coverage
but does not solve concentration; no frequency cap or balancing was added.

All 72 active mappings passed both dialect screens. Dataset artifact checks,
edit reconstruction, correction roundtrips, task views, label balance and document
split isolation passed. An agent spot-check of 12 pairs containing new spelling
mappings found the intended errors valid and originals acceptable; this small
inspection is not human review or a corpus precision estimate. All 55 code tests
passed; Danish code and inputs were unchanged in this expansion.

Receipts: [comparison](/artifacts/english-spelling-comparison.json),
[validation](/artifacts/english-spelling-validation.json),
[agent spot-check](/artifacts/english-spelling-agent-review.json).
