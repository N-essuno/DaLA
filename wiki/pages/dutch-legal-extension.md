---
type: Implementation
title: Dutch legal-source extension to English release size
description: Investigation and screened extension using Rechtspraak and modern Officiële bekendmakingen.
status: draft
generated: {by: codex/gpt-6, at: 2026-09-22T06:03:00+00:00}
sources:
  - id: rechtspraak
    resource: https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/blob/d0158defd949699532e59dea5978c5542afb0400/data/de_rechtspraak/de_rechtspraak.md
    title: Pinned Rechtspraak source card
  - id: bekendmakingen
    resource: https://huggingface.co/datasets/danish-foundation-models/dutch-dynaword/blob/d0158defd949699532e59dea5978c5542afb0400/data/officiele_bekendmakingen/officiele_bekendmakingen.md
    title: Pinned Officiële bekendmakingen source card
  - id: base
    resource: ./dutch-full-scale.md
    title: Frozen 188,182-pair base and independent audit
---
# Authorization and objective

The user requested investigation of Rechtspraak and Officiële bekendmakingen,
then starting an extension to Danish/English scale. The concrete target is the
English release size: **478,930 total pairs**, not that many additional pairs.
Preserve the frozen base separately. No upload is authorized.

# Investigation

Pinned both data and metadata Parquets at upstream revision
`d0158defd949699532e59dea5978c5542afb0400`. Downloaded source cards, paths,
exploratory counts and deterministic document samples are preserved under
`wiki/artifacts/dutch-extension-investigation/`.

Rechtspraak contains 918,634 records; 532,556 meet excellent quality, complete
integrity, complete/mostly-content, moderate/substantial length, and written-content
annotation filters. Officiële bekendmakingen contains 1,822,093 records, including
historical OCR. The equivalent annotation filter retains 179,629; an exploratory
first-year heuristic retained 97,724, but this is **not** the production selector.
Production requires a `Jaargang 20xx` or `Gepubliceerd op 20xx` header in the first
2,000 characters. This avoids using an arbitrary cited law's year as a publication
date and intentionally omits many documents without reliable modern headers.

Both source cards report CC0. Rechtspraak is anonymized and often contains
placeholders and substituted names. Officiële bekendmakingen includes repeated
navigation, legal templates and wrapped lines. Professional origin and excellent
annotations do not establish sentence correctness.[^rechtspraak][^bekendmakingen]

Many judgments occupy a single line longer than the old 6,000-character paragraph
limit. Added a configurable `curation.max_paragraph_chars`, retaining the old
6,000 default; extension profiles allow 60,000. This is a parser-input bound,
not a relaxation of the 8–32-word / 40–280-character sentence criteria. Text is
never joined or rewritten; offsets remain exact. No paragraph-count cap.
Additional sentence exclusions cover square-bracket placeholders, `Titel`,
joined navigation headings and lower-to-upper case joins. Existing grammar,
OpenTaal and local checker requirements remain active.

# Pilot and production workflow

`nl_extension_pilot` runs 400 deterministic round-robin documents with eight parser
processes, 128 screening workers and 16 local checker instances. This is a yield
and linguistic-quality gate before the production run, not a final precision
measurement. Pilot output: `la_output/dutch_extension_pilot/`.

`nl_extension_scale` is configured for 320,000 new screened pairs, with 128 local
checker instances / 128 screening workers and 16 parser processes. The extra
capacity leaves a reserve for cross-source deduplication and the final split
quotas. It uses the same uncapped paragraphs, lower article priority and no family
percentage balancing. Checkpoints: `la_output/dutch_extension_scale_checkpoints/`.

`scripts.merge_dutch_extension` validates both parent exports, retains unchanged
base pairs except attributed source exclusions, filters exact/near duplicates
across old/new data, and preserves document splits. It targets 383,144 train,
47,893 validation and 47,893 test pairs, failing explicitly if unique supply is
insufficient. It does not duplicate pairs or relax checks to fill a quota.
The merged artifact uses `nl_legal_extended`, all four pinned sources, parent
manifest hashes and a merger checksum. The ten flags from the full-base audit
are in `config/dutch_extension_source_exclusions.json` alongside the earlier 46.

# Pilot results and quality decision

The 400-document pilot (200 per source) produced **7,685 pairs**: 2,298 from
Rechtspraak and 5,387 from official announcements, with 351 contributing documents.
All artifact/task/source-offset and spelling-veto checks passed. A uniform
200-pair agent audit found **180 acceptable sources, 15 erroneous, five uncertain**;
all 200 injected edits were valid. A separate 15-pair family supplement passed.
The first audit's source counts were Rechtspraak 62/69 acceptable and official
announcements 118/131. Evidence: `wiki/artifacts/dutch-extension-pilot-assessment/`.

Added conservative source rejection patterns for spaces before periods, joined
sentence/number boundaries, trailing section markers, attached letter/digit
footnotes, and ambiguous/faulty word segmentation. These intentionally sacrifice
some valid abbreviations, numeric references and uses of tenminste/tenslotte/teveel;
they never rewrite targets or license a corruption. Applying only these filters
and 20 attributed exclusions to the frozen pilot retained **7,543 unchanged pairs**
(2,236 Rechtspraak, 5,307 official announcements). Exact unchanged-subset provenance
and exported task validation passed. Source round trips/spelling checks are
inherited from the fully checked parent, with explicit lineage.

A **fresh stratified sample of 50 per source**, excluding every previously reviewed
original, found:

| Source | Acceptable | Erroneous | Uncertain | Valid injected edits |
| --- | ---: | ---: | ---: | ---: |
| Rechtspraak | 49 | 1 | 0 | 50 |
| Officiële bekendmakingen | 44 | 4 | 2 | 50 |

This is agent inspection, not native-speaker gold. The combined 93/100 is not a
uniform estimate for a differently weighted production population. The samples
are small and drawn from the same 400-document pilot; neither rate certifies the
much larger corpus. Second-review evidence: `wiki/artifacts/dutch-extension-validation/`.
Word-segmentation judgments were checked against [Taaladvies on eenzelfde](https://taaladvies.net/eenzelfde-of-een-zelfde/)
and the linked normative references in the first review summary.

Seven additional originals are excluded from production. Further conservative
patterns reject common joined legal-heading prefixes and the observed forms
`een zelfde`, `er vanuit gegaan`, and `inspraak avonden`. These post-review changes
are not independently re-audited; the final production sample will provide that
check. Final production exclusions are frozen separately in
`config/dutch_extension_production_source_exclusions.json` (**83 unique hashes**),
so all prior input files and measured pilot artifacts remain reproducible.

Because the second sample showed more residual defects in official announcements,
production uses **10 Rechtspraak documents per one official-announcement document**
in deterministic weighted round-robin order. At pilot yields this suggests roughly
81% Rechtspraak / 19% announcements among new pairs, not a guaranteed source quota.
This is source prioritization only; all corruption priorities and no-family-cap
policy remain unchanged. The optional weights preserve the old order when absent
or uniformly one. Exhausting one source continues through the other without
silently dropping documents.

# Active production launch

Started `python -m scripts.run_dutch_extension` on 2026-09-22. It runs, in order:

1. `nl_extension_scale` toward 320,000 screened new pairs with 128 local instances
   and 128 screening workers; checkpoints and a persistent response cache permit
   recovery.
2. `scripts.merge_dutch_extension` toward **478,930 total pairs**. Preserve base
   records and document splits, remove attributed source flags, filter cross-source
   duplicates, and retain parent manifest hashes. No claims that generation has
   completed before the exported artifacts exist.
3. Independent artifact/source/spelling assessment of the merged artifact and a
   fresh 200-pair sample excluding all earlier reviews. **Linguistic review remains
   a manual agent task**; the driver records it as pending, not automatically passed.

Output paths: `la_output/dutch_extension_scale/` (new-source reserve),
`la_output/dutch_dynaword_extended/` (combined target).
Live status: `la_output/dutch_extension_status.json`.
Logs: `/tmp/dutch_extension_scale.log`, `/tmp/dutch_extension_merge.log`,
`/tmp/dutch_extended_assessment.log`; runner errors: `/tmp/dutch_extension_runner.log`.

**108 tests pass**, including long-paragraph offsets, weighted source order and
unchanged default order, base retention, exclusions, duplicate rejection and
explicit capacity shortfall. A real 30-pair merge smoke test passed full artifact
validation, and all 188,182 base records replay through the merger's deduplication
unchanged. Receipts are under `wiki/artifacts/dutch-extension-investigation/`.
Code and active production inputs are frozen after launch. No Dutch upload.


[^rechtspraak]: Pinned source card and actual Parquet/annotation inspection.
[^bekendmakingen]: Pinned card explicitly describes the historical/modern mixture and missing per-document dates.

# Completion and release follow-up

The driver completed generation, merge and all mechanical/source checks. The
combined parent contains 478,930 pairs. The user subsequently requested the fresh
linguistic audit and upload; findings and release exclusions are recorded in
[Dutch HF release](dutch-hf-release.md). Earlier statements above about no upload
authorization describe the investigation/build phase and are superseded by that
explicit request.
