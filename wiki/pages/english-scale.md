---
type: Evaluation
title: DaLA English — Common Pile
description: Reference sizes, expanded sources, resumable construction and quality assessment.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-21T19:21:33.493341+00:00}
sources:
  - id: reference
    resource: https://huggingface.co/datasets/giannor/dala_tv2r_it
    title: Giannor instruction acceptability dataset
  - id: reference-gec
    resource: https://huggingface.co/datasets/giannor/gec_dala_tv2r_it
    title: Giannor instruction correction dataset
  - id: news
    resource: https://huggingface.co/datasets/common-pile/news
    title: Common Pile news
  - id: gv-license
    resource: https://globalvoices.org/about/global-voices-attribution-policy/
    title: Global Voices republishing and attribution policy
  - id: gv-editorial
    resource: https://globalvoices.org/about/editorial-code/
    title: Global Voices editorial code
---
# Target and source pool

The reference acceptability dataset contains 984,126 rows: 787,300 training,
49,208 validation and 147,618 test rows. Its correction counterpart contains
930,565 rows.[^reference][^reference-gec] English targets 492,063 distinct clean/corrupted
pairs, yielding 984,126 rows in each task view. Pair quotas are 393,650 / 24,604 /
73,809, with document-hash assignment at 80/5/15. Size comes from more source
sentences, without repeating pairs or increasing variants per original.

`config/english_sources_scale.json` adds 32,153 Global Voices articles of at
least 1,800 characters to the existing 1,673 360info and 266 Public Domain Review
documents. All snapshots remain pinned to immutable Common Pile revisions.[^news]
Global Voices has an editorial code, but that is not evidence that every sentence
is free of errors, especially translations and reproduced quotations.[^gv-editorial]

The publisher states CC BY 3.0 while the captured Common Pile metadata says
CC BY 4.0. Document provenance retains both declarations, the publisher policy
URL, author, source URL, snapshot checksum and source offsets. The operative
license field uses the publisher's BY 3.0 statement.[^gv-license] This build uses
text only. Other inspected news sources were not enabled merely to boost volume.
Authors are preserved where supplied by the snapshot; missing author values
remain null. Of the eligible source documents, 737 lack author metadata. Source
URLs remain available for attribution and inspection.

# Implementation and checks

The scale profile reuses the English rules, candidate selection, guards, checker
and task exporter. A shared `prepare_sentence` function keeps ordinary and batched
construction aligned. Eight parser processes prepare deterministic document
batches; a local checker screens them in batch order. Checksummed candidate and
screened receipts permit resuming without recomputing completed batches. Input,
profile, parser and implementation hashes reject incompatible restarts.

During implementation review, the indexed deduplicator was made to explicitly
disable SequenceMatcher's popular-token heuristic, matching the original
implementation even on long repetitive texts. A >200-token adversarial regression
now covers this case. The initial run's parser/checker receipts are reusable:
the finalizer verifies the complete prior code hashes, permitting only explicit
final-deduplication and edit-validation patches, and repeats final deduplication
with the corrected predicate. Generation and finalization code hashes are
recorded separately. Deduplication/validator patches are reversed literally and
the complete prior file hash must match. Subsequent JSONL-reader-only fixes use
recorded before/after hashes and diffs. Arbitrary generator changes cannot reuse
receipts.

The scale checker uses 16 server threads, 64 clients and batched cache commits;
these settings change throughput, not acceptance criteria. Indexed near-duplicate
retrieval preserves the original trigram/SequenceMatcher predicate while avoiding
frequent-trigram scans. Final selection applies global deduplication, review
exclusions and split quotas. Insufficient source yield fails with checkpoints
preserved; it does not silently repeat records to meet the target.

A 100-document smoke run produced exactly the same 1,550 pair dictionaries using
ordinary construction, checkpointed construction and checkpoint resumption.
The randomized/adversarial near-duplicate comparison also passed. These are
implementation checks, not linguistic validation.

Run with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m dala.multilingual \
  --profile config/languages/en_scale.json --offline \
  --output-dir la_output/english_common_pile_scaled
```

The output path must be absent. Completed batch receipts live in
`la_output/english_scale_checkpoints/`. Review exclusions are a separate final
selection input; their content hash is recorded in the exported manifest.
If the inputs or generation code change, use a fresh checkpoint directory rather
than editing a receipt. For the initial run's completed receipts, use the finalizer
below; it validates the narrowly scoped deduplication compatibility change.

# Initial quality finding

Agent inspection of 70 family-stratified pairs from the first screened batch
accepted all injected errors, but flagged 12 originals (nine rejected, three
uncertain). Examples include `Prime Minster`, `more ... that`, missing
prepositions, comma splices and an embedded language-link marker. Their hashes
are excluded from final selection. This finding demonstrates checker blind spots
in the expanded source pool; removing inspected examples does not establish
that unseen sources are clean. The sample is not a corpus-wide precision estimate
and there is no independent human validation.

A separate simple random sample of 60 pairs from the third screened batch
accepted all injected edits, but flagged four originals as erroneous, two as
uncertain and four as navigation/republishing boilerplate. The remaining 50
sources were accepted by the agent for grammar/spelling, without judging their
factual content or style. This sample predates final curation and deduplication;
it is not a post-curation precision result. The source-quality finding is a
material limitation of the expanded corpus, not resolved by removing reviewed
examples.

`scripts/build_scale_review_exclusions.py` combines recorded source flags with
explicit Global Voices navigation and republishing patterns. It also excludes
other sentences from documents with rejected/uncertain source judgments, since
nearby unreviewed prose may share the observed source-quality problems. It emits input
hashes and an exclusion receipt; it never repairs originals. If the complete
source pool falls short of the exact target, `scripts/finalize_scale_dataset.py`
verifies all screened receipts and immutable inputs, repeats global deduplication
after exclusions, and chooses attainable approximately 80/5/15 quotas. No split
is oversampled. The exported card and manifest disclose the quality findings.

After all batches are screened (the exact-quota build may report insufficient
source yield), finalize with a fresh output directory:

```sh
.venv/bin/python -m scripts.build_scale_review_exclusions
.venv/bin/python -m scripts.finalize_scale_dataset --output-dir la_output/english_common_pile_scaled
.venv/bin/python -m dala.validate_dataset la_output/english_common_pile_scaled
.venv/bin/python -m scripts.assess_scale_dataset
```

Preserve any pre-review output under a different name before finalizing. The
finalizer does not overwrite it. For a fresh full reproduction, generate into a
separate staging output, then run the curation/finalization steps. The ordinary
builder alone does not regenerate the corpus-wide boilerplate-exclusion input.

[First-batch review](/artifacts/english-scale-first-batch-review.json),
[random batch review](/artifacts/english-scale-random-batch-review.json),
[smoke equivalence](/artifacts/scale-smoke-equivalence.json),
[reference size receipt](/artifacts/reference-size-dala_tv2r_it.json),
[correction size receipt](/artifacts/reference-size-gec_dala_tv2r_it.json).

# Full-scale validation finding

The first final export stopped before writing any dataset because independent
edit validation rejected two token swaps: `the café → café the` and
`a façade → façade a`. These were legitimate instances of the configured rule;
the generator accepted alphabetic Unicode words but its validator only accepted
ASCII. The validator now accepts alphabetic Unicode tokens and the same whitespace
contract as generation, while rejecting digits, underscores and unchanged swaps.
All screened edits were inspected mechanically to identify the two mismatches,
and a parser integration regression covers both words. No candidate-generation
or checker threshold changed. [Failure receipt](/artifacts/english-scale-edit-validation-failures.json).

The first written export exposed a separate reader bug: Python `splitlines()`
split valid JSON strings at embedded U+2028 separators. JSONL review, validation
and upload readers now split physical records with file iteration; a complete
export/load/validation regression preserves an embedded separator. Source curation
also excludes originals containing Unicode line/paragraph separators, since
inspection found a heading and sentence fragment joined in one such source.
The first written export is preserved at `la_output/english_common_pile_scaled_pre_audit/`;
it is superseded. [Reader fix hashes and diffs](/artifacts/english-scale-jsonl-reader-fix.json).

# Final scale result

`la_output/english_common_pile_scaled/` contains **478,930 distinct pairs**, supplying
**957,860 rows in each task view**. This is 12.46 times the previous 38,434-pair
English corpus. It is 2.67% below the 984,126-row acceptability reference and
2.93% above the 930,565-row correction reference. No records were duplicated to
reach a requested size.

| Split | Pairs | Rows per task |
| --- | ---: | ---: |
| Train | 383,144 | 766,288 |
| Validation | 23,946 | 47,892 |
| Test | 71,840 | 143,680 |
| Total | 478,930 | 957,860 |

The build parsed 1,589,647 sentences from 34,092 eligible documents and proposed
655,277 candidates. The checker retained 490,359; review/format/boilerplate
exclusions removed 2,881 occurrences, exact collisions removed 3,092 and near
duplicates removed 1,509. Of 482,877 unique curated pairs, final split caps removed
3,947. Test availability determined the attainable total. The exported corpus
uses 33,274 documents and has 440,818 Global Voices, 29,606 360info and 8,506 Public
Domain Review pairs.

The final 650 excluded-original hashes cover recorded source flags, the other
sentences from their documents, recurring boilerplate and eight strings with
embedded line separators. These exclusions do not constitute broad linguistic
validation. The dataset card explicitly identifies the output as a provisional
checker-screened corpus rather than a human-validated benchmark.

All **73 tests** pass. Independent artifact validation checks checksums, exact
edit reconstruction, correction round trips, document split isolation, unique
exported text, matching task views, balanced labels and source provenance. Final
implementation, profile, rulebook, source configuration, exclusions and finalizer
hashes matched the exported manifest at build completion. Danish core/profile hashes still match the
earlier successful 77,014-rule/four-full-build equivalence receipt.

Across 691,545 edits, **47,073 distinct surface substitutions** are realized,
versus 14,912 previously. Spelling supplies 506,887 edits (73.3%); the 72-entry
lexical inventory realizes 69 mappings. The largest substitution, `that → taht`,
accounts for 81,585 edits: 11.8% overall and 16.1% within spelling. Its share rose
from 9.4% overall in the smaller corpus; scaling broadens coverage but does not
automatically flatten frequencies.

| Productive operator | Edits | Distinct substitutions |
| --- | ---: | ---: |
| Internal character transposition | 171,083 | 17,760 |
| Consonant repetition | 161,177 | 16,749 |
| Doubled-consonant omission | 4,002 | 1,135 |
| Guarded article/noun swap | 31,195 | 5,308 |
| Guarded auxiliary deletion | 231 | 155 |

All original source spans and attribution fields were independently compared
with the pinned snapshots. All exclusions are absent. The 108 accepted pairs
from the two agent samples remain exactly unchanged in the final output; this
is a transfer of prior judgments, not a fresh post-curation precision sample.
Author metadata is present for 468,193 pairs and missing for 10,737. The longest
normalized sentence is 70 tokens, below SequenceMatcher's popular-token threshold;
the compatibility fix nevertheless protects future longer inputs.

[Manifest](/artifacts/english-scale-manifest.json),
[artifact validation](/artifacts/english-scale-validation.json),
[amount/diversity/source-span assessment](/artifacts/english-scale-comparison.json),
[review transfer](/artifacts/english-scale-final-review.json),
[exclusion receipt](/artifacts/english-scale-exclusion-receipt.json),
[input integrity](/artifacts/english-scale-input-integrity.json),
[Danish unchanged](/artifacts/danish-unchanged-after-scaling.json).

# Naming update

The current name is **DaLA English — Common Pile**, and the current directory is
`la_output/english_common_pile_scaled/`. The earlier `english_tv2r_scale` name
referred only to a size comparison and misleadingly suggested Danish TV 2 source
material. TV2R remains the name of the Danish reference datasets.

Only display metadata, documentation and directory names changed. The original
manifest/card are archived; earlier evaluation receipts retain their original
paths and manifest hashes as historical evidence. The current manifest records
the rename and retains the generation-profile hash. The finalizer permits a
profile display-name change only when every other profile setting is identical.
The pre-audit directory was renamed consistently. All data-artifact checksums
were rechecked after the rename.

[Rename receipt](/artifacts/english-scale-rename.json),
[original manifest](/artifacts/english-scale-before-rename/manifest.json).

[^reference]: Dataset server size endpoint captured in the reference receipt.
[^reference-gec]: Separate correction dataset size captured in its receipt.
[^news]: Pinned source configuration records the exact revision and shard.
[^gv-license]: Publisher attribution policy; captured metadata remains separately traceable.
[^gv-editorial]: Editorial standards support source selection, not a linguistic precision claim.
