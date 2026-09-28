---
type: Evaluation
title: English and Danish source and corruption quality
description: Matched-protocol agent inspection of fresh English and Danish samples.
status: draft
generated: {by: codex/gpt-6, at: 2026-09-21T19:40:00+00:00}
sources:
  - id: danish
    resource: https://huggingface.co/datasets/giannor/dala_tv2r/tree/b2deeb25200996ccedf61858df9d4620ecb3a039
    title: Pinned Danish acceptability reference
  - id: pairs
    resource: https://huggingface.co/datasets/giannor/dala_gen_tv2r/tree/75a61fd90fd929a5859dd5caccde02300e5b5036
    title: Pinned Danish paired correction reference
  - id: mer
    resource: https://ordnet.dk/ddo/ordbog/mer
    title: Den Danske Ordbog — mere and unofficial mer
  - id: shrink
    resource: https://www.oxfordlearnersdictionaries.com/us/definition/english/shrink_1
    title: Oxford Advanced Learner’s Dictionary — shrink
---
# Prespecified comparison

The comparison draws 100 English canonical pairs and 100 Danish rows labeled
incorrect using independent, seeded uniform reservoir samples across all splits.
English sampling uses the final 478,930-pair Common Pile corpus, not its earlier
review queue. Danish sampling uses the full 492,063-negative acceptability
corpus.[^danish] Its paired correction export contains only 438,502 nonidentity
rows plus 492,063 identity controls; sampling that export alone would omit part
of the acceptability corruption distribution.[^pairs]

Danish originals are recovered by exact corrupted-text joins against the paired
export where possible. Other counterparts are retrieved from the published clean
acceptability rows and require explicit alignment inspection; an unresolved
counterpart cannot be counted as a validated pair.

Before inspecting the sample, the criteria are:

- Source: acceptable standard written grammar/spelling; erroneous; uncertain.
- Corrupted output: clearly erroneous; still acceptable; uncertain.
- Intended edits: all introduce errors; at least one remains acceptable; uncertain.
- Incidental formatting damage: changes beyond the intended error, recorded separately.
- Usable strict pair: acceptable original, clearly incorrect output, and all
  intended edits valid as errors. A valid alternative meaning does not make a
  sentence ungrammatical. Stylistic preference and factual accuracy are outside
  scope. Optional Danish start commas are not source errors.

These are same-agent linguistic judgments, not independent native-speaker gold
annotations. Samples, decisions, counterpart recovery, hashes and counts are
retained. Any uncertainty is reported separately rather than silently accepted.

[^danish]: Exact revision used for direct incorrect-row sampling.
[^pairs]: Supplies exact original/corrupted counterparts where available.

# Findings from the frozen sample

Completed on 2026-09-21. No sampled examples were removed from either corpus after
sampling. All 13 retrieved Danish originals were explicitly checked against the
corrupted text and retrieval candidates; they align as the labeled deletion or
neighbor swap. These are inferred counterparts, not newly discovered provenance
links. The other 87 counterparts are exact joins to the paired export.

| Agent judgment, out of 100 pairs | English | Danish |
| --- | ---: | ---: |
| Source acceptable | 80 | 83 |
| Source clearly erroneous | 11 | 14 |
| Source uncertain | 9 | 3 |
| Every intended edit clearly introduces an error | 100 | 87 |
| Intended edit can preserve grammatical acceptability | 0 | 6 |
| Intended edit uncertain | 0 | 7 |
| Strict usable pair | 80 | 71 |
| Added formatting damage | 0 | 2 |

Strict usable means an acceptable source, clearly invalid corrupted output, all
intended edits introducing errors, and no added formatting damage. An already
erroneous source can make a corrupted output clearly wrong even when the injected
edit is questionable; this does not validate the pair. Uncertain cases are not
counted as clear errors or strict usable pairs. Ordinary headlines and contextual
fragments are tracked separately and are not automatically rejected. Four Danish
sources have extraction damage, versus one English source. Seven additional
Danish sources are headlines/fragments.

Approximate 95% Wilson intervals for the strict usable proportions are
71.1–86.7% English and 61.5–79.0% Danish. These only express sampling uncertainty,
not reviewer bias, cross-language judgment differences or adjudication error.
The sample does not establish that one corpus has higher overall population
quality. In particular, 100/100 observed successful English corruptions does not
establish perfect generator precision, and rare generators may be absent.

# Concrete problems

English source examples that remain labeled clean:

- `en-021`: “each other weirdness” lacks the possessive.
- `en-030`: “looking his photos” lacks “at”.
- `en-053`: “experiences of Zimbabwean woman” needs an article or plural.
- `en-094`: “It was 1940s” lacks “the”.
- `en-051`: an uppercase heading is joined directly to running prose.

The corruption reverses to these originals; consequently the correction target
still contains an error, and the clean acceptability control is mislabeled.
This is the main English quality limitation observed here. Existing checker
screening and earlier exclusion rounds did not solve source correctness.
English source flags include punctuation errors as well as grammar/spelling;
regional or context-dependent constructions are retained as uncertain rather
than automatically rejected.

Danish has both source and corruption problems:

- `da-045`: the clean source already contains `forbindelsde` and `vanvidskørserl`.
- `da-054`: the clean source contains `har en ydet en stor indsats`.
- `da-086`: the source has duplicated `de de`; corruption also adds quote spaces.
- `da-055`, `da-088`, `da-090`: swapping parts of organization names can change
  the name without creating an ungrammatical sentence. The factual name change
  alone is outside the grammatical acceptability criterion.
- `da-072`: replacing `hun` with `det` can refer to another neuter entity; the
  output is not necessarily ungrammatical as a standalone sentence.
- `da-022`, `da-044`: `borgerne` becomes `borgene` (citizens → castles). The result
  is semantically implausible/personified, but remains syntactically well formed.
  These judgments explicitly separate semantic plausibility from grammar.

Other Danish cases remain uncertain, including modal ellipsis, headline article
omission, pronoun reference, clause intertwining, and informal shortened spellings.
For example, DDO describes `mer` as a common unofficial form, so `mere → mer`
is a weaker, register-sensitive negative than a clear agreement violation.
See the DDO entry.[^mer] English `has shrunk → has shrank`
was checked against the Oxford verb entry,
which distinguishes past `shrank` from participle `shrunk`.[^shrink]

# Coverage is different from precision

The complete English corpus contains 691,545 edits, of which 506,887 (73.3%) are
in the spelling generator family. Danish has 15,048 explicitly labeled spelling
corruptions among 492,063 negative rows (3.1%), with most other corruptions in
morphology/grammar families. This is not a perfect category equivalence: Danish
morphological and homophone errors can also be regarded as spelling errors, and
English permits two edits per pair while Danish labels one corruption per row.
Nevertheless, English currently emphasizes nonword typos much more heavily.
High surface diversity alone does not establish balanced grammatical coverage
or realistic learner-error frequencies.

English's sample contains 91 Global Voices, eight 360info and one Public Domain
Review pair. Source judgments are 73 acceptable / 10 erroneous / 8 uncertain for
Global Voices, 6 / 1 / 1 for 360info, and 1 / 0 / 0 for Public Domain Review.
These tiny secondary-source samples cannot establish publisher-specific rankings.
The pooled comparison samples each actual dataset's distribution; it is not
matched for sentence length, genre or corruption family.

# Release implication

English's guarded corruptions appear more reliable in this inspection, while
source correctness remains a substantial issue in both datasets. Danish is a
useful scale and task reference, not a gold-quality threshold. The English
release should not be described as high-quality validated correction data on
this evidence. Before publication, improve source selection/screening beyond
excluding these particular rows, reassess spelling/grammar balance, and perform
a new held-out audit with independent linguistic adjudication. Keep this frozen
sample for diagnosis; reusing it after targeted fixes would be optimistically
biased. No dataset was uploaded or mutated during this comparison.

# Reproducible evidence

- [Sampling receipts](../artifacts/language-quality-comparison/sampling.json): seed,
  pinned HF revisions, full source checksums and English manifest hash.
- [English decisions](../artifacts/language-quality-comparison/en-sample.json) and
  [Danish decisions](../artifacts/language-quality-comparison/da-sample.json): all
  originals, corrupted outputs, IDs, judgments and reasons.
- [Readable review](../artifacts/language-quality-comparison/review.md).
- [Summary](../artifacts/language-quality-comparison/summary.json),
  [full generator distributions](../artifacts/language-quality-comparison/full-distributions.json),
  and [Danish counterpart candidates](../artifacts/language-quality-comparison/danish-pair-recovery.json).
- Sampling implementation: `scripts/sample_language_quality.py`; rerunning writes
  fresh unannotated sample files, so preserve completed annotations first.

[^mer]: Consulted for register-sensitive spelling judgment, not a source of corpus labels.
[^shrink]: Consulted to check the participle corruption in en-042.
