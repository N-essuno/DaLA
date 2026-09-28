---
type: Evaluation
title: Productive spelling and English token fallback
description: Character generators, English deletion and swap guards, and evaluation of their outputs.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-21T17:45:43.128974+00:00}
sources:
  - id: keystrokes
    resource: https://aclanthology.org/P12-2073/
    title: Baba and Suzuki 2012 spelling errors in keystroke logs
  - id: synthesis
    resource: https://aclanthology.org/W19-4415/
    title: Xu et al. 2019 erroneous data generation for grammatical error correction
  - id: variant
    resource: https://www.larousse.com/en/dictionaries/english-german/reprogram/35800
    title: Larousse reprogram inflection variants
---
# Policy

The active English rulebook retains its 72 lexical spelling entries and nine
existing grammar rules, and adds three productive character mechanisms:
internal adjacent-letter transposition, omission of a doubled consonant, and
insertion of a repeated consonant. Operation parameters, bounds, allowed POS,
protected endings and evidence references are rulebook data. Each pattern links
to two attested mappings. Research describes these spelling-error mechanisms,
but does not establish our generated word pairs or their frequency as observed
errors.[^keystrokes][^synthesis]

Candidates use 6–24-letter ASCII words, exclude names and proper nouns, preserve
case, and veto replacements present in the parser's vector vocabulary. Attested lexical spelling has priority over productive characters within the
spelling family. This preference is an input in the English profile, and prevents
the much larger generated candidate pool from crowding out the curated inventory.
Every
selected productive spelling requires an exact misspelling diagnostic in both
US and British English, while the original spelling must be recognized in both.
Each pair then receives the existing source and contextual edit checks. Building
these rules with the checker disabled fails explicitly. Exported records contain
per-edit lexical evidence and the independent validator requires it.

The first probe found `reprogrammed → reprogramed` incorrectly accepted by both
checker dictionaries. The latter is a documented variant.[^variant] Doubling
operations now abstain at protected suffix boundaries, including ed/ing/er/ers/
est/ment/ments/en/s. This deliberately also excludes some genuine errors, such as
stopping → stoping, rather than assuming an exhaustive list of regional forms.
Names, vocabulary filtering and dictionary agreement remain heuristics, not a
proof of linguistic correctness.

# Token fallback

When the existing targeted grammar and lexical rules produce no candidates,
English considers two bounded token operations before productive spelling:

- Delete a finite progressive be-auxiliary immediately before its root VBG verb,
  with an explicit preceding subject and no other finite verb or auxiliary.
- Swap an article with its immediately adjacent object/prepositional-object noun,
  at a following punctuation/preposition boundary, excluding names and compounds.

One fallback edit is selected using the existing seeded family-priority/hash
selection. Deletion has priority over swapping when both apply; this is not a
uniform draw between families. Other candidates may be selected at different
seeds. Fallback protects the whole sentence from a second edit. Deletion records
include the surviving neighbour as a nonempty span anchor; spacing and source
offsets are preserved. These are synthetic syntax rules, not claimed attested
lexical corrections. No arbitrary adjective/adverb/subject removal is enabled.

# Probe before full build

A seeded sample of 1,500 originals from the previous build supplied up to 60
candidates per new rule. The first 100 originals also received each historical
Danish POS operation. These unequal, capped counts are diagnostic samples, not
estimates of natural error frequency or precision.

| Operator | Proposed | Checker retained |
| --- | ---: | ---: |
| Legacy deletion | 100 | 23 |
| Legacy neighbour swap | 100 | 21 |
| Guarded progressive auxiliary deletion | 9 | 2 |
| Guarded article/noun swap | 60 | 20 |
| Doubled-consonant omission | 60 | 58 |
| Consonant repetition | 60 | 60 |
| Internal transposition | 60 | 58 |

Legacy token reconstruction changed spacing at apostrophes/hyphens, and some
word deletions remained grammatical; a generic checker could react to incidental
formatting instead of the intended corruption. Thus the broad legacy operators
remain disabled for English. Low checker retention of guarded token rules is
not an estimate that their rejected outputs were correct; many obvious errors
were simply undetected. No attempt was made to relax screening to boost yield.

After the variant fix, agent inspection found all 22 retained guarded-token
pairs and all 176 retained character substitutions valid as injected errors.
Source correctness was judged for the 22 full token pairs, not inferred for the
176 lexical checks. Two defective originals encountered in the broader probe
were added to the source exclusion input. This is agent review, not independent
human linguistic validation or a corpus-wide precision estimate.

[Initial counterexample receipt](/artifacts/english-generator-probe-initial.json).
[Revised probe and agent judgments](/artifacts/english-generator-probe.json).

[^keystrokes]: Primary empirical study of spelling mechanisms; used as mechanism evidence, not a frequency model.
[^synthesis]: Primary GEC synthetic-error paper; related methodology, not validation of this implementation.
[^variant]: Publisher dictionary explicitly lists both reprogrammed and reprogramed.

# Final-build review decisions

The first full build yielded 38,207 pairs. A 59-pair agent sample covered all 11
retained auxiliary deletions and 12 pairs per other new operator. Every injected
edit was judged invalid English as intended, but one original was rejected and
three originals were uncertain. All four originals were added to the source
exclusion input. The [review receipt](/artifacts/english-generators-agent-review.json)
retains these judgments rather than marking all checker-passed sources clean.

The first full-build metrics also showed that random selection among the much
larger character candidate pool displaced most attested spellings. The final
selection explicitly prioritizes lexical spelling before productive character
rules. This is not a frequency cap or a natural-error-frequency model. Grammar
priority and the one-grammar-edit constraint are preserved.

The earlier full output is retained at `la_output/english_generators_pre_audit/`.
Danish files still match the equivalence receipt:
[hash comparison](/artifacts/danish-unchanged-after-generators.json).

# Final result

`la_output/english_generators/` contains **38,434 pairs / 76,868 rows per task**,
compared with 21,090 pairs before the generators (+82.2%). It uses the same pinned
1,939-document pool and 81,749 parsed sentences; 1,680 documents contribute pairs.
Splits contain 30,544 training, 3,749 validation and 4,141 test pairs.

| New operator | Retained edits | Distinct surface substitutions |
| --- | ---: | ---: |
| Internal letter transposition | 15,230 | 5,939 |
| Consonant repetition | 14,446 | 5,543 |
| Doubled-consonant omission | 361 | 253 |
| Guarded article/noun swap | 2,105 | 1,095 |
| Guarded auxiliary deletion | 11 | 11 |

The lexical inventory contributes 11,540 edits across 66 realized mappings,
compared with 11,665 edits across 67 mappings previously. It is preserved as the
preferred spelling source rather than displaced by generated candidates.
Across all families, **14,912 distinct substitutions** are realized, versus 2,088
previously. `that → taht` accounts for **9.4% of all edits**, down from 22.0%, and
12.6% of spelling edits, down from 45.3%.

Spelling now accounts for **74.6% of edits**. These figures reflect configured
coverage and selection, not a natural error distribution or a balanced benchmark.
Auxiliary deletion is especially scarce because ordinary present-tense agreement
rules already cover many such contexts and take precedence over fallback.

The four flagged originals are absent from the final build. All 55 retained
originals from the previous review were matched to final outputs, and changed
corruptions were reinspected: their originals and injected edits were accepted
by the agent. This is a follow-up to that sample, not a fresh random sample or a
human precision estimate. [Final review](/artifacts/english-generators-final-review.json).

All **65 tests** pass. Final validation confirms artifact checksums, edit
reconstruction, inverse offsets, task views, balanced labels, document split
isolation, and required lexical evidence. Final code, profile, rulebook and
source-exclusion hashes match the build manifest. Danish implementation/profile
hashes still match the earlier successful equivalence receipt.

[Manifest](/artifacts/english-generators-manifest.json),
[validation](/artifacts/english-generators-validation.json),
[comparison](/artifacts/english-generators-comparison.json).
