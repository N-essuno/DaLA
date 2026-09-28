---
type: Reference
title: TV2R reference and implementation history
description: Reference task structure, inspected DFM11 integration and superseded
  pilot decisions.
status: stable
generated:
  by: codex/gpt-6
  at: '2026-09-21T15:13:36+00:00'
sources:
- id: tv2r
  resource: https://huggingface.co/datasets/giannor/dala_tv2r_it
  title: TV2R acceptability instruction reference
- id: gec
  resource: https://huggingface.co/datasets/giannor/gec_dala_tv2r_it
  title: TV2R correction instruction reference
---
# TV2R reference

The project-owner-directed inspection of `giannor/dala_tv2r`, `giannor/dala_tv2r_it`
and `giannor/gec_dala_tv2r_it` established raw acceptability and instruction-shaped
acceptability/correction views. Clean correction inputs map to themselves;
corrupted inputs map to the original. Sampled rows contain spelling corruptions.
The HRM-Text DFM11 prefix policy includes both instruction datasets at repeat 1.
Its knowledge record reports 492,063 underlying clean TV2R news sentences, expanded
into 984,126 acceptability and 930,565 correction rows. These totals were read from
that repository, not recomputed over the full Hub datasets in this session.[^tv2r]

Pinned inspected revisions: raw `b2deeb25200996ccedf61858df9d4620ecb3a039`,
acceptability IT `9976093cd4a6d6ee126d6389a267019e361a841c`, correction IT
`308570cc643b1d4d2ab9ba2771a0093764838003`. Reference files:
`/work/mimir/HRM-Text/data_io/prefix_config_dfm11.yaml`,
`scripts/convert_dfm8_giannor_tv2r.py`, and
`wiki/pages/dfm8-plan/danish-linguistic-acceptability-and-gec-data.md`.

# Superseded pilot

The `multilingual` branch initially added English EWT using gold UD annotations,
six grammar rules and exact single-span changes. A full pilot produced 3,833 pairs
and passed 17 tests. Inspection revealed pre-existing errors in EWT and led the
owner to request observed correction patterns and cleaner Common Pile sources.
The owner further clarified that spelling and multi-error examples are in scope.
The current contract supersedes the EWT production-source proposal, grammar-only
scope and restriction of mining to single-error sentences. EWT remains an
explicit optional pilot for regression comparisons; Danish rules are retained.

GECToR and CoEdIT were investigated as pointers to human-corrected corpora.
Their synthetic pretraining data and non-correction editing tasks do not establish
natural error patterns or unacceptability. W&I/FCE/NUCLE, native LOCNESS and GMEG
were identified as useful evidence sources; the implemented rulebook currently
uses W&I training evidence only.

[^tv2r]: Inspected TV2R instruction dataset and its companion correction dataset.

# Integration check

Both English instruction views load into Hugging Face `Dataset` and pass the
existing HRM-Text TV2R converter's content/response handling. That converter
hardcodes `language=da` and a Danish category: actual English ingestion requires
an English configuration or adapter. HRM-Text was inspected only, not modified.

# Danish code audit: diversity and lexical inventories

Inspected `dala/dala_corrupt.py` and `dala/create_dala.py` directly. The active
registry has 15 functions: 14 targeted families plus `corrupt_basic`, which
chooses deletion or neighbouring-token swapping. Three additional experimental
functions below the excluded-corruptions heading are not active.

Diversity is encouraged by a fixed first-success ordering described in code as
ascending Danish UD eligibility frequency: uncommon applicable families get
priority before broad families consume their sentences. This is not adaptive
balancing, a per-word cap, or a guarantee of equal family coverage. Most targeted
functions choose the first eligible token with default flip probability 1.
Genitive alternatives and the basic fallback introduce randomness. Source
preparation removes duplicate documents and requires more than five distinct
POS tags; output rows are shuffled. These do not guarantee lexical diversity.

There is no closed inventory of all exact Danish substitutions. Six families
operate productively on suffixes or genitive markers across matching vocabulary:
`flip_en_et_suffix`, `corrupt_ende_ene`, `corrupt_verb_r`, `corrupt_noun_r`,
`corrupt_adjective_r`, and `corrupt_genitive`. The fallback also generalizes over
sentence tokens. Actual distinct output edit counts would require counting a
specified generated dataset.

Counting only explicitly fixed whole-word directed replacements, ignoring case,
gives 74: spelling 45, pronoun case 14, ligge/lægge forms 6, en/et 2, nogle/nogen 2,
som→der 1, han/hun→det 2, and får/for 2. AST inspection confirms the spelling
lists have 46 entries but only 45 unique mappings (`rimelig→rimlig` is repeated).
These are hardcoded mappings, not 74 corpus-attested pairs with evidence counts.

Consequently, the English implementation's requirement that every grammatical
surface substitution appear in a 92-entry inventory is more restrictive than
Danish DaLA. Its rare-first selection follows a similar principle but does not
fix limited lexical generalization. English correction evidence should justify
carefully guarded productive error rules as well as lexical spelling entries;
expanding an exact-pair list alone misses the Danish design's central source of
lexical coverage. This audit records the distinction; it does not change the
current English generator or its build receipts.
