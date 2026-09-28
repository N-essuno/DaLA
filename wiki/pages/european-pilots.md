---
type: Experiment
title: Twelve European DaLA pilots
description: Shared-pipeline candidate pilots, language-isolated inputs, measured coverage, source audit and scale-up requirements.
status: draft
generated: {by: codex, at: '2026-09-26T06:29:53.028942+00:00'}
sources:
  - id: ud
    resource: https://universaldependencies.org/format.html
    title: CoNLL-U format and annotation boundaries
  - id: dictionaries
    resource: https://github.com/wooorm/dictionaries/tree/8cfea406b505e4d7df52d5a19bce525df98c54ab
    title: Pinned standard-specific spelling dictionaries
  - id: bosque
    resource: https://github.com/UniversalDependencies/UD_Portuguese-Bosque
    title: Bosque mixed European and Brazilian provenance
  - id: voikko
    resource: https://voikko.puimula.org/
    title: Voikko Finnish morphology and spelling
---
# Contract and scope

The user requested pilots for German, French, Spanish, Italian, Czech, European
Portuguese, Finnish, Estonian, Catalan, Greek, Romanian and Ukrainian. Czech was
listed twice and is implemented once. The user additionally required sharing all
general code and keeping language-specific grammar/mistake inventories separate.
Only grammar and spelling are tasks; paraphrase, simplification, style transfer
and conversion between accepted standards are excluded.

The outputs are **candidate pilots**, not production releases. They exercise the
existing architecture with pinned annotated training prose, source filtering,
synthetic grammatical constraints and independently screened nonword spelling.
They do not yet mine the observed-error corpora catalogued in the
[language ranking](european-language-ranking.md). UD annotations and observed forms
are not exhaustive normative morphology, grammatical gold or natural error rates.
This limitation must not disappear when increasing the dataset size.

# Shared implementation and language isolation

`dala.pipeline.build` remains the entry point. The existing pair pipeline performs
candidate selection, document splits, near/exact deduplication, correction offset
reconstruction, provenance, balanced task views and atomic exports.
`MorphologyPack` remains the common generator. No new per-language Python adapters
or alternate export pipelines were created.

`config/european/{code}.json` supplies each language's own treebank/standard,
lexical backend, morphological constraints and controllers, spelling substitutions,
prompts, parser preparation settings and source-risk policy. Generated public
profiles are `config/languages/{code}.json`. Profile/resource/rulebook/rule/evidence
language mismatches fail closed; source language is checked by the annotation
adapter. pt-PT uses its own dictionary and only Bosque `CP` sentence IDs; `CF`
Brazilian records are excluded from both morphology mining and sentence sources.
The parser language can be `pt` while task and source identity remain `pt-PT`.[^bosque]

The inflection compiler was extracted from the earlier six-language preparation
script into `dala/rule_compiler.py`; old callers still use their old defaults.
New packs preserve all supplied features (including possessor features) and exclude
possessive determiners. These precautions avoid changing the possessor as a side
effect of an agreement corruption. Existing Nynorsk mappings and rule semantics
match a recompilation: the only serialization difference is sorting a
membership-only controller list. Receipt: `wiki/artifacts/european-pilots/
compiler-equivalence.json`. Danish production configuration was not changed.

Shared source/parser additions are in `dala/conllu_source.py`. They preserve
sentence text and exact offsets, reject annotated typos/foreign tokens and
unalignable multiword expansions, use immutable source commits and checksums,
and preserve annotated document groups. Where document boundaries are absent,
the complete source file stays in one split. Sentences are not relabelled as
independent documents just to fill validation/test. Sampling selects at most 3,000
eligible training sentences by stable sentence-ID hash; official UD dev/test text
is not used. Only one training shard per language is sampled.[^ud]

German HDT distributes token-spaced text. Its source configuration explicitly opts
into conservative punctuation detokenization (whitespace only). Provenance records
this representation; the pinned original bytes remain available. Corruption and
reconstruction offsets refer to the resulting source document text.

Spelling currently uses three configured mechanisms: internal character deletion,
internal transposition and language-specific diacritic/letter substitutions.
Correct originals must be recognized and replacements rejected by the lexical
backend. These are synthetic nonword mechanisms, not curated observed spelling
mappings. Hunspell dictionaries are separately pinned; Finnish uses locally
extracted Voikko packages through a shared configurable lexical interface.[^dictionaries] [^voikko]

# Reproduction and artifacts

Run from the repository root with the existing environment:

```sh
/tmp/dala-six-venv/bin/python -m scripts.prepare_european_pilots
/tmp/dala-six-venv/bin/python -m scripts.run_european_pilots --run-id NEW_RUN_ID --workers 4
```

Never reuse an existing output ID. Preparation records source commits, bytes,
licenses, dictionaries, analysis counts and rule inventories. Finnish package
versions are explicit in its input file; preparation downloads/extracts the
packages locally using apt-get/dpkg-deb, without system installation. Resource
files live under `la_output/resources/european/{code}`. Corpus licenses differ;
see each pinned README/LICENSE and exported document provenance. No upload or
blanket redistribution-license claim is made.

The selected current run is `la_output/european_pilots/audited_v2/{code}`.
`diagnostic_v1` and `pilot_v1` are superseded diagnostics, retained for evidence;
the superseded long Czech diagnostic was stopped before export. All twelve
`audited_v2` workers completed successfully. Per-language canonical pairs and
acceptability/correction task views are under `train`, `validation` and `test`.
Every pair produces two rows per task, including its unchanged clean control.

Automatic validation receipts, logs, coverage and stratified samples are under
`wiki/artifacts/european-pilots/audited_v2/`. The existing generic coverage reporter
(`scripts/report_six_language_coverage.py`, despite its historical filename) was
reused, as was `dala.validate_dataset`. All twelve passed artifact checksums,
source provenance, edit reconstruction, reverse correction, duplicate/split checks,
balanced labels and exact task-view reconstruction. Inputs and generation code
were stable during these builds. Additional language-isolation hardening was
applied after all builds completed; it does not rewrite their provenance.

# Measured pilot sizes and coverage

All counts below are distinct sentence pairs, not task rows. Total: **9,346 pairs**.
Grammar coverage shows selected/configured-active families, not linguistic quality.

| Language | Pairs | Grammar | Spelling | Grammar families | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| ca | 1,367 | 1,202 | 165 | 8/8 | 1,045 | 181 | 141 |
| cs | 880 | 531 | 349 | 9/10 | 700 | 68 | 112 |
| de | 1,264 | 830 | 434 | 11/11 | 1,264 | 0 | 0 |
| el | 34 | 29 | 5 | 4/8 | 29 | 5 | 0 |
| es | 1,099 | 959 | 140 | 7/8 | 866 | 120 | 113 |
| et | 957 | 345 | 612 | 6/6 | 772 | 79 | 106 |
| fi | 1,064 | 207 | 857 | 4/4 | 1,064 | 0 | 0 |
| fr | 449 | 125 | 324 | 6/8 | 449 | 0 | 0 |
| it | 649 | 565 | 84 | 8/8 | 623 | 13 | 13 |
| pt-PT | 142 | 116 | 26 | 3/8 | 107 | 17 | 18 |
| ro | 626 | 272 | 354 | 6/6 | 539 | 49 | 38 |
| uk | 815 | 261 | 554 | 7/7 | 732 | 70 | 13 |

Small or absent cells are retained as coverage gaps. In particular, pt-PT realizes
three grammatical families in this source sample; the configured verb families
have not been demonstrated by this pilot. Greek has only 34 pairs and no test
pairs. German/French/Finnish training-only results follow conservative grouping,
not successful held-out evaluation. Source exhaustion here means the bounded pilot
sample, not exhaustion of available text in the language.

# Quality evidence and limitations

A 212-pair agent inspection sampled up to two examples per selected family plus
three uniform examples per language from `pilot_v1`. It is a diagnostic sample,
not a statistical precision estimate. The agent provisionally accepted 192,
rejected 8 and marked 12 uncertain. All 20 rejected/uncertain originals were added
to language-specific exclusion files. Detailed judgments and original/corrupted
text: `wiki/artifacts/european-pilots/initial-agent-sample.json`.

Examples of discovered problems: an Italian original already had determiner-number
disagreement; a Czech original contained a masked number; a Romanian original
included a heading; a Portuguese proper name was misspelled. Uncertain source
punctuation, fragments and disputed usages were also excluded. The shared
source-agreement guard now optionally checks determiners as well as adjectives;
these pilot profiles opt in while earlier profiles retain their defaults. Czech
placeholder rejection, Romanian heading rejection and ellipsis rejection are data
in the relevant inputs. These changes produced `audited_v2`.

No native-speaker validation has occurred. Dictionary nonrecognition is not proof
that every replacement is invalid, and absence from a sampled morphology table
is not proof against syncretism or another valid analysis. The checker metadata
explicitly states that grammatical validity rests on generator constraints and
needs audit. Independent grammar-checker certification is not claimed. Uniform
and family-stratified final samples are exported for further review.

# Extending toward full size

The same packs can consume quality-screened clean prose through the existing
DynaWord/Common Pile adapters and a pinned live parser, instead of this annotated
pilot source. `production_parser` in each input supplies language-specific Stanza
settings; `scripts.prepare_european_pilots --languages CODE --live-parsers` uses the
shared parser-preparation helper to fetch and checksum models, writing
`live-parser-settings.json`. Model preparation is optional and was not run for
these pilots. Set a separate production profile's parser backend/settings and
source configuration; retain the same adapter, rules, prompts and task exporter.

Before scaling toward 478,930 pairs per language:

1. Mine and audit the language's own observed grammar/spelling evidence from the
   ranking; add curated mappings and error families without importing another
   language's assumptions. Preserve accepted dialect/orthographic alternatives.
2. Replace incomplete training-attestation morphology with independently normative
   analyses/generators, preserving all licensed analyses and applying syncretism
   vetoes. Add negative-context and valid-variant tests for each new family.
3. Pin appropriately licensed, quality-screened clean sources with reliable
   document IDs. Audit originals independently; annotated text is not automatically
   clean. Resolve pt-PT modern/historical orthography explicitly.
4. Validate live-parser behavior and domain shift on fresh source text. Complete
   missing family coverage and independently held-out evaluation before claiming
   production readiness. Greek and pt-PT especially need broader usable evidence;
   German, French and Finnish need independent validation/test source documents.
5. Use the existing resumable batch pipeline, unchanged output schema and
   document-level splits. Remove the pilot sample cap, measure yield and coverage,
   and only then schedule a full-size run. Do not duplicate sentences or relax
   linguistic constraints to meet the numerical target.

[^ud]: CoNLL-U format, including raw text, multiword tokens and document markers.
[^dictionaries]: Resource-specific spelling dictionaries; their licenses do not determine corpus-text licensing.
[^bosque]: The treebank documents both CETEMPúblico and CETENFolha origins.
[^voikko]: Finnish spell checking and morphology; backend availability is not native linguistic review.

# Subsequent CPU expansion

See [the CPU expansion experiment](european-cpu-expansion.md) for completed live-parser preparation, new independent source pools, descriptive lexical inputs, observed correction mining and stricter source audits. The baseline counts above remain unchanged; the newer versioned candidates have separate receipts and current pointers.
