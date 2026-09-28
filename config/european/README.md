# European DaLA pilot inputs

The twelve `<language>.json` files are language-specific data, not generation
implementations. Each selects its own treebank/standard, dictionary, grammatical
constraints, spelling substitutions, prompts and exclusions. Czech occurs once.
`*-sources.json` files are generated pinned source manifests; `*-exclusions.json`
contains hashes of rejected or uncertain originals from agent inspection.

Shared implementation:

- `dala.rule_compiler.compile_inflections`: the same compiler used by the earlier
  six-language preparation script. New packs preserve every supplied feature and
  exclude possessives; legacy defaults remain unchanged.
- `dala.language_packs.morphology.MorphologyPack`: contextual constraints, lexical
  screening and selection for both old and new morphology packs.
- `dala.conllu_source`: annotated-source adapter, exact offsets, source identity,
  deterministic bounded sampling and conservative document grouping.
- `dala.pipeline.build`: existing deduplication, checker integration, task views,
  document splits, artifact checksums and exports.

These are diagnostic grammar/spelling pilots. UD attestation is not exhaustive
morphology or observed-error-frequency evidence. Candidate spelling is dictionary-
screened character corruption, not a claimed curated misspelling list. Research
resources for observed mistakes are in the OKF language ranking; corpus mining and
native linguistic review remain distinct future work.

To expand, replace the pilot source adapter/configuration and annotated parser
with quality-screened clean documents and a pinned live parser, retaining the same
language-specific rulebook, prompts and shared pair pipeline. Expand morphology
with the independently normative resources in the ranking, retaining all analyses
and accepted variants. Remove `pilot_max_sentences` only after source/rule audits.
Use the existing checkpointed batch pipeline and select toward 478,930 pairs.
Pilot test splits are not independent grammatical gold; some source files lack
reliable document boundaries and therefore cannot provide all three splits.
