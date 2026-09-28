# Separate language inputs, shared CPU pipeline

These inputs extend the twelve original packs in `config/european/`. They do not
replace the Danish, English, Dutch or earlier six-language packs.

- `resource-lock.json`: immutable downloads and byte checksums for Wikipedia,
  Europarl and UniMorph; fetched by `scripts.fetch_european_expansion`.
- `sources.json`: source format, written-standard identity, license provenance,
  complete-document pilot sampling and Europarl sitting grouping.
- `morphology/CODE.json`: language-specific tag conventions and lexical resource.
  Unknown tags are retained for vetoes and cannot generate errors. No entry is
  deemed normative merely because it occurs in UniMorph.
- `evidence/CODE.json`: training-only correction input hashes, revisions, categories
  and support units. There are currently observed-corpus recipes for de/cs/uk.
- `spelling/CODE.json`: explicitly agent-reviewed subsets of observed nonwords;
  these are not native-speaker validation. Other mined candidates stay inactive.
- `checker-evidence.json`: contextual expert-authored examples, kept separate from
  observed corrections and never activated as context-free substitutions.
- `audit-policy.json`: per-language source exclusions and screening policies.
  Source checker support differs; cs/fi/et have no backend in the pinned runtime.

Resource assembly, parsing, candidate generation, screening, splits and task
exports remain shared. All Stanza inference uses `use_gpu=False`. Versioned
profiles are passed to `dala.pipeline.build`, directly or through
`scripts.run_european_pilots --profile-root ...`.

See `wiki/pages/european-cpu-expansion.md` for measurements, limitations and the
runbook. Larger runs can use the existing resumable batch pipeline after source,
variant and per-family quality review. Increasing a source sample alone does not
establish production readiness.
