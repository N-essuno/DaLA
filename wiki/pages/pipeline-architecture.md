---
type: Architecture
title: Language input packs and compatibility
description: Shared pipeline, declarative Danish operators, productive English adapter and equivalence boundaries.
status: stable
generated: {by: codex/gpt-6, at: '2026-09-21T16:49:31+00:00'}
sources:
  - id: inflection
    resource: https://lemminflect.readthedocs.io/en/latest/inflections/
    title: Dictionary-backed inflection API
---
# Inputs and shared components

`dala.pipeline.build(profile, **options)` is the common API.
`python -m dala.multilingual --profile config/languages/da.json` and the matching
English profile use this API. A custom profile path does not require editing a
language-code registry. `--language da` and `--language en` are shorthand.

A profile supplies language identity, parser, source settings, rule resources,
selection policy and, for pair exports, prompts, screening dialects and an adapter.
Resources resolve relative to the profile file. There are two explicit output
policies, not a language-code conditional:

- `legacy`: historical sentence splitting, first-success corruption, reconstruction,
  random-number behavior, balanced CSV output and optional full_train view.
- `pairs`: document-level splitting, exact character edits, independent-error
  composition, provenance, checker screening, instruction exports and review.

Keeping the compatibility policy is necessary for exact Danish equivalence;
forcing the new document-split or reconstruction policy on Danish would change
its results. Both policies are language-parameterized. The EWT prototype remains
an explicit historical option, not another production language implementation.

# Danish data pack

`config/languages/da.json` supplies all active rule order, lexical mappings,
negation/antecedent vocabulary, suffix changes, morphology predicates, casing
policy, parser name, DDT URLs, source filters and split settings. The production
code does not contain Danish word lists. Generic operators implement lookup,
suffix changes, child lookup, following-noun constraints, antecedent constraints,
genitive-marker alternatives, and the historical POS-constrained fallback.

Compatibility entry points in `create_dala.py`, `dala_corrupt.py`, `dala_utils.py`
and `dala_enums.py` read this pack. The frozen old implementation lives only in
`tests/reference_danish`. Its historical quirks, including lowercase `case=Gen`
and the indefinite-article diagnostic's reported token, are intentionally retained.
This refactor is not a Danish linguistic-rule correction.

# English input adapter

`config/languages/en.json` supplies English vocabulary/contexts and references
`config/english_productive_rules.json`, source snapshots and source exclusions.
The English adapter proposes edits; the shared pair pipeline handles source
iteration, selection invocation, splits, exports, screening and provenance.
Other languages can supply an adapter implementing the same methods without
copying these stages. An adapter is Python code referenced by the input pack;
complex language analyses are not expressed as an unrestricted JSON mini-language.

Five grammatical families now use productive dictionary-backed inflection:
present agreement, modal complements, do-support, perfect participles and noun
number. The original empirical mappings remain embedded as family evidence.
New surface pairs are explicitly generalized, not claimed to occur in W&I.
Dictionary forms must confirm the original tag; replacement forms that are also
valid originals are excluded. Out-of-vocabulary inflection is disabled.[^inflection]

Demonstrative substitutions remain lexical evidence-backed entries. Spelling
retains the lexical inventory and now also supports screened character patterns.
The English adapter adds syntactic guards for token fallback; see
[generator evaluation](/pages/english-generators.md). Character operations are
shared primitives parameterized by rule data; source orchestration is unchanged.
Demonstratives additionally exclude ambiguous number forms such as fish, series
and means, which can otherwise make both demonstrative choices grammatical.
Agreement uses personal pronouns, distributive each/every singular nouns and
restricted plural noun subjects. Unrestricted singular noun agreement is excluded
because British collective agreement can make both alternatives acceptable.
Multi-error examples retain at most one grammar edit plus independent spelling.

# Danish equivalence

The complete real-DDT comparison passed 77,014 targeted-rule evaluations over
5,501 distinct sentences. Both fixed-size and proportional pipelines were compared
with random seeds 4242 and 73. Every output row, CSV byte and final random state
matched. All 14 targeted rules had positive cases; complete builds also exercised
both fallback operations. See [machine receipt](../artifacts/danish-equivalence.json).

The equality claim is conditional on identical inputs and parser annotations,
model version and random state. It does not assert equality across future model
versions or a changing upstream master branch. The receipt pins the tested DDT
commit. Production source URLs remain configurable, including the historical
master URL needed to preserve the existing default.

[^inflection]: LemmInflect 0.2.3 API, with `inflect_oov=False`; dictionary membership is not itself a grammaticality guarantee.
