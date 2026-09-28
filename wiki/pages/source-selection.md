---
type: Reference
title: Source selection and screening
description: Pinned news and contemporary essay sources with document provenance and
  sentence checks.
status: stable
generated:
  by: codex/gpt-6
  at: '2026-09-21T15:13:36+00:00'
sources:
- id: news
  resource: https://huggingface.co/datasets/common-pile/news
  title: Common Pile news
- id: pdr
  resource: https://huggingface.co/datasets/common-pile/public_domain_review
  title: Common Pile Public Domain Review
- id: lt
  resource: https://dev.languagetool.org/http-server
  title: Local LanguageTool server
---
# Selected sources

`config/english_sources.json` pins two Common Pile collections:

| Subset | Commit | Documents after source metadata checks |
| --- | --- | ---: |
| News, 360info file only | 13e76bdc8d49ed14d710fac7e3b61186cf74c8d3 | 1,673 |
| Public Domain Review, essays file only | e9c7669206b95871601fe4672a484d8215e6fe6f | 266 |

These are candidate pools of edited contemporary prose, not a guarantee that
all text is acceptable. Each row retains its source URL and source-specific
license (360info CC BY 4.0; PDR CC BY-SA 4.0). The downloader checks the expected
host, source label and captured license string; records immutable commits and
compressed-file SHA-256 receipts; and supports offline reuse.[^news][^pdr]

# Filters

The pipeline selects unchanged prose lines, excluding short/oversize paragraphs,
metadata, headings, exercise prompts, markup and boilerplate. Parsed sentences
must have 8–45 alphabetic words and 40–400 characters, terminal punctuation, an
overt main subject and finite verbal root. Direct quotations, ellipses, known
misspellings, obvious determiner mismatches and unknown non-name vocabulary are
excluded. Named entities and known inflected lemmas are permitted. Parser and
vocabulary decisions can be wrong and bias the retained text toward easy cases.

A local LanguageTool check is required by default.[^lt] The original must have no
relevant grammar, spelling or typographical diagnostics in at least one of US
and British English, excluding dictionary complaints wholly inside recognized
named entities. Each injected spelling edit must receive an exact-span spelling
diagnostic; grammar edits require overlapping grammar/typographical diagnostics.
Style suggestions do not determine acceptability. Detailed evidence is stored
per pair. Missing diagnostics cause abstention, never automatic acceptance.

Initial agent spot-checks found three malformed or suspect originals that LanguageTool missed. A subsequent 70-pair stratified inspection of the productive build found three more source flags (two rejected, one uncertain).
`config/english_source_exclusions.json` excludes their exact sentence hashes,
recording source URLs and reasons. The manifest fingerprints this list. This
small exclusion list is an audit trail, not a substitute for broader review.

# Isolation and deduplication

Assign canonical URL document IDs to train/validation/test by a seeded 80/10/10
hash before selecting sentences. Remove exact normalized originals and near
duplicates sharing a token trigram with at least 90% token-sequence similarity.
Also prohibit exact duplicate originals/corruptions across all exported pairs.
The near-duplicate heuristic is not semantic deduplication and does not guarantee
that all related articles are grouped together.

Pressbooks and LibreTexts remain possible future curated sources. They are not
enabled by default: inspected samples contained exercises, tables and navigation.

[^news]: Common Pile news dataset; only the pinned 360info shard is used.
[^pdr]: Common Pile Public Domain Review dataset; only the essays shard is used.

[^lt]: Official instructions for running LanguageTool locally.
