---
type: Dataset Design
title: English dataset contract
description: Grammar and spelling on curated real text, with acceptability and correction
  views.
status: stable
generated:
  by: codex/gpt-6
  at: '2026-09-21T15:13:36+00:00'
sources:
- id: session
  resource: Project-owner instructions in this DaLA multilingual implementation conversation
  title: Accepted English dataset requirements
---
# Scope

The user requested multilingual DaLA starting with English, empirical correction
patterns generalized carefully, high-quality Common Pile source sentences, grammar
and spelling errors, and both useful tasks illustrated by TV2R. Simplification,
paraphrasing and style transfer are excluded. English targets contemporary standard
written usage; UK and US spelling variants must not be treated as errors merely
for differing from each other.

# Decisions

- Preserve clean source sentences verbatim. Store document, sentence and edit offsets.
- Mine edits from multi-error sentences; isolate edits during rule calibration only.
- Use attested lexical spelling substitutions, observed spelling mechanisms generalized with lexical guards, and productive grammatical rules supported by observed correction families. Token fallback requires separately evaluated English syntax guards.
- Multi-error generation allows one grammar edit plus independent spelling edits.
  Never combine subject and verb changes that could restore agreement.
- Produce acceptability and correction instruction views. Both include clean controls;
  correction controls must return unchanged input. Prompts mention grammar and spelling.
- All variants and task views stay in the split assigned to their original document.
- Retain provenance and diagnose each error family independently. Automatic filtering
  and LanguageTool screening are not a measured human precision result.

# Canonical schema

Each pair has `pair_id`, `document_id`, source name/dataset/revision/file/line,
upstream ID, URL, license, document SHA-256, sentence start/end, language, split,
`original`, `corrupted`, `edits`, quality status, and checker diagnostics. Each edit
has an evidence-backed rule ID, type, original/replacement strings, original and
corrupted spans, and protected parser-token IDs. Those token IDs are local to the
parsed paragraph and are diagnostic only; character offsets are canonical.

Instruction views use `direction` plus `samples.content` and `samples.response`,
with metadata outside the prompt. Single errors retain affected-token compatibility;
multiple errors use the full edits list. See [runbook](/pages/dataset-runbook.md).

# History

The initial EWT pilot had six grammar rules and single-span generation. Its use
as the production source and its grammar-only scope were superseded by this
contract. It remains explicitly accessible as `--source ewt` for regression work.
