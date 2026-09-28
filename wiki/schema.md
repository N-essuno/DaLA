---
type: Reference
title: Bundle conventions
description: OKF 0.2 structure, provenance, trust and maintenance for DaLA.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-21T15:13:36+00:00}
sources:
  - id: okf
    resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
    title: Open Knowledge Format version 0.2
---
# Conventions

This `wiki/` directory is an OKF 0.2 bundle. Concept files have YAML frontmatter
with `type`, `title`, `description`, `status`, and `generated`. `index.md` and
`log.md` are reserved navigation/history files, not concepts. Concept IDs are
bundle paths without `.md`; internal links use bundle-relative or relative paths.
External evidence belongs in `sources` with stable IDs and keyed footnotes.[^okf]

Use `draft`, `stable`, or `deprecated` for lifecycle. Keep superseded findings
with explicit context. Timestamps use ISO 8601 with a UTC offset. Do not claim
human review for agent judgments. `verified` is reserved for actual verification
events; a successful code test does not verify linguistic precision. Build
receipts remain with generated data; record their path/checksum and substantive
findings in the wiki. Update this bundle and its log when durable decisions,
commands, outputs or limitations change.

[^okf]: The requested OKF 0.2 specification, read before creating this bundle.
