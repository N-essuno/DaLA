---
type: Release
title: DaLA English Common Pile Hugging Face release
description: Published two-configuration English dataset with frozen provenance and disclosed source-label errors.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-21T20:42:54+00:00}
sources:
  - id: release
    resource: https://huggingface.co/datasets/schneiderkamplab/dala-english-common-pile
    title: Public dataset repository
  - id: configs
    resource: https://huggingface.co/docs/hub/datasets-manual-configuration
    title: Hugging Face manual dataset configurations
---
# Published artifact

The user explicitly authorized upload after seeing the fresh quality audit.
Published publicly as `schneiderkamplab/dala-english-common-pile`, with
`acceptability` and `correction` configurations.[^release] Each contains 957,860
chat rows over 478,930 shared original/corrupted pairs. Splits preserve the
existing 766,288 / 47,892 / 143,680 rows per task. The release does not remove
the latest sample's source errors, claim human validation, or combine both
configurations as independent data.

The card names the 80 accepted / 11 erroneous / 9 uncertain source judgments,
spelling-heavy distribution, source concentration and mixed source licenses.
Canonical pairs, edit spans, checker diagnostics, pinned source metadata,
per-row attribution and fresh English audit decisions are bundled separately
from training shards. Chat metadata exposes labels and is not model input.

Staged at `export-upload/dala-english-common-pile`. Builder:
`scripts/prepare_hf_dataset.py`; standalone validator/recreator:
`scripts/hf_package_runtime.py`, copied as `recreate_dataset.py`. Deterministic
gzip shards are explicitly assigned by YAML configuration.[^configs]

# Verification

Full mechanical validation checked all pairs/rows, forward and inverse edits,
document splits, provenance and file checksums. Every chat instruction, input,
answer, pair ID and row position matches the prior instruction exports. Both
configurations loaded in full with the Hugging Face datasets library. A fixture
test verifies exact recreation and rejection of damaged artifacts.

Post-upload verification matched the complete remote file set, sizes and every
Git/LFS hash. Both task configurations then loaded remotely in streaming mode.
The final immutable revision and package manifest checksum are recorded in
[upload receipt](../artifacts/english-hf-upload.json); local full-loader counts
are in [loader receipt](../artifacts/english-hf-loader-validation.json).
The standalone validator permits the Hub-generated `.gitattributes` file.

[^release]: User-authorized publication; no credentials are recorded.
[^configs]: Multiple task views in one repository, without custom loading code.
