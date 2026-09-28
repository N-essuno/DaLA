---
type: Release
title: DaLA Dutch DynaWord Hugging Face release
description: Audited four-source Dutch corpus with two task configurations and explicit source-quality limitations.
status: stable
generated: {by: codex/gpt-6, at: 2026-09-22T07:48:00+00:00}
sources:
  - id: release
    resource: https://huggingface.co/datasets/schneiderkamplab/dala-dutch-dynaword
    title: Authorized Dutch dataset destination
  - id: extension
    resource: ./dutch-legal-extension.md
    title: Legal-source investigation and full-size extension
---
# Authorization and release selection

The user explicitly requested **audit and then upload**, continuing the established
`schneiderkamplab` organization and two-task export convention. Destination:
`schneiderkamplab/dala-dutch-dynaword`.[^release]

The extension reached **478,930 pairs** after generating a 320,000-pair new-source
reserve, merging with the audited base, cross-source deduplication and document
split preservation. All 478,930 originals were recovered from pinned source texts;
all 241,701 spelling edits passed the independent OpenTaal checks. All exported
artifacts, labels, views, edit round trips and provenance checks passed.[^extension]

# Final linguistic audit

The pre-release uniform sample contains 200 previously unreviewed pairs, drawn
from 478,080 eligible originals after excluding 850 earlier-reviewed originals.
Agent inspection found **187 acceptable, ten erroneous and three uncertain
sources**. All **200 injected edits** were judged valid. A separate 25-pair family
supplement found 24 acceptable sources and one uncertain; all edits were valid.
These are agent judgments, not native-speaker gold, and do not imply perfect
corruption precision or a human-validated evaluation set.

Errors include merged headings, duplicated function words, broken compounds,
missing articles, malformed name substitutions and wrong participle forms.
The **14 flagged pairs** were removed in a separate release export without
rewriting or replacing any retained records. The frozen measured parent remains
unchanged. Removing reviewed errors does not independently establish a new
precision rate for the release. Source-label errors can remain elsewhere.

Audits, judgments, source exclusions and mechanical receipts:
`wiki/artifacts/dutch-extended-assessment/`. The final corpus is
`la_output/dutch_dynaword_release/` with **478,916 pairs**, **957,832 rows per task**.
Split pairs: train 383,131; validation 47,892; test 47,893. Manifest SHA256:
`41fb35b5acf5e40e7149f58524c80a2f23ceeabc297e5c6a55aa815d337b6436`.

| Source | Release pairs |
| --- | ---: |
| Rechtspraak | 240,251 |
| EUR-Lex | 107,033 |
| Government web prose | 81,134 |
| Officiële bekendmakingen | 50,498 |

There are 64 active rules and 31,616 distinct case-folded surface substitutions.
Spelling and article changes dominate. Relative-pronoun and d/dt families have
only 45 and 44 examples. Source and corruption distributions are synthetic
selection outcomes, not natural error frequencies.

# Package and verification

Staging: `export-upload/dala-dutch-dynaword/`.
Builder: `scripts/prepare_dutch_hf_dataset.py`; uploader/verifier:
`scripts/upload_dutch_release.py`. Reuses the English release's deterministic
chat exporter/standalone validator, `scripts/hf_package_runtime.py`.

The `acceptability` and `correction` configurations have explicit YAML shard
assignments. Each pair contributes a clean control and a corrupted input.
Chat metadata exposes labels and is not model input. Canonical pairs, exact
source/edit offsets, checker evidence, audit decisions, exclusions and parent
manifest lineage are bundled separately from task shards.

The card discloses the pre-exclusion audit and residual source-label risk.
Source-card licenses remain CC0 for government, Rechtspraak and announcements,
and CC BY 4.0 for EUR-Lex. Original article URLs/authors are unavailable; dataset
file coordinates, upstream IDs and hashes are retained without inventing them.

Published publicly at [schneiderkamplab/dala-dutch-dynaword](https://huggingface.co/datasets/schneiderkamplab/dala-dutch-dynaword).
Immutable revision: `a09dffe55c9db956f3ae39ed1916cb2d5dfaff73`.
All 44 staged files matched the remote file set, sizes and Git/LFS hashes.
Both configurations loaded completely with the datasets library, and both loaded
remotely in streaming mode at that exact revision. Full loader split counts are
766,262 / 95,784 / 95,786 rows per configuration. Package validation confirms
957,832 rows per task, exact chat reconstruction, edit round trips and provenance.

Upload receipt: `wiki/artifacts/dutch-hf-upload.json`; full local loader receipt:
`wiki/artifacts/dutch-hf-loader-validation.json`. Package manifest SHA256:
`45ec3f00dd3202db5e8c7c8dc81001d69d4f0dda134f1f14fe8ae6b4b7945c1a`. The shared wiki validator now excludes archived upstream
cards under `artifacts/` from authored-concept checks; all 21 OKF concepts and
bundle links validate.

[^release]: User-authorized publication; credentials are not recorded.
[^extension]: The completed extension retains source provenance and unchanged source excerpts.
