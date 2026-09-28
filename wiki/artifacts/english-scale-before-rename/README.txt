# English DaLA — TV2R scale — grammar and spelling

478,930 original/corrupted pairs. Status: **checker_screened**.
No human linguistic precision measurement is implied. Checker-screened builds
require no relevant source diagnostics and a diagnostic at every injected edit.

Pinned Common Pile 360info, Public Domain Review essays and Global Voices long-form articles. Source URLs and licenses accompany each document; supplied author metadata is preserved, with missing values left null. Global Voices publisher states CC BY 3.0; captured Common Pile metadata says BY 4.0 and is preserved separately. Expanded sources have known checker-undetected errors. Agent review of 130 pre-curation pairs accepted all injected edits, but flagged 18 originals for grammar/spelling uncertainty or errors and four as boilerplate. Documents with grammar/spelling flags, boilerplate and ambiguous embedded line separators are excluded; remaining source correctness is not established by this review. This is a provisional checker-screened corpus, not a human-validated benchmark.

Each split provides canonical pairs, raw acceptability CSV/JSONL and instruction
acceptability/correction JSONL. Both tasks include clean controls. Source URLs,
licenses, revisions, offsets and error evidence accompany the data. Rule and
word frequencies are not natural error-frequency estimates.

Documents receive seeded train/validation/test assignments before selection.
Exact and heuristic near-duplicate filters reduce leakage. Up to
2 independent edits are permitted, at most one grammar
edit. Correct input maps to itself in correction. No paraphrasing or style task
is included. `rules.json` distinguishes attested lexical substitutions from
productive rules supported by observed error families.

`review.csv` samples by split/family/error count. The review command requires
explicit positive judgments for the source, corruption and intended edits.
See `manifest.json` for resources, counts and checksums. Automatic screening can
miss errors; it is not human validation.
