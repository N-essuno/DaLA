# Frozen Danish equivalence oracle

These Python files were copied from the working tree immediately before the
language-pack refactor. They are test fixtures, never used by the production
pipeline. Do not modernize or fix them: the comparison deliberately detects
changes to rule order, morphology conditions, RNG consumption, reconstruction,
splitting and output rows. `wiki/artifacts/danish-equivalence.json` fingerprints
the files and the actual source snapshot used by the full comparison.

Run `python -m scripts.check_danish_equivalence` from the repository root after
installing `da_core_news_md` 3.8.0. Both engines receive identical real parser
annotations. Default rule probability, both split modes, and two RNG seeds are
covered. Exact agreement does not establish linguistic correctness of the
historical Danish corruptions.
