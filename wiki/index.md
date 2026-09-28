# DaLA knowledge bundle

* [Six-language expansion](pages/six-language-expansion.md) - Separate Polish, Swedish, Bokmål, Nynorsk, Faroese and Icelandic datasets; independent size and coverage goals.

Open Knowledge Format 0.2. Start with the contract, then the runbook.

* [Bundle conventions](schema.md) - OKF structure, provenance, trust and maintenance.
* [English dataset contract](pages/english-contract.md) - Accepted scope, task views and quality constraints.
* [Error evidence](pages/error-evidence.md) - Observed corrections and carefully bounded generalization.
* [English spelling](pages/english-spelling.md) - Expanded sourced spellings, screening and random-corruption distinctions.
* [English generators](pages/english-generators.md) - Productive spelling and evaluated deletion/swap fallback.
* [DaLA English — Common Pile](pages/english-scale.md) - Expanded sources, resumable construction and scale assessment.
* [English–Danish quality comparison](pages/language-quality-comparison.md) - Fresh 100-pair samples, source errors, corruption ambiguity and release implications.
* [Source selection](pages/source-selection.md) - Pinned Common Pile subsets and source filtering.
* [Dataset runbook](pages/dataset-runbook.md) - Reproduce, audit and export the dataset.
* [Pipeline architecture](pages/pipeline-architecture.md) - Language packs, Danish equivalence and productive English rules.
* [Multilingual assessment](pages/multilingual-assessment.md) - Danish equivalence and final English amount, diversity and quality.
* [Build results](pages/build-results.md) - Actual runs, checks and remaining quality limitations.
* [TV2R and prior work](pages/tv2r-and-history.md) - Inspected reference datasets and superseded EWT pilot.
* [Update log](log.md) - Dated project decisions and implementation history.

* [English HF release](pages/hugging-face-release.md) - Published configurations, provenance and verification.
* [Dutch DynaWord](pages/dutch-dynaword.md) - Selected next language and implementation evidence.
* [Dutch scale readiness](pages/dutch-scale-readiness.md) - Cleaner source selection and expanded, tested Dutch corruption coverage.
* [Larger Dutch validation](pages/dutch-larger-validation.md) - Stricter sources, 1,000-document construction, family caps and a fresh audit.
* [Uncapped Dutch generation](pages/dutch-uncapped.md) - Active production profile: all eligible paragraphs, lower article priority, no percentage downsampling.
* [Full-source Dutch build](pages/dutch-full-scale.md) - Completed 188k-pair build, independent checks, fresh quality audit and coverage limits.

* [Dutch legal-source extension](pages/dutch-legal-extension.md) - Rechtspraak and Officiële bekendmakingen investigation, pilot and 479k-pair target.

* [Dutch HF release](pages/dutch-hf-release.md) - Final audit, source exclusions, two-task package and publication verification.

* [European language priorities](pages/european-language-ranking.md) - Ranked grammar/spelling evidence, resource access, and next-language readiness.
* [Twelve European pilots](pages/european-pilots.md) - Shared pipeline, language-isolated packs, 9,346 candidate pairs, measured coverage and scale-up gates.

* [European CPU expansion](pages/european-cpu-expansion.md) - Separate source and lexical inputs, training-only error evidence, exact-offset parsing and independent diagnostics.

* [European coverage preparation](pages/european-coverage-preparation.md) - Shared grammar fixes, targeted CPU pilots, full source inputs, and explicit linguistic review gaps.

* [European contraction recovery](pages/european-mwt-recovery.md) - Shared parser implementation, paired yield tests, checker rejection audit and isolated validation.

* [European campaign upgrade](pages/european-campaign-upgrade.md) - Active v2 campaign, early filtering, checker pooling and provenance-preserving checkpoint continuation.

* [European orthography investigation](pages/european-orthography-investigation.md) - German/Estonian export failures, Greek dictionary normalization bug, and checked upload status.

* [Checker resilience](pages/checker-resilience.md) - Retried checks, audited sentence rejection, outage guards and Ukrainian 128-parser continuation.

* [European pair audit](pages/european-pair-audit.md) - Full canonical-pair audit using the existing eight Gemma servers; automated signals and review limits.

* [Audited release integration](pages/audited-release-integration.md) - HF upload preparation and train-only DFM12 integration of audit-passed pairs.
