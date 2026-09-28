---
license: cc-by-sa-4.0
language:
- pl
pretty_name: Polish DynaWord
task_categories:
- text-generation
size_categories:
- 1M<n<10M
tags:
- polish
- pretraining
- dynaword
---

# Polish DynaWord

A continuously developed, **openly-licensed**, human-text Polish corpus — a Polish
edition in the [Dynaword](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword)
family (Enevoldsen et al., [arXiv:2508.02271](https://arxiv.org/abs/2508.02271)).

> **v0.2.5 stable** · 4,319,200 documents · **9.64B tokens**
> (tiktoken proxy; canonical Llama-3 count at release) · 18 sources
> Updated: **2026-08-14**

> **v0.3-dev experimental track** · quality/diversity workflow, source-gate
> validation and candidate-data audits. This is development work, not a released
> corpus version, and it does not replace the v0.2.5 stable parquets.

> **Europeana validation artifact (2026-07-10)** · a separate reproducible
> 810-document direct-ingestion sample stored under the legacy path
> `previews/v0.3.1/europeana.parquet`. It is not a full dataset release.

## Releases

### Stable releases

| version | status | documents | tokens | notes |
|---|---|---:|---:|---|
| `v0.2.5` | stable release | 4,319,200 | 9.64B | Adds community-contributed `samorzad_gov_pl` with institution-level author attribution and a URL-bearing attribution sidecar. |
| `v0.2.4` | previous stable | 4,246,429 | 9.60B | Expanded `european_hplt_v3_pl` to WDS bins 10-5 (+1.49M docs / 2.04B tok); phone PII-scrub hardening (independent dual-lens) + cross-source dedup (zero-overlap). |
| `v0.2.3` | previous stable | 2,710,974 | 6.88B | Adds community-contributed `european_hplt_v3_pl`, `global_voices` and `nkjp1m`. |
| `v0.2.2` | previous stable | 2,579,963 | 6.36B | Added community-contributed `govpl`: 88,190 docs / 80.7M tokens. |
| `v0.2.1` | previous stable | 2,491,773 | 6.28B | 12-source corpus with `license` and `author` metadata columns; added `1000_novels`. |
| `v0.2.0` | previous stable | 2,490,773 | 6.22B | Provenance-first corpus from 11 open/official sources. |

![Stable dataset size by version](artifacts/dataset_size_by_version.png)

### Development and validation artifacts

| name | type | scope | notes |
|---|---|---|---|
| `v0.3-dev` | experimental development track | workflow and candidate audits | Quality remix, legal-source caps, source QA, deduplication and direct-upstream ingestion. Not a corpus release. |
| `Europeana validation artifact 2026-07-10` | reproducible sample | 810 documents / 140,042 tokens | Stored at the legacy path `previews/v0.3.1/europeana.parquet`; preserves per-record rights and creator metadata. Not a corpus release. |

### Version details

#### v0.2.5 — current stable

Added `samorzad_gov_pl`: **72,771 documents / 41,154,506 cl100k-proxy
tokens** from 246 Polish local public institutions and one central platform
tenant. The contribution preserves the publishing institution as `author`,
discovered and fetched article URLs in an attribution sidecar and the verified
platform-wide CC-BY-SA-4.0 basis.

#### v0.2.4 — previous stable

Expanded `european_hplt_v3_pl` from WDS bins 10+9 to **WDS bins 10-5** (added
quality bins 8/7/6/5 and further WDS-9 shards): **+1,493,384 documents /
2,043,141,874 cl100k-proxy tokens**, 100% CC0-1.0. Includes a phone PII-scrub
hardening pass (v12b) verified by two independent lenses (a systematic
recall-gap masked by pipeline self-report was caught pre-release), and
cross-source deduplication against the existing WDS-10+9 shards (exact-dup 0,
near-dup overlap 0.0000%, byte-verified) — resolving the "downstream near-dedup
remains" caveat from the original release.

#### v0.2.3 — previous stable

Community expansion release. Adds `european_hplt_v3_pl` from
[PR #5](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/5)
and `global_voices` from
[PR #7](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/7),
plus `nkjp1m` from
[PR #8](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/8).
The complete 16-source release contains **2,710,974 documents /
6,881,277,362 cl100k-proxy tokens**, summed from the released per-source parquet
statistics.

#### v0.2.2 — previous stable

Added `govpl` from
[PR #9](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/9):
88,190 Polish government press releases collected directly from 133 gov.pl
ministry and agency subsites.

#### v0.2.1 — previous stable

Introduced the canonical eight-column release schema:
`id, text, source, added, created, token_count, license, author`. Added
`1000_novels` from
[PR #1](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/1)
and recomputed all release statistics from parquet files.

#### v0.2.0 — initial stable corpus

The provenance-first baseline: 11 reviewed open or official sources,
2,490,773 documents and 6.22B proxy tokens.

#### v0.3-dev — experimental quality workflow

Candidate-only work on a better-balanced training mixture: legal-source caps,
temperature sampling, source QA, deduplication and direct-upstream ingestion.
It is not a replacement for the stable corpus.

#### Europeana validation artifact — 2026-07-10

A separate reproducible 810-document Europeana sample preserving per-record
rights and creator metadata. It does not change stable-release totals.
`v0.3.1-preview` remains only as the legacy storage-path label.

## What this dataset contributes
The raw texts come from existing open corpora (redistributed via SpeakLeash and,
where applicable, fetched from upstream). **The value added here is the curation,
not the bytes**, following the Dynaword methodology:

1. **License review per source** — each source vetted for an *openly-licensed,
   traceable* legal basis (documented in its datasheet); sources that fail the
   review are **excluded with a stated reason** (see table below), not silently
   kept. This is the core editorial work.
2. **Filtering & normalization** — minimal, reproducible gates (short-doc,
   non-Polish, exact cross-source dedup, OCR garble) applied uniformly to one
   clean schema: `id, text, source, added, created, token_count, license, author`.
3. **Documentation** — a datasheet per source (Gebru et al. 2021) + this card,
   so provenance and licensing are auditable rather than assumed.
4. **Reproducibility & versioning** — `src/` rebuilds the corpus from sources;
   new sources and removals are tracked in the CHANGELOG.

Credit for the underlying texts belongs to the upstream sources and to SpeakLeash
as the redistributing aggregator; this release does not claim ownership of them
(see Disclaimer).

## Contributors

| contributor | contribution | release / PR |
|---|---|---|
| [Kacper Wikieł](https://huggingface.co/kacperwikiel) | Project maintainer; corpus curation, source and license review, release engineering, documentation, validation and reproducible build workflow. | all releases |
| [Bart Kobyliński](https://huggingface.co/bartoszkobylinski1) | Added `1000_novels`; expanded Biblioteka Nauki and Europeana ingestion with per-document license and author metadata. | `v0.2.1`, [PR #1](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/1) |
| [Paweł Puzio](https://huggingface.co/ppuzio) | Built `govpl`: subsite discovery, direct ingestion pipeline, dataset artifact, contract tests and documentation. | `v0.2.2`, [PR #9](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/9) |
| [Arkadiusz Słota](https://huggingface.co/Maggio33) | Built & expanded `european_hplt_v3_pl`: HPLT v3 WDS 10+9 (PR #5) then WDS 10-5 expansion (+1,493,384 docs / 2.04B tok), modular cleaning pipeline, phone-recall PII-scrub (v12b, independent dual-lens), cross-source dedup-vs-base (zero-overlap), validation and documentation. | `v0.2.3` (PR #5), `v0.2.4` |
| [Dawid Majewski](https://huggingface.co/dawidmajewski) | Added the reviewed Global Voices Polish corpus; built `samorzad_gov_pl` with direct acquisition, institution attribution, licensing evidence, release artifact and reproducible conversion. | `v0.2.3`, [PR #7](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/7); `v0.2.5` |
| [1am](https://huggingface.co/1am) | Added `nkjp1m`: the manually annotated 1-million-word NKJP subcorpus, direct fetch/build pipeline, source documentation and release artifact. | `v0.2.3`, [PR #8](https://huggingface.co/datasets/SlayerLab/polish-dynaword/discussions/8) |

Contributions are credited when they add a verifiable dataset artifact, pipeline,
validation, documentation or release work. A merged discussion with no resulting
files is not listed as a data contribution.

## Guiding principles
1. **Open & traceable licensing** — every source is *openly licensed* with a documented
   legal basis (see each datasheet's "traceable basis"), not a vague "public domain".
2. **Reproducibility** — `src/build_dynaword.py` rebuilds the corpus from sources.
3. **Documented** — a datasheet per source under `data/<source>/`.
4. **Extensibility** — versioned; new sources via PR.

## Sources
| source | description | license | documents | tokens |
|---|---|---|---:|---:|
| [european_hplt_v3_pl](data/european_hplt_v3_pl/european_hplt_v3_pl.md) | HPLT v3.0 Polish (web, top WDS bins 10-5) | `CC0-1.0` | 1,603,974 | 2,555.5M |
| [eurlex](data/eurlex/eurlex.md) | EUR-Lex (EU legal acts, Polish) | `CC-BY-4.0` | 243,060 | 2,378.1M |
| [parliamentary](data/parliamentary/parliamentary.md) | Polish Parliamentary Corpus (Sejm/Senat) | `public-domain (official documents)` | 324,622 | 1,646.8M |
| [wikisource](data/wikisource/wikisource.md) | Polish Wikisource | `CC-BY-SA-3.0` | 632,005 | 801.9M |
| [wikipedia](data/wikipedia/wikipedia.md) | Polish Wikipedia | `CC-BY-SA-3.0` | 1,171,897 | 707.2M |
| [biblioteka_nauki](data/biblioteka_nauki/biblioteka_nauki.md) | Biblioteka Nauki | `per-record upstream license` | 42,071 | 673.5M |
| [dziennik_ustaw](data/dziennik_ustaw/dziennik_ustaw.md) | Dziennik Ustaw + Monitor Polski (Polish primary legislation) | `public-domain (official documents)` | 35,442 | 486.1M |
| [wolne_lektury](data/wolne_lektury/wolne_lektury.md) | Wolne Lektury (school readings) | `CC-BY-SA-4.0 / Wolna Sztuka 1.3` | 6,141 | 103.0M |
| [govpl](data/govpl/govpl.md) | gov.pl — Polish government press releases | `CC-BY-SA-4.0` | 88,190 | 80.7M |
| [1000_novels](data/1000_novels/1000_novels.md) | 1000 Novels Corpus (CLARIN-PL) | `CC-BY-4.0` | 1,000 | 60.5M |
| [samorzad_gov_pl](data/samorzad_gov_pl/samorzad_gov_pl.md) | samorzad.gov.pl — Polish public-sector institutions | `CC-BY-SA-4.0` | 72,771 | 41.2M |
| [wikiquote](data/wikiquote/wikiquote.md) | Polish Wikiquote (quotations) | `CC-BY-SA-3.0` | 30,363 | 31.9M |
| [eltec_pol](data/eltec_pol/eltec_pol.md) | ELTeC-pol (European Literary Text Collection, Polish) | `CC-BY-4.0` | 100 | 21.5M |
| [wikivoyage](data/wikivoyage/wikivoyage.md) | Polish Wikivoyage (travel guides) | `CC-BY-SA-3.0` | 13,645 | 17.1M |
| [wikibooks](data/wikibooks/wikibooks.md) | Polish Wikibooks (open textbooks) | `CC-BY-SA-3.0` | 9,112 | 15.6M |
| [wikinews](data/wikinews/wikinews.md) | Polish Wikinews | `CC-BY-2.5` | 24,386 | 12.1M |
| [global_voices](data/global_voices/global_voices.md) | Global Voices Polish | `CC-BY-3.0` | 2,040 | 3.7M |
| [nkjp1m](data/nkjp1m/nkjp1m.md) | The manually annotated 1-million word subcorpus of the National Corpus of Polish | `CC-BY` | 18,381 | 2.6M |
| **total** | | | **4,319,200** | **9,639.1M** |

## Sources on main, not yet released
Built and documented, but **not** part of v0.2.5 and excluded from every total above. They join a release when a CHANGELOG entry admits them.

| source | description | license | documents | tokens |
|---|---|---|---:|---:|
| [sejm_api](data/sejm_api/sejm_api.md) | Sejm API parliamentary speeches (2023 onward) | `public-domain (official documents)` | 38,812 | 36.3M |
| [wiktionary_examples](data/wiktionary_examples/wiktionary_examples.md) | Polish Wiktionary usage examples | `CC-BY-SA-3.0` | 8,639 | 1.1M |

## Method
Only **human-authored** text — no synthetic, machine-translated, or auto-transcribed
data. Gates are intentionally minimal (drop short docs, non-Polish, exact duplicates,
OCR garble); heavy quality filtering and mix-weighting are left to downstream training.
Evaluation-set decontamination is applied/marked separately. Schema:
`id, text, source, added, created, token_count, license, author`. The `license`
and `author` columns are per-document metadata when upstream exposes them; older
sources use the source-level license and an empty author field.

## v0.3 quality roadmap and current status

The v0.2.x raw corpus is intentionally provenance-first, but its token mix is too
heavy in legal/parliamentary language for natural general pretraining. The v0.3
workflow therefore separates **source inclusion** from **training mix**:

- cap `eurlex + parliamentary + dziennik_ustaw` to roughly **10-20%** of training
  tokens combined;
- use source-level temperature sampling (`sqrt`, alpha `0.5`) instead of raw
  token-proportional sampling;
- add traceably licensed contemporary/natural Polish: open web, academic prose,
  cultural heritage text, guides, technical documentation/blogs, Q&A, and
  dialogue/instruction data;
- run aggressive exact, normalized, and near-duplicate removal;
- reserve the final **5-15%** of training for higher-quality sources rather than
  the largest sources;
- evaluate per-source perplexity and style contamination, not only global loss.

Current v0.3 source-ingestion status:

- `biblioteka_nauki`: included from the direct-upstream rebuild with
  per-document license and author metadata.
- `europeana`: prepared in the source registry as a direct-upstream rebuild
  target with per-record rights statements and creator metadata; raw SpeakLeash
  Europeana remains excluded.
- Europeana release policy: split conservatively at pre-1929 records for
  US-sensitive downstream reuse, and keep later/unknown records separately
  labeled or held until legal review.
- `european_hplt_v3_pl`: WDS bins 10-5 are included in v0.2.5 after contract
  validation. HPLT packaging is CC0, while underlying crawled web documents can
  carry independent rights; downstream near-dedup and web-content review remain
  recommended before training.

Current review artifacts:

- `configs/source_candidates_v0_3.json` — candidate decisions and license policy.
- `artifacts/source_license_review_v0_3.md` — source-by-source license review.
- `artifacts/source_candidate_audit_v0_3.md` — generated Hugging Face metadata audit.
- `artifacts/training_mix_v0_3.md` — example 1B-token training mix with legal sources capped at 15%.
- `artifacts/bartek_source_ingestion_plan_2026-07-02.md` — PR contract for
  Biblioteka Nauki and Europeana ingestion.

## Excluded sources (transparency)
Sources we reviewed and **deliberately left out** — part of the curation:

| source | reason |
|---|---|
| `open_subtitles_corpus` | Derivative of copyrighted film/TV dialogue; OpenSubtitles uploads largely unlicensed. Same copyright lesson as Danish Gigaword's OpenSubtitles (paper 2508.02271). Not openly licensed. |
| `europeana_eu_pl_corpus_raw_speakleash` | Aggregated items with mixed per-record rights (PD / CC-BY-NC / rights-reserved). The raw SpeakLeash redistribution is excluded; only a direct rebuild preserving per-record rights metadata may be included. |
| `1000_novels_corpus_CLARIN-PL` | CC-BY-4.0 label, but 'novels' likely include in-copyright contemporary works; verify titles/years on CLARIN handle 11321/312 before inclusion. |
| `project_gutenberg_pl_corpus` | Only 31 PL books (4.3MB) — PG is ~99% English; Polish PD literature already covered by wolne_lektury + wikisource (so near-redundant after dedup). Dropped to avoid the PD-in-EU per-work check (PG claims PD-in-US only) for negligible token gain. |

## Personal & sensitive data
This corpus contains **only** text that its upstream sources already published
under open licenses or as official public-domain record. It therefore includes
names and statements of **public figures acting in a public capacity** — e.g.
parliamentary speakers (PPC), authorities named in legal acts (EUR-Lex), and
people described in encyclopedic articles (Wikipedia/Wikisource). No private,
non-public personal data was collected or added. If you are a data subject and
want content concerning you removed, contact **k.wikiel@gmail.com** — it will be dropped
from the next version (see retroactive-removal policy below).

## Disclaimer & legal
- **Provenance in good faith.** Per-source licenses are reproduced *as documented
  by the upstream sources and by SpeakLeash* (the intermediate aggregator), to the
  best of our knowledge. We make no independent legal warranty about the copyright
  status of any individual document.
- **No ownership claim.** This release is a *curated, license-reviewed, documented
  aggregation*. We claim no ownership of the underlying texts; rights remain with
  the original authors/rightsholders under their respective licenses.
- **Provided "as is"**, without warranty of any kind, express or implied. This is
  not legal advice.
- **Your compliance is yours.** Downstream users must satisfy each upstream
  license themselves — in particular **CC-BY-SA-4.0 attribution and share-alike**
  for derivatives of this dataset, and attribution to the upstream sources and to
  SpeakLeash.
- **Notice-and-takedown.** Any source or rightsholder raising a substantiated
  objection can have material removed: contact **k.wikiel@gmail.com**; it is dropped from
  the next version and recorded in the CHANGELOG. Removal is retroactive
  going-forward (prior immutable snapshots/commits may persist).

## License & attribution
Released under **CC-BY-SA-4.0** (copyleft inherited from CC-BY-SA sources such as
Wikipedia/Wikisource/Wolne Lektury). Attribution due to each upstream (see datasheets)
and to **SpeakLeash** as the intermediate aggregator. Retroactive-removal policy: a
source that raises an objection is dropped from subsequent versions, recorded in the
CHANGELOG.

## Reproduce
```bash
python3 src/build_dynaword.py --all --speakleash-dir <speakleash_zst_dir> --out .
python3 src/make_docs.py
```

## Results

### Corpus phrase frequency (normalized by tokens)

Raw counts and token-normalized shares are regenerated from the current Parquet files with `src/pattern_frequency_report.py`.

- Total token count (tiktoken proxy): **9,639,062,573**

| Pattern | Count | Share of all tokens |
|---|---:|---:|
| `w roku` | 585,072 | 0.0061% |
| `klasyfikacji` | 164,757 | 0.0017% |
| `ustawa` | 659,103 | 0.0068% |
| `artykuł` | 2,416,361 | 0.0251% |
| `parlament` | 1,285,792 | 0.0133% |
| `rozporządzenie` | 1,527,456 | 0.0158% |
| `w pobliżu` | 127,784 | 0.0013% |
| `mieszkańców` | 476,710 | 0.0049% |
| `Dz.U.` | 987,048 | 0.0102% |

### Per-source shares

| source | pattern | count | share of source tokens |
|---|---|---:|---:|
| `1000_novels` | `w roku` | 659 | 0.00109% |
| `1000_novels` | `klasyfikacji` | 14 | 0.00002% |
| `1000_novels` | `ustawa` | 656 | 0.00108% |
| `1000_novels` | `artykuł` | 729 | 0.00120% |
| `1000_novels` | `parlament` | 240 | 0.00040% |
| `1000_novels` | `rozporządzenie` | 65 | 0.00011% |
| `1000_novels` | `w pobliżu` | 1,262 | 0.00209% |
| `1000_novels` | `mieszkańców` | 868 | 0.00143% |
| `1000_novels` | `Dz.U.` | 0 | 0.00000% |
| `biblioteka_nauki` | `w roku` | 28,728 | 0.00427% |
| `biblioteka_nauki` | `klasyfikacji` | 9,045 | 0.00134% |
| `biblioteka_nauki` | `ustawa` | 48,791 | 0.00724% |
| `biblioteka_nauki` | `artykuł` | 91,241 | 0.01355% |
| `biblioteka_nauki` | `parlament` | 32,868 | 0.00488% |
| `biblioteka_nauki` | `rozporządzenie` | 19,594 | 0.00291% |
| `biblioteka_nauki` | `w pobliżu` | 3,378 | 0.00050% |
| `biblioteka_nauki` | `mieszkańców` | 34,392 | 0.00511% |
| `biblioteka_nauki` | `Dz.U.` | 33,098 | 0.00491% |
| `dziennik_ustaw` | `w roku` | 33,096 | 0.00681% |
| `dziennik_ustaw` | `klasyfikacji` | 10,896 | 0.00224% |
| `dziennik_ustaw` | `ustawa` | 77,376 | 0.01592% |
| `dziennik_ustaw` | `artykuł` | 27,773 | 0.00571% |
| `dziennik_ustaw` | `parlament` | 42,338 | 0.00871% |
| `dziennik_ustaw` | `rozporządzenie` | 166,383 | 0.03423% |
| `dziennik_ustaw` | `w pobliżu` | 1,087 | 0.00022% |
| `dziennik_ustaw` | `mieszkańców` | 8,025 | 0.00165% |
| `dziennik_ustaw` | `Dz.U.` | 158 | 0.00003% |
| `eltec_pol` | `w roku` | 108 | 0.00050% |
| `eltec_pol` | `klasyfikacji` | 1 | 0.00000% |
| `eltec_pol` | `ustawa` | 153 | 0.00071% |
| `eltec_pol` | `artykuł` | 173 | 0.00081% |
| `eltec_pol` | `parlament` | 95 | 0.00044% |
| `eltec_pol` | `rozporządzenie` | 35 | 0.00016% |
| `eltec_pol` | `w pobliżu` | 246 | 0.00114% |
| `eltec_pol` | `mieszkańców` | 214 | 0.00100% |
| `eltec_pol` | `Dz.U.` | 0 | 0.00000% |
| `eurlex` | `w roku` | 40,009 | 0.00168% |
| `eurlex` | `klasyfikacji` | 59,428 | 0.00250% |
| `eurlex` | `ustawa` | 30,368 | 0.00128% |
| `eurlex` | `artykuł` | 1,774,958 | 0.07464% |
| `eurlex` | `parlament` | 780,286 | 0.03281% |
| `eurlex` | `rozporządzenie` | 1,202,658 | 0.05057% |
| `eurlex` | `w pobliżu` | 6,088 | 0.00026% |
| `eurlex` | `mieszkańców` | 9,441 | 0.00040% |
| `eurlex` | `Dz.U.` | 915,707 | 0.03851% |
| `european_hplt_v3_pl` | `w roku` | 113,501 | 0.00444% |
| `european_hplt_v3_pl` | `klasyfikacji` | 24,675 | 0.00097% |
| `european_hplt_v3_pl` | `ustawa` | 16,067 | 0.00063% |
| `european_hplt_v3_pl` | `artykuł` | 283,875 | 0.01111% |
| `european_hplt_v3_pl` | `parlament` | 45,516 | 0.00178% |
| `european_hplt_v3_pl` | `rozporządzenie` | 11,841 | 0.00046% |
| `european_hplt_v3_pl` | `w pobliżu` | 44,108 | 0.00173% |
| `european_hplt_v3_pl` | `mieszkańców` | 172,896 | 0.00677% |
| `european_hplt_v3_pl` | `Dz.U.` | 9,487 | 0.00037% |
| `global_voices` | `w roku` | 160 | 0.00431% |
| `global_voices` | `klasyfikacji` | 5 | 0.00013% |
| `global_voices` | `ustawa` | 157 | 0.00423% |
| `global_voices` | `artykuł` | 1,047 | 0.02818% |
| `global_voices` | `parlament` | 366 | 0.00985% |
| `global_voices` | `rozporządzenie` | 26 | 0.00070% |
| `global_voices` | `w pobliżu` | 105 | 0.00283% |
| `global_voices` | `mieszkańców` | 508 | 0.01367% |
| `global_voices` | `Dz.U.` | 1 | 0.00003% |
| `govpl` | `w roku` | 3,372 | 0.00418% |
| `govpl` | `klasyfikacji` | 526 | 0.00065% |
| `govpl` | `ustawa` | 4,727 | 0.00586% |
| `govpl` | `artykuł` | 2,417 | 0.00300% |
| `govpl` | `parlament` | 3,967 | 0.00492% |
| `govpl` | `rozporządzenie` | 4,102 | 0.00509% |
| `govpl` | `w pobliżu` | 892 | 0.00111% |
| `govpl` | `mieszkańców` | 13,105 | 0.01625% |
| `govpl` | `Dz.U.` | 3,114 | 0.00386% |
| `nkjp1m` | `w roku` | 109 | 0.00419% |
| `nkjp1m` | `klasyfikacji` | 9 | 0.00035% |
| `nkjp1m` | `ustawa` | 79 | 0.00304% |
| `nkjp1m` | `artykuł` | 153 | 0.00588% |
| `nkjp1m` | `parlament` | 216 | 0.00830% |
| `nkjp1m` | `rozporządzenie` | 20 | 0.00077% |
| `nkjp1m` | `w pobliżu` | 77 | 0.00296% |
| `nkjp1m` | `mieszkańców` | 159 | 0.00611% |
| `nkjp1m` | `Dz.U.` | 2 | 0.00008% |
| `parliamentary` | `w roku` | 198,192 | 0.01203% |
| `parliamentary` | `klasyfikacji` | 12,637 | 0.00077% |
| `parliamentary` | `ustawa` | 459,179 | 0.02788% |
| `parliamentary` | `artykuł` | 182,038 | 0.01105% |
| `parliamentary` | `parlament` | 309,695 | 0.01881% |
| `parliamentary` | `rozporządzenie` | 113,547 | 0.00689% |
| `parliamentary` | `w pobliżu` | 4,964 | 0.00030% |
| `parliamentary` | `mieszkańców` | 78,048 | 0.00474% |
| `parliamentary` | `Dz.U.` | 23,809 | 0.00145% |
| `samorzad_gov_pl` | `w roku` | 3,661 | 0.00890% |
| `samorzad_gov_pl` | `klasyfikacji` | 520 | 0.00126% |
| `samorzad_gov_pl` | `ustawa` | 1,823 | 0.00443% |
| `samorzad_gov_pl` | `artykuł` | 1,269 | 0.00308% |
| `samorzad_gov_pl` | `parlament` | 1,218 | 0.00296% |
| `samorzad_gov_pl` | `rozporządzenie` | 1,409 | 0.00342% |
| `samorzad_gov_pl` | `w pobliżu` | 401 | 0.00097% |
| `samorzad_gov_pl` | `mieszkańców` | 14,450 | 0.03511% |
| `samorzad_gov_pl` | `Dz.U.` | 1,380 | 0.00335% |
| `wikibooks` | `w roku` | 319 | 0.00205% |
| `wikibooks` | `klasyfikacji` | 37 | 0.00024% |
| `wikibooks` | `ustawa` | 165 | 0.00106% |
| `wikibooks` | `artykuł` | 732 | 0.00470% |
| `wikibooks` | `parlament` | 283 | 0.00182% |
| `wikibooks` | `rozporządzenie` | 131 | 0.00084% |
| `wikibooks` | `w pobliżu` | 125 | 0.00080% |
| `wikibooks` | `mieszkańców` | 204 | 0.00131% |
| `wikibooks` | `Dz.U.` | 16 | 0.00010% |
| `wikinews` | `w roku` | 449 | 0.00370% |
| `wikinews` | `klasyfikacji` | 639 | 0.00526% |
| `wikinews` | `ustawa` | 407 | 0.00335% |
| `wikinews` | `artykuł` | 2,474 | 0.02038% |
| `wikinews` | `parlament` | 2,530 | 0.02084% |
| `wikinews` | `rozporządzenie` | 168 | 0.00138% |
| `wikinews` | `w pobliżu` | 455 | 0.00375% |
| `wikinews` | `mieszkańców` | 1,014 | 0.00835% |
| `wikinews` | `Dz.U.` | 27 | 0.00022% |
| `wikipedia` | `w roku` | 143,023 | 0.02022% |
| `wikipedia` | `klasyfikacji` | 46,043 | 0.00651% |
| `wikipedia` | `ustawa` | 9,536 | 0.00135% |
| `wikipedia` | `artykuł` | 28,165 | 0.00398% |
| `wikipedia` | `parlament` | 57,863 | 0.00818% |
| `wikipedia` | `rozporządzenie` | 5,637 | 0.00080% |
| `wikipedia` | `w pobliżu` | 41,915 | 0.00593% |
| `wikipedia` | `mieszkańców` | 122,766 | 0.01736% |
| `wikipedia` | `Dz.U.` | 232 | 0.00003% |
| `wikiquote` | `w roku` | 608 | 0.00191% |
| `wikiquote` | `klasyfikacji` | 15 | 0.00005% |
| `wikiquote` | `ustawa` | 357 | 0.00112% |
| `wikiquote` | `artykuł` | 606 | 0.00190% |
| `wikiquote` | `parlament` | 1,271 | 0.00398% |
| `wikiquote` | `rozporządzenie` | 27 | 0.00008% |
| `wikiquote` | `w pobliżu` | 207 | 0.00065% |
| `wikiquote` | `mieszkańców` | 527 | 0.00165% |
| `wikiquote` | `Dz.U.` | 6 | 0.00002% |
| `wikisource` | `w roku` | 16,571 | 0.00207% |
| `wikisource` | `klasyfikacji` | 166 | 0.00002% |
| `wikisource` | `ustawa` | 8,156 | 0.00102% |
| `wikisource` | `artykuł` | 16,230 | 0.00202% |
| `wikisource` | `parlament` | 6,119 | 0.00076% |
| `wikisource` | `rozporządzenie` | 1,651 | 0.00021% |
| `wikisource` | `w pobliżu` | 13,921 | 0.00174% |
| `wikisource` | `mieszkańców` | 14,335 | 0.00179% |
| `wikisource` | `Dz.U.` | 5 | 0.00000% |
| `wikivoyage` | `w roku` | 609 | 0.00356% |
| `wikivoyage` | `klasyfikacji` | 21 | 0.00012% |
| `wikivoyage` | `ustawa` | 46 | 0.00027% |
| `wikivoyage` | `artykuł` | 657 | 0.00384% |
| `wikivoyage` | `parlament` | 249 | 0.00145% |
| `wikivoyage` | `rozporządzenie` | 34 | 0.00020% |
| `wikivoyage` | `w pobliżu` | 6,480 | 0.03783% |
| `wikivoyage` | `mieszkańców` | 4,100 | 0.02394% |
| `wikivoyage` | `Dz.U.` | 1 | 0.00001% |
| `wolne_lektury` | `w roku` | 1,898 | 0.00184% |
| `wolne_lektury` | `klasyfikacji` | 80 | 0.00008% |
| `wolne_lektury` | `ustawa` | 1,060 | 0.00103% |
| `wolne_lektury` | `artykuł` | 1,824 | 0.00177% |
| `wolne_lektury` | `parlament` | 672 | 0.00065% |
| `wolne_lektury` | `rozporządzenie` | 128 | 0.00012% |
| `wolne_lektury` | `w pobliżu` | 2,073 | 0.00201% |
| `wolne_lektury` | `mieszkańców` | 1,658 | 0.00161% |
| `wolne_lektury` | `Dz.U.` | 5 | 0.00000% |

![Overall pattern counts](artifacts/pattern_frequency_overall.png)

![w roku by source](artifacts/pattern_frequency_w_roku.png)
![klasyfikacji by source](artifacts/pattern_frequency_klasyfikacji.png)
![ustawa by source](artifacts/pattern_frequency_ustawa.png)
![artykuł by source](artifacts/pattern_frequency_artykul.png)
![parlament by source](artifacts/pattern_frequency_parlament.png)
![rozporządzenie by source](artifacts/pattern_frequency_rozporządzenie.png)
![w pobliżu by source](artifacts/pattern_frequency_w_pobliżu.png)
![mieszkańców by source](artifacts/pattern_frequency_mieszkańców.png)
![Dz.U. by source](artifacts/pattern_frequency_dzu.png)
