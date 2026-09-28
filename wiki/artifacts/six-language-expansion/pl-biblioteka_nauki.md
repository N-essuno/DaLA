# biblioteka_nauki

Biblioteka Nauki — Polish open-access scholarly corpus (scientific articles, books, chapters)

## Dataset description
- **Source (upstream):** https://bibliotekanauki.pl (operated by ICM, University of Warsaw — Centrum Otwartej Nauki)
- **Domain:** academic (scientific articles, books, chapters)
- **Language:** Polish (pl)
- **License:** per-document — CC BY 4.0 28,335 · CC BY-SA 4.0 13,733 · Public Domain 3
- **Created (range):** per-document publication year, from each item's bibliotekanauki.pl metadata
- **Added:** 2026-06-27

## Licensing — traceable basis
The SpeakLeash blob carries no per-document license field, and ~53% of the upstream collection is
all-rights-reserved, so a per-item gate was mandatory. Each item's license was recovered from the
bibliotekanauki.pl API (`bibliotekanauki.pl/api/{type}/<id>` → `license.iconId`). Only openly-licensed
items are kept; all-rights-reserved and unknown items are gated out. Recovery yielded 43,576 clean ids.

| license | documents |
|---|---:|
| CC BY 4.0 | 28,335 |
| CC BY-SA 4.0 | 13,733 |
| Public Domain | 3 |

The per-document license is carried in the parquet `license` column.

## Attribution
Per-document attribution (author · year · title · publisher) was re-fetched 1:1 from
bibliotekanauki.pl (43,576 / 43,576 ok) and is carried in the parquet `attribution` column,
satisfying the CC BY / CC BY-SA attribution requirement that applies to every document here.

## Provenance
Pulled from SpeakLeash's public redistribution (`speakleash-ds-pub`, key `biblioteka_nauki_pl_corpus`)
of the upstream source above. SpeakLeash credited as intermediate aggregator; upstream license and
attribution recovered per document and preserved.

## Statistics
| documents | characters | tokens (tiktoken proxy) |
|---:|---:|---:|
| 42,071 | 1,748,333,935 | 673,488,831 |

## Filters applied (build_dynaword.py)
Minimal, per Dynaword guidelines (heavy filtering left to downstream use):
- drop documents < 200 chars: **0**
- drop non-Polish (diacritic ratio): **925**
- exact cross-source dedup (sha1): **180**
- OCR alpha-ratio < 0.70 (OCR source): **400**
- read 43,576 → kept 42,071

Token counts are a fast tiktoken (cl100k) proxy (~1% off Llama-3); the canonical
Llama-3 count is computed at release.

## Limitations
- **Mixed license.** Roughly one third of the documents are CC BY-SA 4.0 (share-alike); the exact
  per-document license is in the `license` column. Downstream redistribution must respect share-alike
  for those documents.
- **OCR / extraction noise.** Text is extracted from publisher PDFs; some layout and hyphenation
  artefacts remain after the alpha-ratio filter.
- **Per-document dates.** `created` is the publication year from upstream metadata; finer-grained
  dates are not available.
