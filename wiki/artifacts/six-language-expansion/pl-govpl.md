# govpl

gov.pl — Polish government press releases

## Dataset description
- **Source (upstream):** https://www.gov.pl/
- **Domain:** government
- **Language:** Polish (pl)
- **License:** `CC-BY-SA-4.0`
- **Created (range):** 2005-09-26, 2026-07-19
- **Added:** 2026-07-19

## Licensing — traceable basis
All textual content on gov.pl is published under CC-BY-SA 4.0 (site footer: "Treści tekstowe publikowane w serwisie … są udostępniane na licencji CC BY-SA 4.0"). Authored and hosted by the Polish government, which holds the right to license it — same open family as Polish Wikipedia.

## Provenance
Ministry/agency press releases fetched directly from gov.pl (editor-content per article, HTML stripped) by src/fetch_govpl.py. Subsites + their non-uniform news-section paths are enumerated by src/discover_govpl.py into the committed manifest src/govpl_subsites.json (gov.pl has no sitemap/API). Each record's meta.url resolves on gov.pl.

## Statistics
| documents | characters | tokens (tiktoken proxy) |
|---:|---:|---:|
| 88,190 | 218,099,000 | 80,666,129 |


## Per-document license metadata
| license | documents |
|---|---:|
| `CC-BY-SA-4.0` | 88,190 |

Author metadata present for **0** documents. Empty values mean the upstream record did not expose a machine-readable author field.

## Filters applied (build_dynaword.py)
Minimal, per Dynaword guidelines (heavy filtering left to downstream use):
- drop documents < 200 chars: **0**
- drop non-Polish (diacritic ratio): **24**
- exact cross-source dedup (sha1): **1,926**
- OCR alpha-ratio < 0.70 (OCR sources only): **0**
- read 90,140 → kept 88,190

Token counts are a fast tiktoken (cl100k) proxy (~1% off Llama-3); the canonical
Llama-3 count is computed at release.
