# eurlex

EUR-Lex (EU legal acts, Polish)

## Dataset description
- **Source (upstream):** https://eur-lex.europa.eu
- **Domain:** legal
- **Language:** Polish (pl)
- **License:** `CC-BY-4.0`
- **Created (range):** 1952-01-01, 2024-12-31
- **Added:** 2026-07-02

## Licensing — traceable basis
EU documents reusable under Commission Decision 2011/833/EU; normative acts fall outside copyright (PL art. 4 pr. aut.). Reuse authorised with source acknowledgement.

## Provenance
Pulled from SpeakLeash's public redistribution (`speakleash-ds-pub`, key `eurlex_corpus`) of the upstream source above. SpeakLeash credited as intermediate aggregator; upstream license/attribution preserved.

## Statistics
| documents | characters | tokens (tiktoken proxy) |
|---:|---:|---:|
| 243,060 | 5,976,949,249 | 2,378,055,718 |


## Per-document license metadata
| license | documents |
|---|---:|
| `CC-BY-4.0` | 243,060 |

Author metadata present for **0** documents. Empty values mean the upstream record did not expose a machine-readable author field.

Statistics were recomputed directly from the released parquet file.

## Filters applied (build_dynaword.py)
Minimal, per Dynaword guidelines (heavy filtering left to downstream use):
- drop documents < 200 chars: **0**
- drop non-Polish (diacritic ratio): **0**
- exact cross-source dedup (sha1): **0**
- OCR alpha-ratio < 0.70 (OCR sources only): **0**
- read 243,060 → kept 243,060

Token counts are a fast tiktoken (cl100k) proxy (~1% off Llama-3); the canonical
Llama-3 count is computed at release.
