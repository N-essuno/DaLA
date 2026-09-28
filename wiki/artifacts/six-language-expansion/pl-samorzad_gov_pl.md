# samorzad_gov_pl

Polish-language articles published by 246 local public institutions and one
central platform tenant using the shared
[samorzad.gov.pl](https://samorzad.gov.pl/) platform. The collection includes
municipalities, counties, schools, social-service institutions, cultural
institutions and other public bodies.

## Dataset description

- **Published source dataset:** https://huggingface.co/datasets/dawidmajewski/samorzad-gov-pl-articles
- **Source revision:** `d5eaefc32c3f17cfa5a01d14573f8e4ee7d43385` (`v0.1`)
- **Original platform:** https://samorzad.gov.pl/
- **Domain:** local government, education, social services and culture
- **Language:** Polish (`pl`)
- **License:** `CC-BY-SA-4.0`
- **Created:** 2020–2026 for dated records; most source records do not expose a date
- **Added:** 2026-08-14

## Statistics

| read | kept | characters | tokens (`cl100k_base` proxy) | publishers |
|---:|---:|---:|---:|---:|
| 81,422 | 72,771 | 109,513,754 | 41,154,506 | 247 |

The released Parquet uses the canonical DynaWord schema:

`id, text, source, added, created, token_count, license, author`

Every retained document has the human-readable name of its publishing
institution in `author`. The names were obtained from the `og:site_name`
metadata on each institution's platform homepage and frozen in
`src/samorzad_gov_pl_publishers.json`.

## Provenance

The source release was collected directly from public samorzad.gov.pl pages.
It contains every article URL found and successfully extracted during that
crawl, rather than a topical or length-selected sample. The published source
release contains 81,422 extracted articles from 247 institutions.

`src/clean_samorzad_gov_pl.py` converts the published source Parquets into the
DynaWord schema, applies the common minimum-length and Polish-language checks,
removes exact duplicate text and counts tokens with `cl100k_base`. The retained
article text is otherwise unchanged.

## Licensing — traceable basis and attribution

The platform's shared footer states that textual content published in the
service, excluding audiovisual content, is available under CC BY-SA 4.0. A
release-time audit verified the live shared footer script and confirmed that
all 81,422 preserved source pages referenced that script. The evidence URL,
byte count and SHA-256 digest are recorded in
`samorzad_gov_pl.license-evidence.json`.

The fixed DynaWord schema does not contain article titles or URLs. Therefore,
`samorzad_gov_pl.attribution.jsonl` maps every released `id` to its title,
discovered `source_url`, fetched `final_url`, recommended `attribution_url`,
publishing institution, platform tenant, license URL and modification notice.
For 693 retained records the discovered and final URLs differ because the
platform redirected the request; `attribution_url` uses the fetched final URL.

Recommended attribution:

> Publishing institution listed in the `author` column; article title and
> canonical samorzad.gov.pl URL from the attribution sidecar; CC BY-SA 4.0.
> Article text was extracted from HTML and normalized for inclusion in Polish
> DynaWord.

Photographs, recordings, videos and attachment bytes are not redistributed.
The source platform gives audiovisual materials separate terms, and linked
attachments can carry their own conditions. URLs and descriptive metadata do
not redistribute those files.

## Filters applied

| result | documents |
|---|---:|
| source records read | 81,422 |
| shorter than 200 characters | 6,624 |
| below the Polish-language heuristic | 35 |
| exact duplicate text | 1,992 |
| retained | 72,771 |

Exact duplicate removal is performed after short-text and language checks.
Existing DynaWord sources take priority when the same text is already present
elsewhere in the corpus.

## Limitations and recommended use

- The collection is not a complete historical archive of samorzad.gov.pl. It
  contains the pages found and successfully extracted during one crawl.
- Only 21,892 of the 81,422 source records expose a publication date. After
  filtering, 52,214 retained records have an empty `created` value; no date is
  inferred for them.
- The source text can contain spelling errors, extraction remnants or captions.
- Very short source records are excluded by DynaWord's 200-character minimum.
- Institution names describe the publishing body, not necessarily the person
  who wrote an article.
- Government and institutional language is overrepresented; this source should
  complement, not stand in for, conversational or general Polish.
- Canonical URLs may stop working as institutions update their websites.

## Included files

- `samorzad_gov_pl.parquet` — canonical DynaWord records;
- `samorzad_gov_pl.stats.json` — build and release statistics;
- `samorzad_gov_pl.attribution.jsonl` — per-document source attribution;
- `samorzad_gov_pl.license-evidence.json` — release-time license evidence;
- `NOTICE.md` — attribution and redistribution notice.

## Reproduction

The converter refuses to build unless the downloaded `manifest.json` and all
10 Parquet shards match the pinned release. It verifies the manifest SHA-256,
artifact set, shard sizes and SHA-256 values, row counts, schema, unique record
IDs and the release-wide record-text digest before conversion. Outputs are
written to temporary files, cross-checked, synced and transactionally replaced;
a failed replacement restores the previous artifact set.

Download the published source Parquet shards and run:

```bash
hf download dawidmajewski/samorzad-gov-pl-articles \
  --repo-type dataset \
  --revision d5eaefc32c3f17cfa5a01d14573f8e4ee7d43385 \
  --include 'data/*.parquet' --include manifest.json \
  --local-dir /tmp/samorzad-gov-pl-articles

uv run --with pyarrow --with tiktoken python src/clean_samorzad_gov_pl.py \
  --input /tmp/samorzad-gov-pl-articles/data \
  --repo-root . \
  --added 2026-08-14
```

Use `--refresh-publishers` only when intentionally updating the frozen
institution-name manifest from live tenant homepages. Review any changed name
before publishing a new release.
