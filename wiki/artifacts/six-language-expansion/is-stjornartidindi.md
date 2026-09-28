---
pretty_name: "Stj\xF3rnart\xED\xF0indi (Official Journal of Iceland)"
language:
- is
license: other
license_name: Icelandic Copyright Law
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
source_datasets:
- original
domains:
- Legal
---

# Dataset Card for Stjórnartíðindi (Official Journal of Iceland)

<!-- START-SHORT DESCRIPTION -->
Icelandic laws, regulations and international agreements as published in the official state gazette.
<!-- END-SHORT DESCRIPTION -->

[Stjórnartíðindi](https://island.is/stjornartidindi) is the official gazette of Iceland, published under *lög um Stjórnartíðindi og Lögbirtingablað nr. 15/2005*. A legal instrument takes effect in Iceland only once it has been published here, so the gazette is the authoritative record of Icelandic law as enacted.

The gazette appears in three divisions, identified in each sample's `id`:

| Division | Content |
|---|---|
| A | Acts of parliament (*lög*), presidential decrees and resolutions |
| B | Regulations (*reglugerðir*), municipal rules, fee schedules, planning notices and other secondary legislation |
| C | International agreements to which Iceland is a party |

Each published document is a single sample, and the collection runs from 1995 to the present. The dataset is renewable: re-running `create.py` picks up everything published since the last build.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 39.76K
- **Number of tokens (Llama 3)**: 91.53M
- **Average document length in tokens (min, max)**: 2.30K (96, 1.03M)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "stjornartidindi_B_2026_870",
  "text": "REGLUGERÐ um leyfilegan heildarafla og veiðar í atvinnuskyni fiskveiðiárið 2026/2027 og almanaksárið[...]",
  "source": "stjornartidindi",
  "added": "2026-08-12",
  "created": "2026-07-30, 2026-07-31",
  "token_count": 10365
}
```

### Data Fields

An entry in the dataset consists of the following fields:

- `id` (`str`): A unique identifier for each document.
- `text` (`str`): The content of the document.
- `source` (`str`): The source of the document (see [Source Data](#source-data)).
- `added` (`str`): The date when the document was added to this collection.
- `created` (`str`): The date range when the document was originally created.
- `token_count` (`int`): The number of tokens in the sample computed using the Llama 3 tokenizer.
<!-- END-SAMPLE -->

The `id` encodes the division and the document's official publication number, so `stjornartidindi_B_2026_870` is document 870/2026 of B-deild and can be looked up directly in the gazette. The `created` range spans the document's signature date and its publication date.

### Dataset Statistics

<!-- START-DATASET PLOTS -->
<p align="center">
<img src="./images/dist_document_length.png" width="600" style="margin-right: 10px;" />
</p>
<!-- END-DATASET PLOTS -->

### Annotation Overview
<!-- START-ANNOTATION PLOTS -->
Each document comes with annotations describing its content, such as content quality, information density, and educational value.
Each bar shows the share of documents at each level of one annotation, from worst (light) to best (dark).
To learn more about the annotations and how to use them, see the [annotations section](https://huggingface.co/datasets/danish-foundation-models/icelandic-dynaword#annotations) of the main readme.

<p align="center">
<img src="./images/annotation_profile.png" width="700" style="margin-right: 10px;" />
</p>
<!-- END-ANNOTATION PLOTS -->


## Additional Information

### Data Collection and Processing
The gazette's publisher offers each document in two forms: as web pages, and as the PDF it was printed from. The web pages carry the full text of laws and regulations from 2006 onwards. Before that they carry nothing but a page footer, and for international agreements they carry only the short notice announcing the agreement, never the agreement itself. In those two cases the text has to come from the printed PDF.

Where the web pages are used, the text is taken as it stands, with formatting removed. Tables are kept and laid out one row per line, since much of the gazette's substance — fee schedules, fishing quotas, customs tariffs — lives in them.

The PDFs are typeset pages rather than documents. They contain real text and needed no OCR, but the printed layout has to be undone:

- International agreements are printed with the Icelandic text and a foreign translation side by side. Read naively the two languages come out interleaved line by line, so the columns are separated and the foreign one discarded — a little over half of what those PDFs contain.
- The header at the top of every page, giving the publication number and date, is removed.
- Words broken across a line are rejoined, and printed lines reassembled into paragraphs.
- Wide tables are printed sideways, a quarter turn from the rest of the page. They are turned back, which puts their rows and columns in reading order.
- Each PDF is a page crop, so where a document begins partway down a page the crop also picks up its neighbours, sometimes a whole one. Anything printed above the document's own heading is dropped, and the heading is identified by checking it against the document's official title. The gazette numbers its documents in sequence, so the same check applied to the *next* document's title says where the current document ends.
- A few documents cannot be read at all: some embed fonts that carry no character mapping, and a handful were scanned rather than typeset, leaving a picture of the page and no text. Both produce characters that spell nothing, and are discarded.

### Limitations

A document's own text can only be separated from its neighbours' where their headings are found, and a heading printed too badly to read is not found. Where that happens nothing is trimmed and the text keeps a neighbour's alongside its own; `C_2004_49` is worse than that, holding another agreement's text entirely. A few headings were printed twice over themselves to imitate bold type, which leaves them unreadable; those were dropped and the title taken from the publisher's metadata instead. One document, `B_2004_44`, is missing altogether: the publisher serves a Word file at its PDF address.

Not every trace of the foreign text is gone. Whether a passage is Icelandic is judged on the words in it, so a fragment carrying too few words to judge survives — an annex heading in English, or a column of one-letter codes left stranded when the English descriptions beside it were removed.

B-deild makes up most of the corpus and is repetitive by nature — planning notices, fee schedules and municipal rules reuse standard wording heavily — so expect near-duplicate text even though identical documents have been removed. A few documents are almost entirely numeric tables, the largest being an act on waste-disposal fees that consists mostly of a customs tariff list.

Acts still in force appear both here in A-deild and in [igc-law](../igc-law/igc-law.md)'s `Law3`, but they read differently and neither replaces the other. `Law3` is the consolidated statute book, with later amendments merged into the text and coverage reaching back to 1275, while A-deild records each act as it was originally passed and so is the only one of the two to show amendment history. Regulations and international agreements have no counterpart in `igc-law`.

The corpus is legal text, so it reflects a formal legal register rather than everyday Icelandic.

### Source Data

The publisher's REST API at `https://api.stjornartidindi.is`, documented at `https://api.stjornartidindi.is/swagger`. PDFs are served from `https://adverts.stjornartidindi.is`.

## License Information

Everything in this dataset is excluded from the Icelandic copyright law ([Höfundalög](https://www.althingi.is/lagas/nuna/1972073.html)) by its article 9.
