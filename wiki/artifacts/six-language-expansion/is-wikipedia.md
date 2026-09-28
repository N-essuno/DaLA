---
pretty_name: Wikipedia
language:
- is
license: cc-by-sa-4.0
license_name: CC-BY-SA 4.0
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
- Encyclopedic
---

# Dataset Card for Wikipedia

<!-- START-SHORT DESCRIPTION -->
The Icelandic subsection of [wikipedia](https://wikipedia.org/).
<!-- END-SHORT DESCRIPTION -->

You can read more about wikipedia on their [about](https://en.wikipedia.org/wiki/Wikipedia:About) page.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 61.26K
- **Number of tokens (Llama 3)**: 36.16M
- **Average document length in tokens (min, max)**: 590.22 (9, 48.33K)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "wikipedia_3",
  "text": "Finnland\nFinnland (finnska Suomi; sænska: Finland), formlegt heiti Lýðveldið Finnland (finnska: Suom[...]",
  "source": "wikipedia",
  "added": "2026-07-28",
  "created": "2026-03-26, 2026-03-26",
  "token_count": 13580
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

For this dataset the generated field description above understates what `created` holds: it is the date of the page's latest revision in the dump, not the date the text was written. See [Data Collection and Processing](#data-collection-and-processing).

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

The articles are processed from the official Wikimedia `pages-articles` dump. 

The `created` field records the date of the article's latest revision in the dump, not when the article was first created. 

Wiki markup is converted to plain text with the [wtf_wikipedia](https://github.com/spencermountain/wtf_wikipedia) library; a very small number of articles may retain minor residual markup (e.g. from image galleries or tables).

Because the parser is written in JavaScript, you need to have Node.js installed on your machine. To run the `create.py` file you first need to do:

```bash
$ cd parser/ && npm install && cd ..
```

A very small fraction of articles (~0.2%) are MediaWiki redirect pages that were not filtered out, because `wtf_wikipedia`'s redirect detection misses some redirects (including ones using the Icelandic-language `#tilvísun` magic word) — their text is just a short note naming the redirect target rather than real article content.

### Limitations

Wikipedia is collaboratively edited, so article quality, length, and coverage vary widely. 

A few of the articles are redirect pages. See [Data Collection and Processing](#data-collection-and-processing).

### Source Data

The data is derived from the Icelandic Wikipedia's `pages-articles` dump published by the Wikimedia Foundation (`https://dumps.wikimedia.org/iswiki/`).

## License Information

This dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), matching the license Wikipedia contributors release their article text under (per the Wikimedia Foundation's [Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use), Section 7: text is dual-licensed CC BY-SA 4.0 and GFDL).
