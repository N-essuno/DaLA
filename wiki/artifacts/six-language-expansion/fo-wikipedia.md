---
pretty_name: Wikipedia
language:
- fo
license: cc-by-sa-4.0
license_name: CC-BY-SA 4.0
size_categories:
- 10k-100k
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
source_datasets:
- wikipedia
domains:
- Encyclopedic
---

# Dataset Card for Wikipedia

<!-- START-SHORT DESCRIPTION -->
The Faroese subsection of [Wikipedia](https://fo.wikipedia.org/wiki/Forsíða).
<!-- END-SHORT DESCRIPTION -->

You can read more about Wikipedia on its [about](https://en.wikipedia.org/wiki/Wikipedia:About) page.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 12.80K
- **Number of tokens (Llama 3)**: 5.45M
- **Average document length in tokens (min, max)**: 425.51 (8, 28.77K)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "wiki_787",
  "text": "Føroyar\nFøroyar (danskt: Færøerne; 62° norðurbreidd, 7° vesturlongd) eru oyggjaland í Norðuratlantsh[...]",
  "source": "wiki",
  "added": "2026-07-03",
  "created": "2026-06-09, 2026-06-09",
  "token_count": 20350
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
To learn more about the annotations and how to use them, see the [annotations section](https://huggingface.co/datasets/danish-foundation-models/faroese-dynaword#annotations) of the main readme.

<p align="center">
<img src="./images/annotation_profile.png" width="700" style="margin-right: 10px;" />
</p>
<!-- END-ANNOTATION PLOTS -->


## Additional Information

### Processing

The source is the latest Faroese Wikipedia pages/articles dump from Wikimedia. The dump is parsed with [`wtf_wikipedia`](https://github.com/spencermountain/wtf_wikipedia/tree/dev), with `mwparserfromhell` as a fallback. Dynaword keeps one row per article and stores the article title together with the parsed text.

### Limitations

The text reflects the content and coverage of Faroese Wikipedia at the dump date. The source is parsed from Wikipedia markup, so some formatting, tables, references, or template content may be absent or simplified.

### Source

The raw data comes from the [latest Faroese Wikipedia dump](https://dumps.wikimedia.org/fowiki/latest/), specifically `fowiki-latest-pages-articles.xml.bz2`.

### License Information

The source corpus is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), following the Wikipedia source license.
