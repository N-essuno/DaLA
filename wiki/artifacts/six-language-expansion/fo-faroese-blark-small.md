---
pretty_name: Faroese BLARK Small
language:
- fo
license: cc-by-4.0
license_name: CC-BY 4.0
size_categories:
- 100k-1m
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
source_datasets:
- barbaroo/Faroese_BLARK_small
domains:
- Web
---

# Dataset Card for Faroese BLARK Small

<!-- START-SHORT DESCRIPTION -->
[BLARK Small](https://mtd.setur.fo/en/resource/1287/) is a filtered version of the Faroese BLARK text corpus.
<!-- END-SHORT DESCRIPTION -->

BLARK stands for Basic Language Resource Kit. The Faroese BLARK is a collection of
language resources developed by the [Ravnur speech-recognition
project](https://mtd.setur.fo/en/publisher/verkaetlanin-ravnur/). BLARK Small contains
cleaned sentences from the BLARK 1.0 text corpus.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 300.82K
- **Number of tokens (Llama 3)**: 22.25M
- **Average document length in tokens (min, max)**: 73.97 (11, 3.99K)
<!-- END-DESC-STATS -->

## Dataset Structure

<!-- START-SAMPLE -->
```py
{
  "id": "faroese-blark-small_000000",
  "text": "Myndin niðanfyri vísir løntakaratalið fyri bæði kyn. Kámu strikurnar vísa løntakararnar í tali og st[...]",
  "source": "faroese-blark-small",
  "added": "2026-08-24",
  "created": "2023-06-28, 2025-11-14",
  "token_count": 215
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

BLARK Small excludes short sentences, archaic Faroese, lists, formatting noise,
URL strings, rows with little linguistic content, and duplicates. For Dynaword,
the `Text` column is kept and normalized duplicates are removed within BLARK Small
and against existing Dynaword datasets.

### Source

BLARK Small was published by Barbara Scalvini, [Máltøknidepilin and the Ravnur
Project](https://mtd.setur.fo/en/resource/1287/).

The [BLARK paper](https://aclanthology.org/2022.lrec-1.495/) describes the background
corpus as a mix of formal and informal Faroese text. The paper also discusses
[FTS](https://spraakbanken.gu.se/en/resources/fts) in its overview of Faroese language
resources.

The source list includes
[Birkblog](https://birkblog.blogspot.com/), [fmr.fo](https://www.fmr.fo/),
[mmr.fo](https://www.mmr.fo/), [vedur.fo](https://www.vedur.fo/),
[hagstova.fo](https://www.hagstova.fo/), [kvf.fo](https://www.kvf.fo/),
[portal.fo](https://www.portal.fo/), [megd.fo](https://www.megd.fo/),
[starvsbladid.fo](https://www.starvsbladid.fo/),
[sosialurin.fo](https://www.sosialurin.fo/), and [dimma.fo](https://www.dimma.fo/).

### Limitations

The corpus mixes several genres. The source, author and publication date are not
available for each row. The `created` field records the BLARK Small release and
revision dates.

### Citation Information

```bibtex
@inproceedings{simonsen-etal-2022-creating,
  title = {Creating a Basic Language Resource Kit for Faroese},
  author = {Simonsen, Annika and Lamhauge, Sandra Saxov and Debess, Iben Nyholm and Henrichsen, Peter Juel},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages = {4637--4643},
  year = {2022},
  publisher = {European Language Resources Association}
}
```

### License Information

BLARK Small is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
