---
pretty_name: Forskning & Framsteg
language:
- sv
license: cc-by-4.0
license_name: CC-BY 4.0
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
- News
---

# Dataset Card for Forskning & Framsteg

<!-- START-SHORT DESCRIPTION -->
Sentences from Forskning & Framsteg, distributed by [Språkbanken Text](https://spraakbanken.gu.se/resurser/fof).
<!-- END-SHORT DESCRIPTION -->

This subset contains sentences from [Forskning & Framsteg](https://fof.se), a Swedish popular science magazine. Each sample corresponds to a single sentence — the source corpus supplies sentences in scrambled order for copyright and privacy reasons, so the original article context is not recoverable.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 43.79K
- **Number of tokens (Llama 3)**: 1.44M
- **Average document length in tokens (min, max)**: 32.98 (2, 436)
<!-- END-DESC-STATS -->


## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "forskning-framsteg_00000001",
  "text": "Ungefär 15 000 personer i Sverige har Parkinsons sjukdom.",
  "source": "forskning-framsteg",
  "added": "2026-07-27",
  "created": "1992-01-01, 1992-12-31",
  "token_count": 21
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


## Additional Information

### Limitations

Each sample is a single sentence. The source corpus deliberately scrambles sentence order for copyright and privacy protection, so the original article context is not recoverable.

The source material is tokenized and does not preserve the original spacing between tokens. We approximate readable spacing when reconstructing plain text.

As popular science writing, this material covers a broad range of scientific topics aimed at a general audience.

### Source

We download the corpus from Språkbanken Text's official source file for *Forskning & Framsteg*. Språkbanken Text distributes the corpus with DOI `10.23695/9aaa-4q88`.

### Citation Information

If you use this subset, cite the original source release:

> Språkbanken Text (2025). Forskning & Framsteg. [Data set]. Språkbanken Text. https://doi.org/10.23695/9aaa-4q88

```bibtex
@misc{forskning_framsteg,
  doi = {10.23695/9aaa-4q88},
  url = {https://spraakbanken.gu.se/en/resources/fof},
  author = {Språkbanken Text},
  language = {swe},
  title = {Forskning & Framsteg},
  publisher = {Språkbanken Text},
  year = {2025}
}
```

## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
