---
pretty_name: Akademiliv
language:
- sv
license: cc-by-4.0
license_name: CC-BY 4.0
size_categories:
- 100K<n<1M
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

# Dataset Card for Akademiliv

<!-- START-SHORT DESCRIPTION -->
Sentences from Akademiliv, distributed by [Språkbanken Text](https://spraakbanken.gu.se/resurser/akademiliv).
<!-- END-SHORT DESCRIPTION -->

This subset contains sentences from [Akademiliv](https://www.akademiliv.se), a University of Gothenburg staff magazine, covering 2011–2024. Each sample corresponds to a single sentence — the source corpus supplies sentences in scrambled order for copyright and privacy reasons, so the original article context is not recoverable.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 114.92K
- **Number of tokens (Llama 3)**: 4.42M
- **Average document length in tokens (min, max)**: 38.43 (2, 952)
<!-- END-DESC-STATS -->


## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "akademiliv_00000001",
  "text": "En konkret åtgärdsplan sjösätts nu för att modernisera läkarutbildningen i Göteborg, bland annat gen[...]",
  "source": "akademiliv",
  "added": "2026-07-27",
  "created": "2011-06-10, 2011-06-10",
  "token_count": 54
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

As a university staff publication, this material covers academic workplace news and internal university affairs.

### Source

We download the corpus from Språkbanken Text's official source file for *Akademiliv*. Språkbanken Text distributes the corpus with DOI `10.23695/7cn4-7126`.

### Citation Information

If you use this subset, cite the original source release:

> Språkbanken Text (2025). Akademiliv. [Data set]. Språkbanken Text. https://doi.org/10.23695/7cn4-7126

```bibtex
@misc{akademiliv,
  doi = {10.23695/7cn4-7126},
  url = {https://spraakbanken.gu.se/en/resources/akademiliv},
  author = {Språkbanken Text},
  language = {swe},
  title = {Akademiliv},
  publisher = {Språkbanken Text},
  year = {2025}
}
```

## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
