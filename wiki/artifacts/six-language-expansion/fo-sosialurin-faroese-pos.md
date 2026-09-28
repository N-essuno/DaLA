---
pretty_name: Sosialurin Faroese PoS
language:
- fo
license: cc-by-4.0
license_name: CC-BY 4.0
size_categories:
- 1-10k
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

# Dataset Card for Sosialurin Faroese PoS

<!-- START-SHORT DESCRIPTION -->
Faroese newspaper text from the POS-tagged [Sosialurin](https://mtd.setur.fo/en/resource/sosialurin-faroese-pos/) corpus.
<!-- END-SHORT DESCRIPTION -->

This subset contains text from the Faroese newspaper *Sosialurin*. The source corpus contains about 100,000 word tokens.

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 6.15K
- **Number of tokens (Llama 3)**: 263.51K
- **Average document length in tokens (min, max)**: 42.85 (2, 210)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "sosialurin-faroese-pos_00000",
  "text": "Guðrið Poulsen í Riberhúsi",
  "source": "sosialurin-faroese-pos",
  "added": "2026-07-01",
  "created": "2022-06-16, 2022-06-16",
  "token_count": 13
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

### Limitations

The source was made for POS tagging. Rows in the dataset usually represent sentences or paragraphs instead of the full articles.

The source does not provide publication dates, so we use the source release date for all samples.

The text is reconstructed from tokenized lines, so punctuation and quotation spacing may differ from the original newspaper text.

### Source

The source is listed by Máltøknidepilin as [Sosialurin Faroese PoS](https://mtd.setur.fo/en/resource/sosialurin-faroese-pos/). The raw dataset is published on Hugging Face as [`vesteinn/sosialurin-faroese-pos`](https://huggingface.co/datasets/vesteinn/sosialurin-faroese-pos).

The Máltøknidepilin page lists the publisher as Vésteinn, the release date as 16 June 2022, and the license as CC BY 4.0.

### Citation Information

If you use this subset, cite the original source:

```bibtex
@misc{sosialurin-pos,
  title = {Marking av teldutøkum tekstsavn},
  author = {Zakaris Svabo Hansen, Heini Justinussen, and Mortan Ólason},
  url = {http://ark.axeltra.com/index.php?type=person&lng=en&id=18},
  year = {2004}
}
```

## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
