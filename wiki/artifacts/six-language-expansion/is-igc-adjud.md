---
pretty_name: IGC-Adjud
language:
- is
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
- Legal
---

# Dataset Card for IGC-Adjud

<!-- START-SHORT DESCRIPTION -->
Icelandic court adjudications.
<!-- END-SHORT DESCRIPTION -->

From the district courts, the Court of Appeal, and the Supreme Court. This dataset combines two releases from the Icelandic Gigaword Corpus: IGC-Adjud 22.10 and the extension IGC-Adjud 2024ext. Together, the releases span adjudications from 1999 to 2023. Each document is a single adjudication.

Personal names have been anonymized in the source and replaced with tagged placeholders (e.g. `R-kvk-nf`).

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 33.36K
- **Number of tokens (Llama 3)**: 225.53M
- **Average document length in tokens (min, max)**: 6.76K (245, 194.57K)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "igc-adjud_Adjud2_landsrettur_100_2018",
  "text": "Staðfestur var úrskurður héraðsdóms um að X skyldi sæta gæsluvarðhaldi á grundvelli aliðar 1. mgr. 9[...]",
  "source": "igc-adjud",
  "added": "2026-07-03",
  "created": "2018-01-16, 2018-01-16",
  "token_count": 469
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
To learn more about the annotations and how to use them, see the [annotations section](https://huggingface.co/datasets/danish-foundation-models/icelandic-dynaword#annotations) of the main readme.

<p align="center">
<img src="./images/annotation_profile.png" width="700" style="margin-right: 10px;" />
</p>
<!-- END-ANNOTATION PLOTS -->


## Additional Information

### Data Collection and Processing

The source court is encoded in each document's `id`, following the IGC's own identifier scheme. `Adjud1` is a district court, `Adjud2` the Court of Appeal (Landsréttur), and `Adjud3` the Supreme Court (Hæstiréttur). For district-court documents the `id` also names the region (e.g. `reykjavikur`, `reykjaness`).


### Limitations

The corpus is legal text, so it reflects a formal legal register rather than general Icelandic. Personal names are anonymized and replaced with tagged placeholders, which introduces non-natural tokens into the text.

### Source Data
The source publisher is [The Árni Magnússon Institute for Icelandic Studies](https://www.arnastofnun.is/). This dataset combines two releases: IGC-Adjud 22.10 (`http://hdl.handle.net/20.500.12537/240`) and the extension, IGC-Adjud 2024ext (`http://hdl.handle.net/20.500.12537/333`).

### Citation Information

If you use this dataset, cite the source releases:
>Barkarson, Starkaður and Steingrímsson, Steinþór, 2022, IGC-Adjud 22.10 (unannotated version), CLARIN-IS, http://hdl.handle.net/20.500.12537/240.
and
>Barkarson, Starkaður and Steingrímsson, Steinþór, 2024, IGC-Adjud 2024ext (unannotated version), CLARIN-IS, http://hdl.handle.net/20.500.12537/333.

## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
