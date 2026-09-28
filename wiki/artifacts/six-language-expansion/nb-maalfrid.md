---
pretty_name: Maalfrid
language:
- nob
- nno
- nor
license: other
license_name: NLOD 2.0
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
domains:
- Web
---

# Dataset Card for Maalfrid

<!-- START-SHORT DESCRIPTION -->
Norwegian content from Norwegian institutions websites.
<!-- END-SHORT DESCRIPTION -->

Documents are derived from the [Målfrid collection](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-69/) as a subsection of the 
[Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC), which is a collection of multiple smaller Norwegian corpuses suitable for training large language models.

The Målfrid Corpus (2021), provided and collected by the Norwegian Language Bank, is the result of a focused web crawl from public institutions’ websites.
These websites store public sector information from a wide range of domains and institutions, ranging from healthcare to taxation.

The main purpose of the crawl was to determine the language use distribution in various institutions. 
This work was conducted in close collaboration with the Language Council of Norway (Språkrådet). 
The output from the crawl was Web ARChive (WARC, 2022) files, which can be processed and turned into differently derived language resources.
In the NCC, the subset containing all PDFs from the Målfrid Corpus was processed using a slightly modified version of MuPDF (Andersson, 2022).

Using the stylistic and layout information contained in the PDFs, they focused the extraction on the main text body of each document, ignoring other elements such as tables, footnotes, and image captions.
This approach produced very long and consistent documents with few errors and PDF-related artifacts. 
The NCC processed around 9.2M public PDF documents harvested from December 2020 to January 2021 belonging to 311 different institutions.


## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 3.23M
- **Number of tokens (Llama 3)**: 2.23B
- **Average document length in tokens (min, max)**: 690.89 (10, 62.24K)
<!-- END-DESC-STATS -->


## Dataset Structure
An example from the dataset looks as follows.
<!-- START-SAMPLE -->
```py
{
  "id": "maalfrid-0",
  "text": "Elever med annet morsmål enn norsk og samisk har rett til særskilt norskopplæring til de har tilstre[...]",
  "source": "maalfrid",
  "added": "2026-01-25",
  "created": "2021-01-01, 2021-12-31",
  "token_count": 711
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
To learn more about the annotations and how to use them, see the [annotations section](https://huggingface.co/datasets/danish-foundation-models/norwegian-dynaword#annotations) of the main readme.

<p align="center">
<img src="./images/annotation_profile.png" width="700" style="margin-right: 10px;" />
</p>
<!-- END-ANNOTATION PLOTS -->


## Additional Information

## License Information

This dataset is licensed under [NLOD 2.0](https://data.norge.no/nlod/en/2.0). 
This license is derived from the original [publication](https://huggingface.co/datasets/NbAiLab/NCC), which is published by the 
[National Library of Norway](https://www.nb.no/en/).

## Filtering

This subset is the result of the following filtering from all available data splits on the [NCC](https://huggingface.co/datasets/NbAiLab/NCC). It
includes the following documents:

- Documents, which are tagged as a part of the Målfrid corpus
- Document which are classified as Norwegian (nn, nb) with a threshold of 0.75
- Document which consist of at least 10 tokens (Llama)
- Documents which are not duplicated

Too see how many documents where filtered at each step see the [log](create.py.log)


### Citation Information

If you use this source please cite the following articles:

```bibtex
@inproceedings{kummervold-etal-2022-norwegian-colossal,
  title     = {The {N}orwegian colossal corpus: A text corpus for training large {N}orwegian language models},
  author    = {Kummervold, Per E  and
               Wetjen, Freddy    and
               De la Rosa, Javier},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC)},
  year      = {2022},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  url       = {https://aclanthology.org/2022.lrec-1.410},
  pages     = {3852--3860},
  abstract  = {Norwegian has been one of many languages lacking sufficient available text to train quality language models. In an attempt to bridge this gap, we introduce the Norwegian Colossal Corpus (NCC), which comprises 49GB of clean Norwegian textual data containing over 7B words. The NCC is composed of different and varied sources, ranging from books and newspapers to government documents and public reports, showcasing the various uses of the Norwegian language in society. The corpus contains mainly Norwegian Bokmål and Norwegian Nynorsk. Each document in the corpus is tagged with metadata that enables the creation of sub-corpora for specific needs. Its structure makes it easy to combine with large web archives that for licensing reasons could not be distributed together with the NCC. By releasing this corpus openly to the public, we hope to foster the creation of both better Norwegian language models and multilingual language models with support for Norwegian.},
}

@inproceedings{kummervold-etal-2021-operationalizing,
  title     = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
  author    = {Kummervold, Per E  and
               De la Rosa, Javier  and
               Wetjen, Freddy  and
               Brygfjeld, Svein Arne},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  year      = {2021},
  address   = {Reykjavik, Iceland (Online)},
  publisher = {Linköping University Electronic Press, Sweden},
  url       = {https://aclanthology.org/2021.nodalida-main.3},
  pages     = {20--29},
  abstract  = {In this work, we show the process of building a large-scale training set from digital and digitized collections at a national library.
  The resulting Bidirectional Encoder Representations from Transformers (BERT)-based language model for Norwegian outperforms multilingual BERT (mBERT) models
  in several token and sequence classification tasks for both Norwegian Bokmål and Norwegian Nynorsk. Our model also improves the mBERT performance for other
  languages present in the corpus such as English, Swedish, and Danish. For languages not included in the corpus, the weights degrade moderately while keeping strong multilingual properties. Therefore,
  we show that building high-quality models within a memory institution using somewhat noisy optical character recognition (OCR) content is feasible, and we hope to pave the way for other memory institutions to follow.},
}
```
