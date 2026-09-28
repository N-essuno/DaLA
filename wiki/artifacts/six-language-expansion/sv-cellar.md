---
pretty_name: Cellar
language:
- sv
license: cc-by-sa-4.0
license_name: CC-BY-SA 4.0
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
domains:
- Legal
---

# Dataset Card for Cellar

<!-- START-SHORT DESCRIPTION -->
The official digital repository for European Union legal documents and open data.
<!-- END-SHORT DESCRIPTION -->

The EU Dataset [Cellar](https://op.europa.eu/en/web/cellar/home) serves as the central access point for all official EU publications, legislation, and open data resources. Maintained by the Publications Office of the European Union, this comprehensive digital archive contains millions of documents in multiple languages, including regulations, directives, decisions, treaties, case law, and preparatory acts dating back decades. The repository employs standardized metadata and unique identifiers to organize its vast collection, making it an essential resource for researchers, legal professionals, policymakers, and citizens seeking authoritative information on EU law and policy. The Cellar's linked data architecture also enables sophisticated search capabilities and integration with other information systems across the European Union's digital landscape.


## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 419.34K
- **Number of tokens (Llama 3)**: 4.73B
- **Average document length in tokens (min, max)**: 11.28K (3, 8.14M)
<!-- END-DESC-STATS -->


## Dataset Structure
An example from the dataset looks as follows.


<!-- START-SAMPLE -->
```py
{
  "id": "cellar_1127769",
  "text": "- ISSN 1977-0820\n- Europeiska unionensofficiella tidning L 221\n- Svensk utgåva Lagstiftning 63 årgån[...]",
  "source": "cellar",
  "added": "2026-07-09",
  "created": "2020-07-10, 2020-07-10",
  "token_count": 153759
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
The dataset was constructed using an external metadata [reference](https://huggingface.co/datasets/danish-foundation-models/cellar-metadata). The metadata set was constructed by iterating through all months since 1955 using Cellar's SPARQL endpoints.

HTML/XML files were parsed with html-to-markdown and PDF's with liteparse. 

### Limitations
Due to the size of the dataset, pdf extraction was conducted without OCR. Moreover files spanning more 100 mb were not parsed. PDF's were also truncated to 100 pages.

Some light cleanup was conducted, but the samples are not always perfect.

There is a large amount of content residing in tables, which might not always be language heavy. 

### License Information

Data is licensed under the [Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/). Thus, attribution to Cellar is required when using this data.

### Citation Information

No citation is applicable for this work.
