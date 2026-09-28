---
pretty_name: IGC-Law
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

# Dataset Card for IGC-Law

<!-- START-SHORT DESCRIPTION -->
Icelandic proposals, parliamentary bills, and laws.
<!-- END-SHORT DESCRIPTION -->

This dataset bundles three distinct subcorpora, each identified in the `id` by the prefix:

- `Law1` (Proposals and Resolutions): parliamentary documents relating to resolutions (instruction from the Alþingi to the government) submitted to Alþingi (1988–2023). Counting documents, 61.3% are *tillögur til þingsályktunar* (proposals for a parliamentary resolution), a sixth of those submitted by the government as *stjórnartillögur*; 18.7% are committee reports (*nefndarálit*), 15.7% the adopted *þingsályktanir* themselves and 4.1% amendment motions (*breytingartillögur*). The remaining 0.2% are motions of no confidence, agenda motions and motions to dismiss. 
- `Law2` (Bills): parliamentary documents relating to bills submitted to Alþingi (1988–2023). 95.9% are the full parliamentary paper — the draft act, usually followed by an explanatory report. The remaining 4.1% are standalone explanatory reports (*greinargerðir*) with no draft act.
- `Law3` (Law): the current Icelandic law as of 2023-09-01, spanning enactment dates from 1275 to 2023.

`Law2` accounts for 71.9% of the tokens, `Law1` 22.9% and `Law3` 5.2%.

Each document (bill, proposal, or law) is a single Dynaword sample. 

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 17.41K
- **Number of tokens (Llama 3)**: 163.02M
- **Average document length in tokens (min, max)**: 9.36K (40, 556.48K)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "igc-law_Law2_111_1",
  "text": "1. Frumvarp til laga um verðbréfaviðskipti og verðbréfasjóði.\n\n(Lagt fyrir Alþingi á 111. löggjafarþ[...]",
  "source": "igc-law",
  "added": "2026-08-11",
  "created": "1988-10-11, 1988-10-11",
  "token_count": 33185
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
This dataset combines two releases: IGC-Law 22.10 and the extension IGC-Law 2024ext. For `Law1`/`Law2`, the two releases are combined. For `Law3`, only the 2024ext snapshot is used, since it republishes the entire current-law corpus rather than extending it.

The subcorpus is encoded in each document's `id` via the IGC's own identifier scheme (`Law1`, `Law2`, or `Law3`, see above). Text extraction was tailored per subcorpus:

- `Law1` (Proposals and Resolutions): each document's `<head>` element is a procedural boilerplate (document number, session label, reprint/correction notices) — the real title is always the first paragraph instead, so `<head>` is dropped.
- `Law2` (Bills): each document's `<head>` is its real title, so it is kept and joined with the body paragraphs. For the 388 standalone explanatory reports the `<head>` is only the generic word `Greinargerð.`, so the bill's name is taken from the bibliographic title field instead, as for `Law3`.
- `Law3` (Law): the source has no `<head>` at all, so the law's name is taken from its bibliographic title field and prepended to the body paragraphs.

The releases 22.10 and 2024ext both contain bills and proposals from year 2021. The overlap has been deduplicated using the dynaword deduplicator. 

### Limitations

Bills and proposals are frequently re-submitted in a later session, often verbatim. Hence 2,734 documents have at least one near-duplicate elsewhere (8-word shingles, Jaccard ≥ 0.8); collapsing each cluster to a single document would remove 1,556 of them (11.9% of the tokens). Both figures are lower bounds.

Smaller repeats sit on top of that: in 1,029 cases an adopted *þingsályktun* appears alongside the proposal it came from (0.65% of the subset's tokens), and 43 `Law2` bills are near-identical to the act as enacted in `Law3`. All of this is how Alþingi publishes rather than a processing artefact, and it is left as published.

The corpus is legal and legislative text, so it reflects a formal register rather than general Icelandic. 


### Source Data
The source publisher is [The Árni Magnússon Institute for Icelandic Studies](https://www.arnastofnun.is/). This dataset combines two releases: IGC-Law 22.10 (`http://hdl.handle.net/20.500.12537/247`) and the extension, IGC-Law 2024ext (`http://hdl.handle.net/20.500.12537/353`).

### Citation Information

If you use this dataset, cite the source releases:
>Barkarson, Starkaður, 2022, IGC-Law 22.10 (unannotated version), CLARIN-IS, http://hdl.handle.net/20.500.12537/247.
and
>Barkarson, Starkaður and Steingrímsson, Steinþór, 2024, IGC-Law 2024ext (unannotated version), CLARIN-IS, http://hdl.handle.net/20.500.12537/353.

## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
