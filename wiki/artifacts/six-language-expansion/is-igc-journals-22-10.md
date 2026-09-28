---
pretty_name: IGC-Journals 22.10
language:
- is
license: cc-by-4.0
license_name: CC-BY 4.0
size_categories:
- 1M<n<10M
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

# Dataset Card for IGC-Journals 22.10

<!-- START-SHORT DESCRIPTION -->
Icelandic scholarly and scientific journal articles.
<!-- END-SHORT DESCRIPTION -->

The source collects scientific and scholarly articles from 23 Icelandic journals and websites, published between 1979 and 2021, spanning fields such as law, medicine, linguistics, archaeology, and the humanities. See [Source Data](#source-data) for the full list of journals and websites.

To satisfy publishers' copyright requirements, the sentences within each article were randomly reshuffled by the source, so their original order is lost. Because of this, each document in this dataset is a single sentence rather than a whole article (see [Limitations](#limitations)).

## Dataset Description

<!-- START-DESC-STATS -->
- **Number of samples**: 1.05M
- **Number of tokens (Llama 3)**: 58.31M
- **Average document length in tokens (min, max)**: 55.48 (4, 945)
<!-- END-DESC-STATS -->

## Dataset Structure
An example from the dataset looks as follows.

<!-- START-SAMPLE -->
```py
{
  "id": "igc-journals-22-10_aif-aif_3_1.1",
  "text": "Jochums sen segir að lærðu Íslendingar að hirða og flokka íslensku ullina, yki það verðmæti prjónles[...]",
  "source": "igc-journals-22-10",
  "added": "2026-07-01",
  "created": "2004-01-01, 2004-12-31",
  "token_count": 56
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


### Limitations

The source is reshuffled on sentence-level to comply with publisher copyright requirements as such we distribute each sentence as it own document. The data is therefore useful for sentence- and token-level language modelling, vocabulary, and morphology, but is unsuitable for tasks that rely on discourse structure, inter-sentence coherence, or long-range context. Identical sentences that recur across articles (e.g. shared headings or citations) are de-duplicated, so each unique sentence appears once.

The corpus is scholarly and scientific writing, so it reflects an academic register and topic distribution rather than general Icelandic. Although the text is overwhelmingly Icelandic, individual sentences occasionally contain quotations or terms in other languages (e.g. English or Latin), intrinsic to the academic sources.


### Source Data
We download the corpus from the official CLARIN-IS repository release of IGC-Journals 22.10 (unannotated version). The source publisher is [The Árni Magnússon Institute for Icelandic Studies](https://www.arnastofnun.is/), and the repository handle for this release is `http://hdl.handle.net/20.500.12537/245`.

The source's own abbreviation for each journal or website is kept as a prefix in each sample's `id` (e.g. `igc-journals-22-10_aif-aif_3_1.1` comes from `aif`, below). The 23 sources are:

| Abbreviation | Title | Field |
|---|---|---|
| aif | Árbók Hins íslenska fornleifafélags | Archaeology |
| bli | Bliki, tímarit um fugla | Ornithology |
| gr | Gripla | Medieval studies |
| hr | Hugrás | Humanities |
| im | Íslenskt mál og almenn málfræði | Linguistics |
| ith | Íslenska þjóðfélagið | Sociology |
| lb | Læknablaðið | Medicine |
| lf | Lögfræðingur | Law |
| ljo | Ljósmæðrablaðið | Midwifery |
| mf | Málfregnir | Icelandic language policy |
| ne | Netla | Education |
| rg | Ritröð Guðfræðistofnunar | Theology |
| ri | Ritið: tímarit Hugvísindastofnunar | Humanities |
| ski | Skírnir | Literature and culture |
| ss | Stjórnmál og stjórnsýsla | Political science and public administration |
| tf | Tímarit félagsráðgjafa | Social work |
| th | Tímarit hjúkrunarfræðinga | Nursing |
| tlf | Tímarit lögfræðinga | Law |
| tlr | Tímarit Lögréttu | Law |
| tu | Tímarit um uppeldi og menntun | Education |
| tv | Tímarit um viðskipti og efnahagsmál | Business and economics |
| vt | Verktækni | Engineering |
| vv | Vísindavefurinn | General science (Q&A website) |


### Citation Information

If you use this subset, cite the source release:

> Barkarson, Starkaður; et al., 2022, IGC-Journals 22.10 (unannotated version), CLARIN-IS, http://hdl.handle.net/20.500.12537/245.


Suggested BibTeX for the source release:

```bibtex
 @misc{20.500.12537/245,
 title = {{IGC}-Journals 22.10 (unannotated version)},
 author = {Barkarson, Starka{\dh}ur and Steingr{\'{\i}}msson, Stein{\th}{\'o}r and Hafsteinsd{\'o}ttir, Hildur and Andr{\'e}sd{\'o}ttir, {\TH}{\'o}rd{\'{\i}}s Dr{\"o}fn and Eir{\'{\i}}ksd{\'o}ttir, Inga Gu{\dh}r{\'u}n and Magn{\'u}sson, Bolli and Magn{\'u}sson, {\'A}rni Dav{\'{\i}}{\dh}},
 url = {http://hdl.handle.net/20.500.12537/245},
 note = {{CLARIN}-{IS}},
 copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
 year = {2022} }
```


## License Information

The source corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
