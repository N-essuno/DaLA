# DaLA: Danish Linguistic Acceptability Evaluation Guided by Real World Errors

## Overview
This repository contains the source code for creating the Danish Corpus of Linguistic Acceptability (DaLA). Related paper: TBD.

The corpus is designed to evaluate linguistic acceptability in Danish. Part of this code matches or has been adapted from [EuroEval](https://aclanthology.org/2023.nodalida-1.20/) (Nielsen, 2023) in order for this dataset to be easily used via the [EuroEval evaluation framework](https://euroeval.com/). 

## Citation
WIP

[//]: # (```)

[//]: # (@inproceedings{)

[//]: # (barmina2026dala,)

[//]: # (title={DaLA: Danish Linguistic Acceptability Evaluation Guided by Real World Errors},)

[//]: # (author={Gianluca Barmina and Nathalie Hau Norman and Peter Schneider-Kamp and Lukas Galke Poech},)

[//]: # (booktitle={The Fifteenth biennial Language Resources and Evaluation Conference &#40;LREC&#41;},)

[//]: # (year={2026},)

[//]: # (url={https://openreview.net/forum?id=MPJ3oXtTZl})

[//]: # (})

[//]: # (```)

## Dataset(s)
The DaLA dataset and its variants are available on Hugging Face (in parentheses the number of sentences for train, validation and test splits):
- [DaLA](https://huggingface.co/datasets/giannor/dala) (1024, 256, 2048)
- [DaLA Medium](https://huggingface.co/datasets/giannor/dala_medium) (4952, 386, 2678)
- [DaLA Large](https://huggingface.co/datasets/giannor/dala_large) (6124, 384, 1148)

## Methodology
This method starts from existing original Danish sentences (from [Universal Dependencies Danish](https://github.com/UniversalDependencies/UD_Danish-DDT)) and corrupts them in various ways to create minimal pairs. For each sentence exactly one error is introduced, and the resulting sentence is labeled as either acceptable or unacceptable. This method can be easily extended to apply more than one error to a sentence, but in this first version we only apply one, as done by many previous works (e.g. BLiMP, ScaLA), in order to be comparable with them.

## Evaluation
The evaluation of the DaLA corpus aims to ensure that the corruptions injected make the sentences not acceptable. This is performed using an hybrid approach that combines human and automatic evaluation. The human evaluation is conducted by native Danish linguists who assess the acceptability of the sentences. The automatic evaluation uses a well performing [Danish industrial tool](https://www.writeassistant.com/) (similar to grammarly) openly available. First a set of sentences is evaluated by the tool, and then the sentences not flagged by the tool are evaluated by the linguists. The sentences flagged by the tool are considered acceptable, while those not flagged are considered unacceptable.

## Running the Code

We suggest using Python $\geq$ 3.12. You can install all the required packeges using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### DaLA Dataset Creation

The DaLA creation script is located in `dala/create_dala.py` file. You can run it with the following command:

```bash
python dala/create_dala.py
```

At the top of the script, you can set some parameters for the dataset creation:

- `MIN_NUM_CHARS_IN_DOCUMENT`: Minimum number of characters in a sentence to be considered for the dataset.
- `MAX_NUM_CHARS_IN_DOCUMENT`: Maximum number of characters in a sentence to be considered for the dataset.
- `USE_SPLIT_PROPORTIONS`: If set to `True`, the dataset will be split into training, validation, and test sets with the proportions defined in
  - `TRAIN_PROPORTION`
  - `TEST_PROPORTION`
  - Validation set will be the remaining proportion.
  - If set to `False`, the dataset will be split according to original ScaLA proportions from EuroEval.
  - For DaLA Medium and DaLA Large, the proportions used (train, test) are respectively (0.6, 0.35) and (0.8, 0.15).
- `DATASET_ID`: The id of the Hugging Face dataset to be created. This will be used to upload the dataset to the Hugging Face Hub.
  - Note: for the dataset to be uploaded, you need to have a Hugging Face account and be logged in. You can log in using the `huggingface-cli login` command.

The script will also save the dataset locally in a directory named `la_output` in CSV format.

The different corruption functions are implemented and briefly documented in `data/dala_corrupt.py`. There is one function for each type of corruption. We analyzed the corruption frequency for each error type in the dataset, and we fixed the order of the corruption functions application to ensure that each error is represented as best as possible in the dataset, considering the fact that some errors are intrinsically more frequent than others.


### Data Proportions
In `data_proportion.ipynb`, you can find the code that calculates the proportions of each type of corruption in the dataset for each split, prints it and plots it. To ensure that the different splits have similar proportions, we also use calculate difference distribution distance measures between the splits.


### DaLA Dataset Evaluation
The evaluation notebook (regarding the automatic evaluation part) is located in `evaluation/write_assistant_evaluation.ipynb`. The evaluation is done at error level, meaning that each error is evaluated separately on the whole original dataset (UD Danish). For each corrupted sentence, the [Write Assistant](https://www.writeassistant.com/) tool is used to check if the sentence is flagged as acceptable or not and we output:

- Number of corrupted sentences
- Number of sentences flagged as unacceptable (True Positives)
- Number of sentences flagged as acceptable (False Positives)
- Precision

We consider only precision because we are only interested in evaluating the corruption quality. This means that we consider and evaluate only the corrupted (positives) sentences.


### Models evaluation on DaLA
The model evaluation is not included in this repository as it is based on the original [EuroEval](https://euroeval.com/) evaluation framework. However, since this dataset matches the linguistic acceptability dataset format expected by EuroEval the process to replicate the evaluation is straightforward:

- Install [EuroEval](https://github.com/EuroEval/EuroEval)
- Go to EuroEval's folder in your virtual environment (e.g. miniconda3/envs/,your_env>/lib/python3.x/site-packages/euroeval/) 
- Go to `dataset_configs/danish.py`
- In `SCALA_DA_CONFIG` replace the `huggingface_id` value with the Hugging Face dataset ID you set in the `create_dala.py` script (or `giannor/dala` (or one of the variants) to use the existing DaLA dataset on Hugging Face).
- After that you can run the evaluation using the EuroEval framework from the code as you would normally do, for example (for evaluating a model only on Danish linguistic acceptability):

```python
from euroeval import Benchmarker

hf_token = your_huggingface_token
model = huggingface_model_id_or_path
la_task = "linguistic-acceptability"

benchmark = Benchmarker(force=True)

benchmark(model=model, task=la_task, api_key=hf_token, verbose=True, raise_errors=True, language="da")
```

