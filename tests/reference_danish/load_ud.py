"""Load the part-of-speech part of a Universal Dependencies treebank."""
import re
from collections import defaultdict
from typing import Dict, Union, List, Callable

import pandas as pd
import requests


def load_dadt_pos() -> Dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Danish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Danish-DDT/raw/master/"
        "da_ddt-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)

def load_ud_pos(
    train_url: str,
    val_url: str,
    test_url: str,
    doc_process_fn: Callable[[str], str] = lambda x: x,
) -> Dict[str, pd.DataFrame]:
    """Load the part-of-speech part of a Universal Dependencies treebank.

    Args:
        train_url:
            The URL of the training data.
        val_url:
            The URL of the validation data.
        test_url:
            The URL of the test data.
        doc_process_fn:
            A function to apply to each document before parsing it.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Download the data
    data = dict(
        train=requests.get(train_url).text.split("\n"),
        val=requests.get(val_url).text.split("\n"),
        test=requests.get(test_url).text.split("\n"),
    )

    # Iterate over the data splits
    dfs = dict()
    for split, lines in data.items():
        # Initialise the records, data dictionary and document
        records = list()
        data_dict: Dict[str, List[Union[int, str]]] = defaultdict(list)
        doc = ""

        # Iterate over the data for the given split
        for line in lines:
            # If we are at the first line of an entry then extract the document
            if line.startswith("# text = "):
                doc = re.sub("# text = ", "", line)

                # Process the document if needed
                doc = doc_process_fn(doc)

            # Otherwise, if the line is a comment then ignore it
            elif line.startswith("#"):
                continue

            # Otherwise, if we have reached the end of an entry then store it to the
            # list of records and reset the data dictionary and document
            elif line == "":
                if len(data_dict["tokens"]) > 0:
                    merged_data_dict: Dict[str, Union[str, List[Union[int, str]]]]
                    merged_data_dict = {**data_dict, "doc": doc}
                    records.append(merged_data_dict)
                data_dict = defaultdict(list)
                doc = ""

            # Otherwise we are in the middle of an entry which is not a comment, so
            # we extract the data from the line and store it in the data dictionary
            else:
                data_tup = line.split("\t")
                data_dict["ids"].append(data_tup[0])
                data_dict["tokens"].append(data_tup[1])
                data_dict["pos_tags"].append(data_tup[3])

        # Convert the records to a dataframe and store it
        dfs[split] = pd.DataFrame.from_records(records)

    # Return the dictionary of dataframes
    return dfs