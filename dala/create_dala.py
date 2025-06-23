"""Create the DaLA datasets and upload them to the HF Hub."""
import warnings

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi

from pandas.errors import SettingWithCopyWarning
from requests.exceptions import HTTPError

from dala_corrupt import corrupt_dala
from load_ud import load_dadt_pos
from dala_utils import join_tokens

MIN_NUM_CHARS_IN_DOCUMENT = 2
MAX_NUM_CHARS_IN_DOCUMENT = 5000

# ScaLA proportions (USE_SPLIT_PROPORTIONS = False)
# train = 512*2 = 1024 samples
# test = 1024*2 = 2048 samples
# validation = 128*2 = 256 samples

USE_SPLIT_PROPORTIONS = False

# DaLA medium proportions
TRAIN_PROPORTION = 0.6
TEST_PROPORTION = 0.35

# DaLA large proportions
# TRAIN_PROPORTION = 0.8
# TEST_PROPORTION = 0.15

DATASET_ID = "giannor/dala_medium"


def main(use_split_proportions: bool) -> DatasetDict[str, Dataset] | None:
    """Create the DaLA dataset and upload it to the HF Hub."""
    lang = "da"
    print(f"Creating DaLA dataset...")

    # Load the POS dataset
    pos_dataset = load_dadt_pos()

    # Merge the DDT POS dataframes to a single dataframe, with columns `ids`,
    # `tokens`, `doc` and `pos_tags`
    df = pd.concat(pos_dataset.values(), ignore_index=True)

    # Drop the duplicates
    df = df.drop_duplicates(subset="doc").reset_index(drop=True)

    # Remove samples with five or fewer tokens
    df = df[df.tokens.map(lambda lst: len(lst) > 5)]

    # Remove samples with five or fewer distinct POS tags
    df = df[df.pos_tags.map(lambda lst: len(set(lst)) > 5)]

    # Remove samples with an odd number of quotes
    df = df[df.doc.map(lambda doc: doc.count('"') % 2 == 0)]

    # Remove samples which starts with punctuation
    df = df[df.pos_tags.map(lambda lst: lst[0] not in ["PUNCT", "SYM"])]

    # Remove samples containing more than one '=' character, as this is used to
    # indicate a tag
    df = df[df.doc.map(lambda doc: doc.count("=") <= 1)]

    # Remove samples containing 'SLUTORD', as this is used to indicate a tag
    df = df[~df.doc.str.contains("SLUTORD")]

    # Create a training, validation, and test set. Note that we
    # will corrupt the data, so this is only half the size of the final
    # datasets. In the case where the dataframe does not contain enough samples
    # for all the splits, we keep halving the test size until we have enough
    # samples.
    full_dataset_size = len(df)

    if use_split_proportions:
        train_size = int(full_dataset_size * TRAIN_PROPORTION)
        test_size = int(full_dataset_size * TEST_PROPORTION)
        val_size = full_dataset_size - train_size - test_size
    else:
        train_size = 512
        test_size = 1024
        val_size = 128

    while test_size >= 128:
        try:
            val_df = df.sample(n=val_size, random_state=4242)
            df_filtered = df[~df.index.isin(val_df.index)]
            test_df = df_filtered.sample(n=test_size, random_state=4242)
            full_train_df = df_filtered[~df_filtered.index.isin(test_df.index)]
            train_df = full_train_df.sample(n=train_size, random_state=4242)
            break
        except ValueError:
            test_size //= 2
    else:
        raise ValueError(
            f"Not enough samples to create the splits. Found {len(df):,} "
            f"samples, but need at least 768."
        )

    # Only work with samples where the document is not very large or small We do
    # it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.doc.str.len()
    new_train_df = (new_train_df
                    .query(f"text_len >= {MIN_NUM_CHARS_IN_DOCUMENT}")
                    .query(f"text_len <= {MAX_NUM_CHARS_IN_DOCUMENT}")
                    )

    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.doc.str.len()
    new_val_df = (new_val_df
                  .query(f"text_len >= {MIN_NUM_CHARS_IN_DOCUMENT}")
                  .query(f"text_len <= {MAX_NUM_CHARS_IN_DOCUMENT}")
                  )

    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.doc.str.len()
    new_test_df = (new_test_df
                   .query(f"text_len >= {MIN_NUM_CHARS_IN_DOCUMENT}")
                   .query(f"text_len <= {MAX_NUM_CHARS_IN_DOCUMENT}")
                   )

    new_full_train_df = full_train_df.copy()
    new_full_train_df["text_len"] = new_full_train_df.doc.str.len()
    new_full_train_df = (new_full_train_df
                         .query(f"text_len >= {MIN_NUM_CHARS_IN_DOCUMENT}")
                         .query(f"text_len <= {MAX_NUM_CHARS_IN_DOCUMENT}")
                         )

    # Add the corrupted data and turn the dataframes into Hugging Face Dataset objects
    train = prepare_df(new_train_df, split="train")
    val = prepare_df(new_val_df, split="val")
    test = prepare_df(new_test_df, split="test")
    if not use_split_proportions:
        full_train = prepare_df(new_full_train_df, split="train")
        dataset = DatasetDict(
            train=train, val=val, test=test, full_train=full_train
        )
    else:
        dataset = DatasetDict(
            train=train, val=val, test=test
        )

    # Save the dataset to CSV files
    train.to_csv(f"../la_output/dala_{lang}_train.csv")
    val.to_csv(f"../la_output/dala_{lang}_val.csv")
    test.to_csv(f"../la_output/dala_{lang}_test.csv")
    if not use_split_proportions:
        full_train.to_csv(f"../la_output/dala_{lang}_full_train.csv")

    # Create dataset ID
    dataset_id = DATASET_ID

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    print(f"DaLA: pushing dataset to HuggingFace ({dataset_id})...")
    dataset.push_to_hub(dataset_id, private=True)




def prepare_df(df: pd.DataFrame, split: str) -> Dataset:
    """Prepare a dataframe by adding an equal number of corruptions to it.

    :param df: The dataframe to prepare.
    :param split: The split to prepare the dataframe for.
    :return: The prepared dataset.
    """
    print(f"DaLA: Creating {split} split...")

    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Get the corrupted strings
    corrupted_list = corrupt_dala(df)

    # Add the corrupted strings to the dataframe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        df["corrupted"] = [tup[0] for tup in corrupted_list]
        df["corruption_type"] = [tup[1] for tup in corrupted_list]

    # Restructure the dataframe to have columns 'text', 'corruption_type' and 'label', with one sample per row
    df = pd.concat(
        [
            pd.DataFrame(
                dict(
                    text=df.tokens.map(join_tokens).tolist(),
                    corruption_type=[None for _ in range(len(df))],
                    label=["correct" for _ in range(len(df))],
                )
            ),
            pd.DataFrame(
                dict(
                    text=df.corrupted.explode().tolist(),
                    corruption_type=df.corruption_type.explode().tolist(),
                    label=["incorrect" for _ in range(len(df))],
                )
            ),
        ]
    )

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    print(f"DaLA: {split} created")

    # Convert the dataframe to a Hugging Face Dataset and return it
    return Dataset.from_pandas(df, split=split)


if __name__ == "__main__":
    main(use_split_proportions=USE_SPLIT_PROPORTIONS)
