"""Create the DaLA datasets and upload them to the HF Hub."""
import os
import warnings
from pathlib import Path

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import HfApi
from dotenv import load_dotenv

from pandas.errors import SettingWithCopyWarning

from dala_corrupt import corrupt_dala
from load_ud import load_dadt_pos
from dala_utils import join_tokens

load_dotenv('envs.env')
hf_token = os.getenv("HF_TOKEN")

if hf_token is None or hf_token == "":
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in envs.env.")

MIN_NUM_CHARS_IN_DOCUMENT = 2
MAX_NUM_CHARS_IN_DOCUMENT = 5000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = "tv2r"  # Options: "ud", "tv2r"
TV2R_DALA_INPUT_PATH = PROJECT_ROOT / "it_version" / "data" / "tv2r_dala_input.parquet"
CORRUPTION_NUM_WORKERS = min(7, max(1, (os.cpu_count() or 2) - 1))
CORRUPTION_CHUNK_SIZE = 500
CORRUPTION_RANDOM_SEED = 4242

# ScaLA proportions (set USE_SPLIT_PROPORTIONS = False)
# train = 512*2 = 1024 samples
# test = 1024*2 = 2048 samples
# validation = 128*2 = 256 samples

# DaLA medium proportions
# TRAIN_PROPORTION = 0.6
# TEST_PROPORTION = 0.35
# SIZE_NAME = "medium_"

# DaLA large proportions
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.15
SIZE_NAME = ""

USE_SPLIT_PROPORTIONS = True

CREATE_GENERATIVE = True
INCLUDE_CORRECT = True # If True and CREATE_GENERATIVE is True, the non corrupted sentences as included as samples

# If creating generative version, keep split proportions and optionally
# sample fixed split sizes after filtering.
GEN_TRAIN_SIZE = None
GEN_TEST_SIZE = None
GEN_VAL_SIZE = None

if CREATE_GENERATIVE:
    USE_SPLIT_PROPORTIONS = True
    TRAIN_PROPORTION = 0.8
    TEST_PROPORTION = 0.15
    SIZE_NAME = ""

GEN_STR = "gen_"

VERSION = "tv2r"

if not CREATE_GENERATIVE:
    GEN_STR = ""

if not USE_SPLIT_PROPORTIONS:
    SIZE_NAME = ""

DATASET_ID = f"giannor/dala_{GEN_STR}{SIZE_NAME}{VERSION}"

EXCLUDED_GENERATIVE_CORRUPTIONS = {"delete", "flip_neighbours"}

def filter_and_sample_split(
    dataset_split: Dataset,
    split_name: str,
    sample_size: int | None = None,
) -> Dataset:
    """Filter disallowed corruption types and optionally sample a fixed-size split."""
    filtered = dataset_split.filter(
        lambda example: example.get("corruption_type") not in EXCLUDED_GENERATIVE_CORRUPTIONS
    )

    if sample_size is None:
        return filtered

    if len(filtered) < sample_size:
        raise ValueError(
            f"Not enough samples in {split_name} after filtering "
            f"({len(filtered)} < {sample_size})."
        )

    return filtered.shuffle(seed=4242).select(range(sample_size))

def load_source_dataframe(
    source_dataset: str,
    source_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a sentence dataframe in the format expected by the corruption pipeline."""
    if source_dataset == "ud":
        # Load the POS dataset
        pos_dataset = load_dadt_pos()

        # Merge the DDT POS dataframes to a single dataframe, with columns `ids`,
        # `tokens`, `doc` and `pos_tags`
        return pd.concat(pos_dataset.values(), ignore_index=True)

    if source_dataset == "tv2r":
        path = Path(source_path) if source_path is not None else TV2R_DALA_INPUT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"TV2R DALA input file not found: {path}. "
                "Create it with: python it_version/preprocess.py --write-dala-input"
            )

        df = pd.read_parquet(path)
        required_columns = {"tokens", "doc", "pos_tags"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required TV2R DALA input column(s): {missing}")
        for column in ["ids", "tokens", "pos_tags"]:
            if column in df.columns:
                df[column] = df[column].map(as_python_list)
        return df

    raise ValueError("source_dataset must be either 'ud' or 'tv2r'.")


def as_python_list(value):
    """Normalize parquet nested values to lists expected by corruption code."""
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def main(
    use_split_proportions: bool,
    create_generative_version=False,
    version="test",
    source_dataset: str = SOURCE_DATASET,
    source_path: str | Path | None = None,
    corruption_num_workers: int = CORRUPTION_NUM_WORKERS,
    corruption_chunk_size: int = CORRUPTION_CHUNK_SIZE,
    corruption_random_seed: int = CORRUPTION_RANDOM_SEED,
) -> DatasetDict[str, Dataset] | None:
    """Create the DaLA dataset and upload it to the HF Hub."""
    lang = "da"

    if create_generative_version:
        print(f"Creating DaLA dataset (generative version)...")
        gen_str = "gen_"
    else:
        print(f"Creating DaLA dataset...")
        gen_str = ""

    train_out_file = f"../la_output/dala_{lang}_{gen_str}{SIZE_NAME}{version}_train.csv"
    val_out_file = f"../la_output/dala_{lang}_{gen_str}{SIZE_NAME}{version}_val.csv"
    test_out_file = f"../la_output/dala_{lang}_{gen_str}{SIZE_NAME}{version}_test.csv"
    full_train_out_file = f"../la_output/dala_{lang}_{gen_str}{SIZE_NAME}{version}_full_train.csv"

    df = load_source_dataframe(source_dataset=source_dataset, source_path=source_path)

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
    train = prepare_df(
        new_train_df,
        split="train",
        create_generative_version=create_generative_version,
        corruption_num_workers=corruption_num_workers,
        corruption_chunk_size=corruption_chunk_size,
        corruption_random_seed=corruption_random_seed,
    )
    val = prepare_df(
        new_val_df,
        split="val",
        create_generative_version=create_generative_version,
        corruption_num_workers=corruption_num_workers,
        corruption_chunk_size=corruption_chunk_size,
        corruption_random_seed=corruption_random_seed,
    )
    test = prepare_df(
        new_test_df,
        split="test",
        create_generative_version=create_generative_version,
        corruption_num_workers=corruption_num_workers,
        corruption_chunk_size=corruption_chunk_size,
        corruption_random_seed=corruption_random_seed,
    )

    # Generative: exclude basic corruptions and optionally sample fixed split sizes
    if create_generative_version:
        train = filter_and_sample_split(train, "train", GEN_TRAIN_SIZE)
        val = filter_and_sample_split(val, "val", GEN_VAL_SIZE)
        test = filter_and_sample_split(test, "test", GEN_TEST_SIZE)

    if not use_split_proportions:
        full_train = prepare_df(
            new_full_train_df,
            split="train",
            create_generative_version=create_generative_version,
            is_full_train=True,
            corruption_num_workers=corruption_num_workers,
            corruption_chunk_size=corruption_chunk_size,
            corruption_random_seed=corruption_random_seed,
        )
        dataset = DatasetDict(
            train=train, val=val, test=test, full_train=full_train
        )
    else:
        dataset = DatasetDict(
            train=train, val=val, test=test
        )

    # Save the dataset to CSV files
    train.to_csv(train_out_file)
    val.to_csv(val_out_file)
    test.to_csv(test_out_file)
    if not use_split_proportions:
        full_train.to_csv(full_train_out_file)

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(DATASET_ID, repo_type="dataset")
    except RepositoryNotFoundError:
        pass

    # Push the dataset to the Hugging Face Hub
    print(f"DaLA: pushing dataset to HuggingFace ({DATASET_ID})...")
    dataset.push_to_hub(DATASET_ID, private=True, token=hf_token)


def prepare_df(
    df: pd.DataFrame,
    split: str,
    create_generative_version: bool,
    is_full_train=False,
    corruption_num_workers: int = CORRUPTION_NUM_WORKERS,
    corruption_chunk_size: int = CORRUPTION_CHUNK_SIZE,
    corruption_random_seed: int = CORRUPTION_RANDOM_SEED,
) -> Dataset:
    """Prepare a dataframe by adding an equal number of corruptions to it.

    :param df: The dataframe to prepare.
    :param split: The split to prepare the dataframe for.
    :return: The prepared dataset.
    """
    if is_full_train:
        temp_split_name = "full_train"
    else:
        temp_split_name = split

    print(f"DaLA: Creating {temp_split_name} split...")

    if is_full_train:
        print(f"\tINFO: Standard proportions do not include all possible samples train/val/test splits. This split contains all of them.")

    if create_generative_version:
        print(f"\tINFO: Creating generative version of the split.")
        if INCLUDE_CORRECT:
            print(f"\tINFO: Including correct sentences in the generative split.")

    print(
        f"\tINFO: Corrupting with {corruption_num_workers} worker(s) "
        f"(chunk size: {corruption_chunk_size})."
    )

    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Get the corrupted strings (corrupted, corruption_type, original, affected_token_1, affected_token_2)
    corrupted_list = corrupt_dala(
        df,
        create_generative_version,
        num_workers=corruption_num_workers,
        chunk_size=corruption_chunk_size,
        random_seed=corruption_random_seed,
    )

    # Add the corrupted strings to the dataframe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        df["corrupted"] = [tup[0] for tup in corrupted_list]
        df["corruption_type"] = [tup[1] for tup in corrupted_list]
        df["original"] = [tup[2] for tup in corrupted_list]
        df["affected_token_1"] = [tup[3] for tup in corrupted_list]
        df["affected_token_2"] = [tup[4] for tup in corrupted_list]

    if create_generative_version:
        # Restructure the dataframe to have columns 'original', 'corrupted', 'corruption_type', with one sample per row
        df = pd.DataFrame(
            dict(
                original=df.original.tolist(),
                corrupted=df.corrupted.explode().tolist(),
                corruption_type=df.corruption_type.explode().tolist(),
                affected_token_1=df.affected_token_1.tolist(),
                affected_token_2=df.affected_token_2.tolist(),
            )
        )
        if INCLUDE_CORRECT:
            correct_df = pd.DataFrame(
                dict(
                    original=df.original.tolist(),
                    corrupted=df.original.tolist(),
                    corruption_type=[None for _ in range(len(df))],
                    affected_token_1=[None for _ in range(len(df))],
                    affected_token_2=[None for _ in range(len(df))],
                )
            )
            df = pd.concat([df, correct_df], ignore_index=True)
    else:
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

    if is_full_train:
        print(f"DaLA: full_train split created (including all samples)")
    else:
        print(f"DaLA: {split} created")

    # Convert the dataframe to a Hugging Face Dataset and return it
    return Dataset.from_pandas(df, split=split)


if __name__ == "__main__":
    main(
        use_split_proportions=USE_SPLIT_PROPORTIONS,
        create_generative_version=CREATE_GENERATIVE,
        version=VERSION,
        source_dataset=SOURCE_DATASET,
        corruption_num_workers=CORRUPTION_NUM_WORKERS,
        corruption_chunk_size=CORRUPTION_CHUNK_SIZE,
        corruption_random_seed=CORRUPTION_RANDOM_SEED,
    )
