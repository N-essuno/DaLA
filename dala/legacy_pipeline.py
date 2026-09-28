"""Shared legacy row construction, driven entirely by a language profile.

Preserves original splitting order, full_train re-corruption and RNG sequencing.
"""
import warnings
from pathlib import Path
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from datasets import Dataset, DatasetDict
from .rule_engine import ConfiguredRules
from .load_ud import load_ud_pos
from .text import join_tokens

def build_legacy(profile, use_split_proportions=False, output_dir="la_output", dataset_id=None, source_frames=None, model=None) -> DatasetDict:
    """Create Danish DaLA, optionally publishing to an explicitly supplied ID."""
    language = profile['language']
    filters, split_config = profile['filters'], profile['splits']
    configured = ConfiguredRules(profile, model)
    source = profile['sources']
    if source['adapter'] != 'ud_pos':
        raise ValueError('Compatibility mode requires a UD POS source')
    pos_dataset = source_frames if source_frames is not None else load_ud_pos(**{
        name + '_url': source['url_template'].format(upstream)
        for name, upstream in source['splits'].items()})

    # Merge the DDT POS dataframes to a single dataframe, with columns `ids`,
    # `tokens`, `doc` and `pos_tags`
    df = pd.concat(pos_dataset.values(), ignore_index=True)

    # Drop the duplicates
    df = df.drop_duplicates(subset="doc").reset_index(drop=True)

    # Remove samples with five or fewer tokens
    df = df[df.tokens.map(lambda lst: len(lst) > filters['min_tokens_exclusive'])]

    # Remove samples with five or fewer distinct POS tags
    df = df[df.pos_tags.map(lambda lst: len(set(lst)) > filters['min_distinct_pos_exclusive'])]

    # Remove samples with an odd number of quotes
    df = df[df.doc.map(lambda doc: doc.count(filters['quote']) % 2 == 0)]

    # Remove samples which starts with punctuation
    df = df[df.pos_tags.map(lambda lst: lst[0] not in filters['forbidden_initial_pos'])]

    # Remove samples containing more than one '=' character, as this is used to
    # indicate a tag
    for char, maximum in filters['max_character_counts'].items():
        df = df[df.doc.map(lambda doc: doc.count(char) <= maximum)]

    # Remove samples containing 'SLUTORD', as this is used to indicate a tag
    for pattern in filters['exclude_patterns']:
        df = df[~df.doc.str.contains(pattern)]

    # Create a training, validation, and test set. Note that we
    # will corrupt the data, so this is only half the size of the final
    # datasets. In the case where the dataframe does not contain enough samples
    # for all the splits, we keep halving the test size until we have enough
    # samples.
    full_dataset_size = len(df)

    if use_split_proportions:
        train_size = int(full_dataset_size * split_config['train_proportion'])
        test_size = int(full_dataset_size * split_config['test_proportion'])
        val_size = full_dataset_size - train_size - test_size
    else:
        train_size = split_config['train_size']
        test_size = split_config['test_size']
        val_size = split_config['val_size']

    while test_size >= split_config['min_test_size']:
        try:
            val_df = df.sample(n=val_size, random_state=split_config['seed'])
            df_filtered = df[~df.index.isin(val_df.index)]
            test_df = df_filtered.sample(n=test_size, random_state=split_config['seed'])
            full_train_df = df_filtered[~df_filtered.index.isin(test_df.index)]
            train_df = full_train_df.sample(n=train_size, random_state=split_config['seed'])
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
                    .query(f"text_len >= {filters['min_chars']}")
                    .query(f"text_len <= {filters['max_chars']}")
                    )

    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.doc.str.len()
    new_val_df = (new_val_df
                  .query(f"text_len >= {filters['min_chars']}")
                  .query(f"text_len <= {filters['max_chars']}")
                  )

    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.doc.str.len()
    new_test_df = (new_test_df
                   .query(f"text_len >= {filters['min_chars']}")
                   .query(f"text_len <= {filters['max_chars']}")
                   )

    new_full_train_df = full_train_df.copy()
    new_full_train_df["text_len"] = new_full_train_df.doc.str.len()
    new_full_train_df = (new_full_train_df
                         .query(f"text_len >= {filters['min_chars']}")
                         .query(f"text_len <= {filters['max_chars']}")
                         )

    # Add the corrupted data and turn the dataframes into Hugging Face Dataset objects
    train = prepare_frame(configured, new_train_df, split="train")
    val = prepare_frame(configured, new_val_df, split="val")
    test = prepare_frame(configured, new_test_df, split="test")
    if not use_split_proportions:
        full_train = prepare_frame(configured, new_full_train_df, split="train")
        dataset = DatasetDict(
            train=train, val=val, test=test, full_train=full_train
        )
    else:
        dataset = DatasetDict(
            train=train, val=val, test=test
        )

    # Local output is the default; publishing requires an explicit dataset ID.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, data in dataset.items():
        data.to_csv(str(output / f"dala_{language}_{split}.csv"))
    if dataset_id:
        dataset.push_to_hub(dataset_id, private=True)
    return dataset



def prepare_frame(configured, df: pd.DataFrame, split: str) -> Dataset:
    """Prepare a dataframe by adding an equal number of corruptions to it.

    :param df: The dataframe to prepare.
    :param split: The split to prepare the dataframe for.
    :return: The prepared dataset.
    """
    split_config = configured.profile['splits']
    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Get the corrupted strings
    corrupted_list = configured.corrupt(df)

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
    df = df.sample(frac=1.0, random_state=split_config['seed']).reset_index(drop=True)

    print(f"DaLA: {split} created")

    # Convert the dataframe to a Hugging Face Dataset and return it
    return Dataset.from_pandas(df, split=split)
