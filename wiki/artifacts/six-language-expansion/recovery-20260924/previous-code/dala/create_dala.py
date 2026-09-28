"""Backwards-compatible Danish entry point; implementation is profile driven."""
from pathlib import Path

MIN_NUM_CHARS_IN_DOCUMENT = 2
MAX_NUM_CHARS_IN_DOCUMENT = 5000
USE_SPLIT_PROPORTIONS = False
TRAIN_PROPORTION = 0.6
TEST_PROPORTION = 0.35


def main(use_split_proportions=False, output_dir='la_output', dataset_id=None):
    from .pipeline import build
    from .profiles import load_profile
    profile = load_profile('da')
    profile['filters'].update(min_chars=MIN_NUM_CHARS_IN_DOCUMENT, max_chars=MAX_NUM_CHARS_IN_DOCUMENT)
    profile['splits'].update(train_proportion=TRAIN_PROPORTION, test_proportion=TEST_PROPORTION)
    return build(profile, use_split_proportions=use_split_proportions, output_dir=output_dir, dataset_id=dataset_id)


def prepare_df(df, split):
    from .profiles import load_profile
    from .rule_engine import ConfiguredRules
    from .legacy_pipeline import prepare_frame
    return prepare_frame(ConfiguredRules(load_profile('da')), df, split)


if __name__ == '__main__':
    import sys
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dala.multilingual import cli
    cli()
