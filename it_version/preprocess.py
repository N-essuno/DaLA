"""Preprocess TV2R articles into a sentence-level dataset.

The source TV2R parquet contains one long Danish article per row. This script
creates a derived dataset with one sentence per row and the id of the original
article the sentence came from.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "tv2r.parquet"
DEFAULT_OUTPUT_PATH = DATA_DIR / "tv2r_sentences.parquet"
DEFAULT_DALA_OUTPUT_PATH = DATA_DIR / "tv2r_dala_input.parquet"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_ID_COLUMN = "id"
DEFAULT_OUTPUT_ID_COLUMN = "original_document_id"
DEFAULT_SPACY_MODEL = "da_core_news_md"

SentenceSplitter = Callable[[str], list[str]]

DATE_LINE_RE = re.compile(
    r"^\d{1,2}\s+[A-Za-zÆØÅæøå]+\s+\d{4}\s+kl\.\s*\d{1,2}[.:]\d{2}$"
)
SPACE_AFTER_SENTENCE_RE = re.compile(r"([.!?][\"')\]]?)(?=[A-ZÆØÅ])")
WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
LINEBREAK_RE = re.compile(r"\n+")

ABBREVIATIONS = {
    "adm",
    "adr",
    "apr",
    "aug",
    "bl.a",
    "ca",
    "cand",
    "dec",
    "dvs",
    "ekskl",
    "etc",
    "feb",
    "fig",
    "f.eks",
    "inkl",
    "jan",
    "jf",
    "jul",
    "jun",
    "kl",
    "kr",
    "maj",
    "mar",
    "mfl",
    "mia",
    "mio",
    "maks",
    "min",
    "mm",
    "mr",
    "mrs",
    "ms",
    "nov",
    "nr",
    "okt",
    "osv",
    "ph.d",
    "prof",
    "sep",
    "sept",
    "stk",
    "tlf",
    "vedr",
}


def build_sentence_splitter() -> tuple[SentenceSplitter, str]:
    """Use Danish NLTK Punkt when available, otherwise fall back to regex."""
    try:
        from nltk.tokenize import sent_tokenize

        # Trigger the data lookup once so failures happen at startup, not mid-run.
        sent_tokenize("Dette er en test. Her er en til.", language="danish")

        def nltk_splitter(text: str) -> list[str]:
            return sent_tokenize(text, language="danish")

        return nltk_splitter, "nltk-danish-punkt"
    except (ImportError, LookupError):
        return regex_sentence_split, "regex-fallback"


def regex_sentence_split(text: str) -> list[str]:
    """Small fallback splitter that avoids common Danish abbreviation splits."""
    chunks = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-ZÆØÅ0-9])", text)
    sentences: list[str] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if sentences and _should_merge_with_previous(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {chunk}"
        else:
            sentences.append(chunk)

    return sentences


def _should_merge_with_previous(sentence: str) -> bool:
    previous = sentence.rstrip("\"')]")
    last_token = previous.rsplit(" ", maxsplit=1)[-1].rstrip(".").lower()

    if last_token in ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[A-ZÆØÅ]\.)+", previous):
        return True
    return bool(re.search(r"\b[A-ZÆØÅ]\.$", previous))


def iter_text_blocks(text: str) -> Iterable[str]:
    """Yield line-aware text blocks so headlines do not attach to sentences."""
    normalized = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
    )
    normalized = SPACE_AFTER_SENTENCE_RE.sub(r"\1 ", normalized)

    for block in LINEBREAK_RE.split(normalized):
        block = WHITESPACE_RE.sub(" ", block).strip()
        if block:
            yield block


def is_metadata_block(block: str) -> bool:
    """Skip TV2R timestamp rows such as '02 april 2017 kl. 17.58'."""
    return bool(DATE_LINE_RE.fullmatch(block))


def clean_sentence(sentence: str) -> str:
    sentence = WHITESPACE_RE.sub(" ", sentence.replace("\xa0", " ")).strip()
    return sentence.strip(" \t\n\r")


def is_valid_sentence(sentence: str, min_chars: int, min_words: int) -> bool:
    if len(sentence) < min_chars:
        return False
    if len(sentence.split()) < min_words:
        return False
    if not any(char.isalpha() for char in sentence):
        return False
    return True


def split_article_into_sentences(
    text: str,
    splitter: SentenceSplitter,
    min_chars: int,
    min_words: int,
) -> list[str]:
    sentences: list[str] = []

    for block in iter_text_blocks(text):
        if is_metadata_block(block):
            continue

        for sentence in merge_sentence_fragments(splitter(block)):
            sentence = clean_sentence(sentence)
            if is_valid_sentence(sentence, min_chars=min_chars, min_words=min_words):
                sentences.append(sentence)

    return sentences


def merge_sentence_fragments(sentences: Iterable[str]) -> list[str]:
    """Merge obvious false sentence breaks after abbreviations."""
    merged: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if merged and _should_merge_with_previous(merged[-1]):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)

    return merged


def create_sentence_dataset(
    input_path: Path,
    output_path: Path,
    text_column: str,
    id_column: str,
    output_id_column: str,
    min_chars: int,
    min_words: int,
) -> pd.DataFrame:
    source = pd.read_parquet(input_path, columns=[id_column, text_column])
    missing_columns = {id_column, text_column} - set(source.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s) in {input_path}: {missing}")

    splitter, splitter_name = build_sentence_splitter()
    records: list[dict[str, str]] = []

    for document_id, text in source[[id_column, text_column]].itertuples(
        index=False, name=None
    ):
        if pd.isna(text):
            continue

        for sentence in split_article_into_sentences(
            text=text,
            splitter=splitter,
            min_chars=min_chars,
            min_words=min_words,
        ):
            records.append({"text": sentence, output_id_column: document_id})

    output = pd.DataFrame.from_records(records, columns=["text", output_id_column])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(output, output_path)

    print(
        f"Created {len(output):,} sentences from {len(source):,} documents "
        f"using {splitter_name}: {output_path}"
    )
    return output


def create_dala_input_dataset(
    sentence_path: Path,
    output_path: Path,
    text_column: str,
    id_column: str,
    spacy_model: str,
    batch_size: int,
    n_process: int,
) -> pd.DataFrame:
    """Create the token/POS format expected by dala/create_dala.py."""
    sentence_df = pd.read_parquet(sentence_path, columns=[text_column, id_column])
    nlp = load_spacy_model(spacy_model)
    records: list[dict[str, object]] = []

    contexts = (
        (str(text), document_id)
        for text, document_id in sentence_df[[text_column, id_column]].itertuples(
            index=False, name=None
        )
        if not pd.isna(text)
    )
    tagged_docs = nlp.pipe(
        contexts,
        as_tuples=True,
        batch_size=batch_size,
        n_process=n_process,
    )

    for doc, document_id in tqdm(
        tagged_docs,
        total=len(sentence_df),
        desc="Tagging TV2R sentences",
    ):
        tokens = [token.text for token in doc]
        records.append(
            {
                "ids": [str(idx) for idx in range(1, len(tokens) + 1)],
                "tokens": tokens,
                "doc": doc.text,
                "pos_tags": [token.pos_ for token in doc],
                id_column: document_id,
            }
        )

    output = pd.DataFrame.from_records(
        records,
        columns=["ids", "tokens", "doc", "pos_tags", id_column],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(output, output_path)
    print(f"Created DALA input with {len(output):,} rows: {output_path}")
    return output


def load_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required to create the DALA input dataset. "
            "Install the project requirements first."
        ) from exc

    if not spacy.util.is_package(model_name):
        raise ValueError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        )

    return spacy.load(model_name, disable=["parser", "lemmatizer", "ner"])


def write_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        dataset.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        dataset.to_csv(output_path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(
            "Unsupported output format. Use .parquet, .csv, .jsonl, or .ndjson."
        )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sentence-level TV2R dataset from the parquet export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input TV2R parquet file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output dataset path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help=f"Column containing article text. Defaults to {DEFAULT_TEXT_COLUMN}.",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help=f"Column containing source document ids. Defaults to {DEFAULT_ID_COLUMN}.",
    )
    parser.add_argument(
        "--output-id-column",
        default=DEFAULT_OUTPUT_ID_COLUMN,
        help=(
            "Name of the output document-id column. "
            f"Defaults to {DEFAULT_OUTPUT_ID_COLUMN}."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=5,
        help="Drop extracted sentences shorter than this many characters.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Drop extracted sentences shorter than this many whitespace tokens.",
    )
    parser.add_argument(
        "--write-dala-input",
        action="store_true",
        help=(
            "Also create a token/POS-tagged parquet that can be used by "
            "dala/create_dala.py with SOURCE_DATASET='tv2r'."
        ),
    )
    parser.add_argument(
        "--dala-output",
        type=Path,
        default=DEFAULT_DALA_OUTPUT_PATH,
        help=f"DALA-compatible output path. Defaults to {DEFAULT_DALA_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help=f"spaCy model used for DALA token/POS columns. Defaults to {DEFAULT_SPACY_MODEL}.",
    )
    parser.add_argument(
        "--spacy-batch-size",
        type=int,
        default=256,
        help="Batch size for spaCy POS tagging when writing DALA input.",
    )
    parser.add_argument(
        "--spacy-n-process",
        type=int,
        default=1,
        help="Number of spaCy worker processes when writing DALA input.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    create_sentence_dataset(
        input_path=parsed.input,
        output_path=parsed.output,
        text_column=parsed.text_column,
        id_column=parsed.id_column,
        output_id_column=parsed.output_id_column,
        min_chars=parsed.min_chars,
        min_words=parsed.min_words,
    )

    if parsed.write_dala_input:
        create_dala_input_dataset(
            sentence_path=parsed.output,
            output_path=parsed.dala_output,
            text_column="text",
            id_column=parsed.output_id_column,
            spacy_model=parsed.spacy_model,
            batch_size=parsed.spacy_batch_size,
            n_process=parsed.spacy_n_process,
        )


if __name__ == "__main__":
    main()
