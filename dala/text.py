"""Shared token reconstruction and capitalization helpers."""
import re
from typing import List

def join_tokens(tokens: List[str]) -> str:
    """
    Joins a list of tokens into a string.
    """
    # Form document
    doc = " ".join(tokens)

    # Remove whitespace around punctuation
    doc = (
        doc.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("{ ", "{")
        .replace(" }", "}")
        .replace(" ?", "?")
        .replace(" !", "!")
    )

    # Remove whitespace around quotes
    if doc.count('"') % 2 == 0:
        doc = re.sub('" ([^"]*) "', '"\\1"', doc)

    # Return the document
    return doc

def flip_preserving_caps(original: str, flip_to: str) -> str:
    """
    Flips the flip_to word while preserving the capitalization of the first character from the original word.
    """
    if original and original[0].isupper():
        return flip_to.capitalize()
    return flip_to
