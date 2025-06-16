from typing import List
import re

from spacy.tokens import Token
from dala_enums import GenitiveTypeEnum

def is_negative(doc) -> bool:
    """
    Easy check for negation in the doc.
    Uses two strategies:
      - Look for morphological feature Polarity=Neg
      - Look for common negation words in Danish: 'ikke', 'aldrig', 'ingen', 'intet', etc.
    """
    negation_words = {"ikke", "aldrig", "ingen", "intet", "ingenting"}
    for token in doc:
        # Check morphological features
        if "Polarity=Neg" in token.morph:
            return True
        # Check typical negation words (case-insensitive)
        if token.text.lower() in negation_words:
            return True
    return False

def is_genitive(token: Token) -> (bool, str):
    """
    Check for different genitive forms in Danish.
    """
    text = token.text
    morph = token.morph

    if "case=Gen" in morph:
        if text.endswith("'s"):
            return True, GenitiveTypeEnum.TYPE_3
        elif text.endswith("s"):
            return True, GenitiveTypeEnum.TYPE_1
        elif text.endswith("'"):
            return True, GenitiveTypeEnum.TYPE_2
        return False, None
    elif token.pos_ in ["PROPN", "NOUN"]:
        if text.endswith("'s"):
            return True, GenitiveTypeEnum.TYPE_3
        elif text.endswith("'"):
            return True, GenitiveTypeEnum.TYPE_2
    return False, None

def is_question(doc) -> bool:
    """
    Easy interrogative check for the doc.
    """
    return any(token.text == "?" for token in doc)

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

def has_antecedent_before(doc) -> (bool, dict):
    """
    Check if the first 'han' or 'hun' in the sentence has a potential
    referent appearing before it in the same sentence.
    """
    # Find the first occurrence of 'han' or 'hun'
    pronoun = None
    for token in doc:
        if token.text.lower() in ["han", "hun"]:
            pronoun = token

            # Find potential antecedents (noun phrases) before the pronoun
            potential_antecedents = []

            # Extract named entities before the pronoun
            for ent in doc.ents:
                if ent.end <= pronoun.i:  # Entity appears before pronoun
                    potential_antecedents.append({
                        "text": ent.text,
                        "type": "named_entity",
                        "position": ent.start
                    })

            # Check if any antecedents were found
            if potential_antecedents:
                # Sort by position (in case we want the closest one)
                potential_antecedents.sort(key=lambda x: pronoun.i - x["position"])

                return True, {
                    "pronoun": pronoun.text,
                    "pronoun_position": pronoun.i,
                    "potential_referents": [ant["text"] for ant in potential_antecedents]
                }

    return False, {"message": "No valid pronoun found in the sentence"}


def flip_preserving_caps(original: str, flip_to: str) -> str:
    """
    Flips the flip_to word while preserving the capitalization of the first character from the original word.
    """
    if original and original[0].isupper():
        return flip_to.capitalize()
    return flip_to
