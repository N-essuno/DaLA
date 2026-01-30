import random
from typing import List, Tuple, Union

import pandas as pd

import spacy
from spacy import Language
from tqdm import tqdm

from dala_utils import flip_preserving_caps, is_negative, is_question, is_genitive, has_antecedent_before, join_tokens
from dala_enums import GenitiveTypeEnum


def get_corruption_functions():
    return [corrupt_genitive, flip_som_der, flip_nogle_nogen, flip_ligge_laegge, corrupt_spelling,
            flip_han_hun_to_det, corrupt_adjective_r, corrupt_verb_r, corrupt_ende_ene, corrupt_noun_r,
            flip_far_for, flip_indefinite_article, flip_pronouns, flip_en_et_suffix, corrupt_basic]

class SpacyModelSingleton:
    """
    Singleton class for loading a spaCy model.
    If no model is instantiated or the model name changes, a new instance is created.
    Otherwise, the existing instance is returned.
    """
    _instance = None
    _name = None

    def __new__(cls, model_name: str = "da_core_news_md"):
        if cls._instance is None or cls._name != model_name:
            # Check if the model is downloaded, if not, download it
            if not spacy.util.is_package(model_name):
                print("Downloading spaCy model:", model_name)
                try:
                    spacy.cli.download(model_name)
                except Exception as e:
                    raise ValueError(f"Failed to download the spaCy model '{model_name}': {e}")
            try:
                cls._instance = spacy.load(model_name)
            except Exception as e:
                raise ValueError(f"Failed to load the spaCy model '{model_name}': {e}")
        return cls._instance


def corrupt_dala(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Corrupt the sentence dataframe passed as input with various types of errors.
    For now the dataframe format expected is Universal Dependencies (UD) Danish sentences.

    :param df: A DataFrame in Danish Universal Dependencies format.
    :return: A Tuple containing the list of (corrupted_string, corruption_type, original_sentence)
    """
    # Corruption function callables sorted by the proportion of sentences corruptible in UD Danish (by each function).
    # This is done to ensure that the lower-proportion corruptions are applied first in order for them to be represented.
    # Otherwise, it can happen that higher-proportion corruptions corrupt all sentences that are corruptible
    # by the lower-proportion corruptions, leading to a lack of diversity in the corrupted dataset.
    df = df.copy()

    paired_orig_corrupt_rows = []

    # Load the Danish spaCy model
    dk_model = SpacyModelSingleton("da_core_news_md")

    # Apply the corruptions to the DataFrame and remove the sentences from the original DataFrame if they are corrupted
    # so to avoid corruption of the same sentence multiple times
    corrupted_sentences = []
    corruption_functions = get_corruption_functions()
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Corrupting sentences"):
        if row["doc"] is not None:
            for func in corruption_functions:
                if func.__name__ == "corrupt_basic":
                    tokens = row["tokens"]
                    pos_tags = row["pos_tags"]
                    tuple_result = func(tokens=tokens, pos_tags=pos_tags, num_corruptions=1)[0]
                    # add doc to tuple_result for reference
                    tuple_result_new = (tuple_result[0], tuple_result[1], row["doc"])
                    corrupted_sentences.append(tuple_result_new)
                    df.at[i, "doc"] = None
                    corruption_done = True
                else:
                    corruption_done, result = func(dk_model, row["doc"])
                    if corruption_done:
                        corrupted_sentences.append((result, func.__name__, row["doc"]))
                        df.at[i, "doc"] = None
                # Corruption expected to be done, break the loop to avoid multiple corruptions and switch to the next sentence
                if corruption_done:
                    break

    return corrupted_sentences


def flip_indefinite_article(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Flip the indefinite article (en <-> et).
    Looks for a determiner child of a NOUN that is "en" or "et" and flips it.

    :param dk_model: A Danish SpaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for token in doc:
        # Skip if we already did a single corruption
        if single_corruption_done:
            break
        # We only look for NOUN tokens
        if token.pos_ == "NOUN":
            # Check all children in the dependency tree
            for child in token.children:
                # We want a determiner that is "en" or "et"
                if child.dep_ == "det" and child.text.lower() in ("en", "et"):
                    if random.random() < flip_prob:
                        # Flip "en" -> "et" or "et" -> "en"
                        flipped = "et" if child.text.lower() == "en" else "en"
                        # Preserve capitalization if needed
                        if child.text[0].isupper():
                            flipped = flipped.capitalize()
                        # Replace in tokens_out
                        tokens_out[child.i] = flipped
                        single_corruption_done = True

                        original_token = token.text
                        corrupted_token = tokens_out[child.i]

    # Rebuild the sentence
    corrupted_tokens = []
    for orig_token, new_token in zip(doc, tokens_out):
        if isinstance(new_token, str):
            corrupted_tokens.append(new_token)
        else:
            corrupted_tokens.append(new_token.text)

    final_sentence = join_tokens(corrupted_tokens)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence

def flip_en_et_suffix(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Flip the singular-definite suffix on Danish nouns (-en <-> -et).

    Finds the first token that
        - is a NOUN,
        - is definite singular,
        - ends in 'en' or 'et'
    Swaps the two-letter suffix.

    :param dk_model: A Danish SpaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        # Stop after the first successful corruption
        if single_corruption_done:
            break

        # Check for definite singular nouns
        if token.pos_ == "NOUN" and token.morph.get("Definite") == ["Def"] and token.morph.get("Number") == ["Sing"]:
            lower = token.text.lower()

            if (lower.endswith("en") or lower.endswith("et")) and len(lower) > 2:
                if random.random() < flip_prob:
                    flipped_suffix = "et" if lower.endswith("en") else "en"
                    stem = token.text[:-2]

                    if token.text.isupper():
                        flipped_suffix = flipped_suffix.upper()

                    flipped = stem + flipped_suffix

                    tokens_out[i] = flipped
                    single_corruption_done = True
                    original_token = token.text
                    corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def flip_nogle_nogen(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Flip 'nogle' <-> 'nogen'
    Check the next NOUN and type of sentence to determine if the flip is valid:
        - 'nogle' -> 'nogen' if the next NOUN is plural and the sentence is not negative or a question.
        - 'nogen' -> 'nogle' if the next NOUN is singular.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    # Check negation/question
    found_negation = is_negative(doc)
    found_question = is_question(doc)

    for i, token in enumerate(doc):
        # Skip if we already did a single corruption
        if single_corruption_done:
            break
        lower_t = token.text.lower()

        # Check if token is 'nogle' or 'nogen'
        if lower_t in ("nogle", "nogen") and random.random() < flip_prob:
            # Find the next NOUN (if any) to check its number
            next_noun = None
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "NOUN":
                    next_noun = doc[j]
                    break

            if next_noun:
                # Get morphological number (singular/plural)
                noun_number = next_noun.morph.get("Number")  # e.g. ['Sing'] or ['Plur']

                # Rule: 'nogle' -> 'nogen' if next noun is plural AND
                #       the sentence is NOT negative or question.
                if lower_t == "nogle" and "Plur" in noun_number:
                    # Now skip flipping if we found negation or a question
                    if not found_negation and not found_question:
                        tokens_out[i] = flip_preserving_caps(token.text, flip_to="nogen")
                        single_corruption_done = True
                        original_token = token.text
                        corrupted_token = tokens_out[i]

                # Rule: 'nogen' -> 'nogle' if next noun is singular
                if lower_t == "nogen" and "Sing" in noun_number:
                    tokens_out[i] = flip_preserving_caps(token.text, flip_to="nogle")
                    single_corruption_done = True
                    original_token = token.text
                    corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence

def corrupt_ende_ene(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Introduce the common "silent d mistake" in Danish

    Flip -ende (for adjectives/participles) <-> -ene (for definite plural nouns).

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        # Skip if we already did a single corruption
        if single_corruption_done:
            break
        word = token.text
        lower_word = word.lower()

        if random.random() < flip_prob:

            # NOUN that ends with -ene -> flip to -ende
            if token.pos_ == "NOUN" and lower_word.endswith("ene") and len(word) >= 3:
                stem = word[:-3]
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "ende"
                else:
                    tokens_out[i] = stem + "ende"
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

            # ADJ/VERB that ends with -ende --> flip to -ene
            elif token.pos_ in ("ADJ", "VERB") and lower_word.endswith("ende") and len(word) >= 4:
                stem = word[:-4]  # remove "ende"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "ene"
                else:
                    tokens_out[i] = stem + "ene"
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def flip_pronouns(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Inject subject/object-pronoun errors in a sentence.

    - If a subject pronoun is used in a subject position, flip it to its object form.
    - If an object pronoun is used in a direct/indirect/oblique-object position, flip it to its subject form.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """

    SUBJECT_TO_OBJECT = {
        "jeg": "mig",
        "du": "dig",
        "han": "ham",
        "hun": "hende",
        "vi": "os",
        "i": "jer",
        "de": "dem",
    }
    OBJECT_TO_SUBJECT = {obj: sub for sub, obj in SUBJECT_TO_OBJECT.items()}

    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        tok_lower = token.text.lower()
        dep = token.dep_

        # Flip subject pronoun to object pronoun
        if tok_lower in SUBJECT_TO_OBJECT and dep in ("nsubj", "nsubj:pass", "conj"):
            if random.random() < flip_prob:
                flip_to = SUBJECT_TO_OBJECT[tok_lower]
                tokens_out[i] = flip_preserving_caps(token.text, flip_to)
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]
        # Flip object pronoun to subject pronoun
        elif tok_lower in OBJECT_TO_SUBJECT and dep in ("obj", "iobj", "obl"):
            if random.random() < flip_prob:
                flip_to = OBJECT_TO_SUBJECT[tok_lower]
                tokens_out[i] = flip_preserving_caps(token.text, flip_to)
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def flip_som_der(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Inject the error of using 'der' where 'som' is required.

    - If the token is 'som'
    - Check if it's used as an object or non-subject role
    - Flip it to 'der'

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        if token.text.lower() == "som" and random.random() < flip_prob:
            # Check dependency label for non-subject roles
            if token.dep_ in ("obj", "iobj", "obl"):
                tokens_out[i] = flip_preserving_caps(token.text, flip_to="der")
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence

def flip_han_hun_to_det(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Flip 'han'/'hun' to 'det' if it's a standard personal pronoun usage.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)

    has_antecedent, res_dict = has_antecedent_before(doc)
    pronoun_index = res_dict.get("pronoun_position", None)

    if has_antecedent and pronoun_index is not None:
        pronoun_index = int(res_dict["pronoun_position"])
        if random.random() < flip_prob:
            pronoun_token = tokens_out[pronoun_index]
            tokens_out[pronoun_index] = flip_preserving_caps(pronoun_token.text, flip_to="det")

        original_token = pronoun_token.text
        corrupted_token = tokens_out[pronoun_index]

        # Rebuild the sentence
        corrupted_words = [
            token if isinstance(token, str) else token.text for token in tokens_out
        ]

        final_sentence = join_tokens(corrupted_words)
        if token_comparison:
            return True, final_sentence, original_token, corrupted_token
        else:
            return True, final_sentence
    else:
        if token_comparison:
            return False, "", None, None
        else:
            return False, ""


def corrupt_spelling(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Detect words which are commonly misspelled and flip them to its common misspelled version.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    correct_words = [
        "sgisme", "vegne", "møg", "absolut", "anerkendelse", "diskussion", "eksplicit",
        "ekstra", "prakke", "investere", "alligevel", "aggressiv", "smøre", "familie",
        "interessant", "uddannelse", "terrasse", "stanniol", "professionel", "nærig",
        "kæreste", "rimelig", "nederen", "ligesom", "penicillin", "grineren", "klæder",
        "kraftedeme", "overhovedet", "anderledes", "hoved", "omgås", "egentligt", "undgås",
        "anbefale", "forhåbentlig", "øjnene", "overhoved", "større", "hundredvis", "dyrlægen",
        "jordemoder", "ægløsning", "retssag", "henholdsvis", "rimelig"
    ]

    mispelled_words = [
        "skisme", "vejne", "møj", "abselut", "anderkendelse", "diskution", "explicit",
        "extra", "pragge", "invistere", "aligevel", "agressiv", "smørre", "famillie",
        "interesant", "uddanelse", "terasse", "staniol", "proffesionel", "nærrig", "kærste",
        "rimlig", "nedern", "ligsom", "pencilin", "grinern", "klær", "kraftedme", "overhovdet",
        "anerledes", "hovede", "omgåes", "engenligt", "undgåes", "anbefalde", "forhåbenlig",
        "øjenene", "overhovede", "størrere", "hundredevis", "dyrelægen", "jordmoder",
        "æggeløsning", "retsag", "henholdvis", "rimlig"
    ]
    spelling_dict = dict(zip(correct_words, mispelled_words))

    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        lower_t = token.text.lower()

        # Check if the token is in the dictionary and if we should flip it
        if lower_t in spelling_dict and random.random() < flip_prob:
            # Get the misspelled version
            misspelled_word = spelling_dict[lower_t]
            # Replace in tokens_out
            tokens_out[i] = flip_preserving_caps(token.text, flip_to=misspelled_word)
            single_corruption_done = True
            original_token = token.text
            corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence

def flip_ligge_laegge(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Corrupt different 'ligge' forms with 'lægge' forms and vice versa.

    General rule: flip a form A of 'ligge' to its corresponding form B of 'lægge' and vice versa.

    Fixed word mapping:
      - 'ligger' <-> 'lægger'
      - 'lå' <-> 'lagde'
      - 'ligge' <-> 'lægge'

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    ligge_laegge_map = {
        "ligger": "lægger",
        "lægger": "ligger",
        "lå": "lagde",
        "lagde": "lå",
        "ligge": "lægge",
        "lægge": "ligge",
    }

    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break
        lower_t = token.text.lower()

        if lower_t in ligge_laegge_map and random.random() < flip_prob:
            tokens_out[i] = flip_preserving_caps(token.text, flip_to=ligge_laegge_map[lower_t])
            single_corruption_done = True
            original_token = token.text
            corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence

def corrupt_verb_r(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Introduces classic Danish 'r-problems' in verbs.

    Corrupt verbs ending in -rer (typical for present tense verbs)
        to their respective ending in -re (typical for infinitive verbs)

    - If it's a present verb and ends with '-rer', flip to '-re'.
    - If it's a infinitive verb and ends with '-re', flip to '-rer'.

    Some of the most common exceptions are excluded.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    exclude_verbs = ["gøre", "være", "have", "kunne", "skulle", "ville"]
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break
        word = token.text
        lower_word = word.lower()

        if token.pos_ == "VERB" and token.lemma_.lower() not in exclude_verbs and random.random() < flip_prob:
            morph = token.morph

            # Present tense verb ending with '-rer' -> flip to '-re'
            if "Tense=Pres" in morph and lower_word.endswith("rer") and len(word) >= 4:
                stem = word[:-3]
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "re"
                else:
                    tokens_out[i] = stem + "re"
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

            # Infinitive verb ending with '-re' -> flip to '-rer'
            elif "VerbForm=Inf" in morph and lower_word.endswith("re") and len(word) >= 3:
                stem = word[:-2]
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "rer"
                else:
                    tokens_out[i] = stem + "rer"
                single_corruption_done = True
                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def corrupt_noun_r(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Introduces classic Danish 'r-problems' in nouns.

    Applied rules:
        - Plural definite (-erne) -> missing 'r' (-ene)
        - Plural indefinite (-ere) -> drop 'e' (-er)
        - Singular definite (-eren) -> pluralize incorrectly (-ern)

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        word = token.text
        lower_word = word.lower()

        if token.pos_ == "NOUN" and random.random() < flip_prob:
            morph = token.morph

            # Target nouns ending with '-erne' (plural definite)
            if "Number=Plur" in morph and "Definite=Def" in morph and lower_word.endswith("erne"):
                stem = word[:-4]  # remove "erne"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "ene"
                else:
                    tokens_out[i] = stem + "ene"
                single_corruption_done = True

            # Target nouns ending with '-ere' (plural indefinite)
            elif "Number=Plur" in morph and "Definite=Ind" in morph and lower_word.endswith("ere"):
                stem = word[:-3]  # remove "ere"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "er"
                else:
                    tokens_out[i] = stem + "er"
                single_corruption_done = True

            # Target nouns ending with '-eren' (singular definite)
            elif "Number=Sing" in morph and "Definite=Def" in morph and lower_word.endswith("eren"):
                stem = word[:-4]  # remove "eren"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "ern"
                else:
                    tokens_out[i] = stem + "ern"
                single_corruption_done = True

            if single_corruption_done:
                if token.text.isupper() and isinstance(tokens_out[i], str):
                    tokens_out[i] = tokens_out[i].upper()

                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)

    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def corrupt_adjective_r(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Introduce classic Danish 'r-problems' in adjectives.

    Corrupt adjectives ending in -ere with -er and vice versa.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        word = token.text
        lower_word = word.lower()

        # Check if token is an adjective
        if token.pos_ == "ADJ" and random.random() < flip_prob:
            # If adjective ends with "ere" (comparative form)
            if lower_word.endswith("ere") and len(word) >= 4:
                # Swap to "er" (positive form)
                stem = word[:-3]  # remove "ere"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "er"
                else:
                    tokens_out[i] = stem + "er"
                single_corruption_done = True

            # If adjective ends with "er" (positive form)
            elif lower_word.endswith("er"):
                # Swap to "ere" (comparative form)
                stem = word[:-2]  # remove "er"
                if word[0].isupper():
                    tokens_out[i] = stem.capitalize() + "ere"
                else:
                    tokens_out[i] = stem + "ere"
                single_corruption_done = True

            if single_corruption_done:
                original_token = token.text
                corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def corrupt_genitive(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Corrupt genitives passing from a type to another.

    - Check if the token is a genitive noun.
    - Switch from the current type to another randomly
        - Ensure that the error is not too odd and unlikely to happen.

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        word = token.text

        genitive, type = is_genitive(token)

        if genitive and random.random() < flip_prob:
            # Remove genitive according to type
            genitive_len = len(type.value)
            stem = word[:-genitive_len]

            # Randomly choose a new type except the current one
            GenetiveTypeList = list(GenitiveTypeEnum)

            # If stem ends already in -s exclude type 1 too
            if stem.endswith("s") or stem.endswith("S"):
                GenetiveTypeList.remove(GenitiveTypeEnum.TYPE_1)

            new_type = random.choice([t for t in GenetiveTypeList if t != type])

            # Add the new type to the stem preserving case of whole word
            tokens_out[i] = stem + new_type.value

            single_corruption_done = True
            original_token = token.text
            corrupted_token = tokens_out[i]


    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence


def flip_far_for(dk_model: Language, sentence: str, flip_prob: float = 1.0, token_comparison: bool = False) -> (bool, str):
    """
    Introduce "får"/"for" confusion errors in a Danish sentence.

    Flip "får" to "for" and vice versa

    :param dk_model: A Danish spaCy model.
    :param sentence: A Danish sentence (string).
    :param flip_prob: Probability of corrupting each eligible sentence.
    :param token_comparison: If True, returns also the original and corrupted tokens.
    :return: A tuple with a boolean indicating if a corruption was done and the corrupted (or not) sentence.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False
    original_token = None
    corrupted_token = None

    for i, token in enumerate(doc):
        if single_corruption_done:  # only one corruption per call
            break

        tok_lower = token.text.lower()

        # Flip "får" to "for"
        if tok_lower == "får":
            if random.random() < flip_prob:
                tokens_out[i] = flip_preserving_caps(token.text, "for")
                single_corruption_done = True
        # Flip "for" to "får"
        elif tok_lower == "for":
            if random.random() < flip_prob:
                tokens_out[i] = flip_preserving_caps(token.text, "får")
                single_corruption_done = True

        if single_corruption_done:
            original_token = token.text
            corrupted_token = tokens_out[i]

    # Rebuild the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]

    final_sentence = join_tokens(corrupted_words)
    if token_comparison:
        return single_corruption_done, final_sentence, original_token, corrupted_token
    else:
        return single_corruption_done, final_sentence



# Basic corruptions from original ScaLA

def delete(tokens: List[str], pos_tags: List[str]) -> Union[str, None]:
    """Delete a random token from a list of tokens.

    The POS tags are used to prevent deletion of a token which does not make the
    resulting sentence grammatically incorrect, such as removing an adjective or an
    adverb.

    Args:
        tokens:
            The list of tokens to delete from.
        pos_tags:
            The list of POS tags for the tokens.

    Returns:
        The deleted token, or None if no token could be deleted.
    """
    # Copy the token list
    new_tokens = tokens.copy()

    # Get candidate indices to remove. We do not remove adjectives, adverbs,
    # punctuation, determiners or numbers, as the resulting sentence will probably
    # still be grammatically correct. Further, we do not remove nouns or proper nouns
    # if they have another noun or proper noun as neighbour, as that usually does not
    # make the sentence incorrect either.
    indices = [
        idx
        for idx, pos_tag in enumerate(pos_tags)
        if pos_tag not in ["ADJ", "ADV", "PUNCT", "SYM", "DET", "NUM"]
        and (
            pos_tag not in ["NOUN", "PROPN"]
            or (
                (idx == 0 or pos_tags[idx - 1] not in ["NOUN", "PROPN"])
                and (
                    idx == len(new_tokens) - 1
                    or pos_tags[idx + 1] not in ["NOUN", "PROPN"]
                )
            )
        )
    ]

    # If there are no candidates then return None
    if len(indices) == 0:
        return None

    # Get the random index
    rnd_idx = random.choice(indices)

    # Delete the token at the index
    new_tokens.pop(rnd_idx)

    # Join up the new tokens and return the string
    return join_tokens(new_tokens)


def flip_neighbours(tokens: List[str], pos_tags: List[str]) -> Union[str, None]:
    """Flip a pair of neighbouring tokens.

    The POS tags are used to prevent flipping of tokens which does not make the
    resulting sentence grammatically incorrect, such as flipping two adjectives.

    Args:
        tokens:
            The list of tokens to flip.
        pos_tags:
            The list of POS tags for the tokens.

    Returns:
        The flipped string, or None if no flip was possible.
    """
    # Copy the token list
    new_tokens = tokens.copy()

    # Collect all indices that are proper words, and which has a neighbour which is
    # also a proper word as well as having a different POS tag
    indices = [
        idx for idx, pos_tag in enumerate(pos_tags) if pos_tag not in ["PUNCT", "SYM"]
    ]
    indices = [
        idx
        for idx in indices
        if (idx + 1 in indices and pos_tags[idx] != pos_tags[idx + 1])
        or (idx - 1 in indices and pos_tags[idx] != pos_tags[idx - 1])
    ]

    # If there are fewer than two relevant tokens then return None
    if len(indices) < 2:
        return None

    # Get the first random index
    rnd_fst_idx = random.choice(indices)

    # Get the second (neighbouring) index
    if rnd_fst_idx == 0:
        rnd_snd_idx = rnd_fst_idx + 1
    elif rnd_fst_idx == len(tokens) - 1:
        rnd_snd_idx = rnd_fst_idx - 1
    elif (
        pos_tags[rnd_fst_idx + 1] in ["PUNCT", "SYM"]
        or pos_tags[rnd_fst_idx] == pos_tags[rnd_fst_idx + 1]
        or {pos_tags[rnd_fst_idx], pos_tags[rnd_fst_idx + 1]} == {"PRON", "AUX"}
    ):
        rnd_snd_idx = rnd_fst_idx - 1
    elif (
        pos_tags[rnd_fst_idx - 1] in ["PUNCT", "SYM"]
        or pos_tags[rnd_fst_idx] == pos_tags[rnd_fst_idx - 1]
        or {pos_tags[rnd_fst_idx], pos_tags[rnd_fst_idx + 1]} == {"PRON", "AUX"}
    ):
        rnd_snd_idx = rnd_fst_idx + 1
    elif random.random() > 0.5:
        rnd_snd_idx = rnd_fst_idx - 1
    else:
        rnd_snd_idx = rnd_fst_idx + 1

    # Flip the two indices
    new_tokens[rnd_fst_idx] = tokens[rnd_snd_idx]
    new_tokens[rnd_snd_idx] = tokens[rnd_fst_idx]

    # If we flipped the first character, then ensure that the new first character is
    # title-cased and the second character is of lower case. We only do this if they
    # are not upper cased, however.
    if rnd_fst_idx == 0 or rnd_snd_idx == 0:
        if new_tokens[0] != new_tokens[0].upper():
            new_tokens[0] = new_tokens[0].title()
        if new_tokens[1] != new_tokens[1].upper():
            new_tokens[1] = new_tokens[1].lower()

    # Join up the new tokens and return the string
    return join_tokens(new_tokens)


def corrupt_basic(
    tokens: List[str], pos_tags: List[str], num_corruptions: int = 1
) -> List[Tuple[str, str]]:
    """Corrupt a list of tokens.

    This randomly either flips two neighbouring tokens or deletes a random token.

    Args:
        tokens:
            The list of tokens to corrupt.
        pos_tags:
            The list of POS tags for the tokens.
        num_corruptions:
            The number of corruptions to perform. Defaults to 1.

    Returns:
        The list of (corrupted_string, corruption_type)
    """
    # Define the list of corruptions
    corruptions: List[Tuple[str, str]] = list()

    # Continue until we have achieved the desired number of corruptions
    while len(corruptions) < num_corruptions:
        # Choose which corruption to perform, at random
        corruption_fn = random.choice([flip_neighbours, delete])

        # Corrupt the tokens
        corruption = corruption_fn(tokens, pos_tags)

        # If the corruption succeeded, and that we haven't already performed the same
        # corruption, then add the corruption to the list of corruptions
        if corruption not in corruptions and corruption is not None:
            corruptions.append((corruption, corruption_fn.__name__))

    # Return the list of corruptions
    return corruptions


# Excluded corruptions

# Excluded (not used) as only 3 sentences in UD can be corrupted for now
def flip_noun_subj_pron_order(dk_model: Language, sentence: str, flip_prob: float = 1.0) -> (bool, str):
    """
    Force noun/subject-pronoun order errors in a Danish sentence.

    Rules for forcing errors:
      • If a noun (common noun or proper noun) and a subject pronoun (jeg, du, …)
        are in a "NOUN og SUBJ_PRONOUN" construction, flip the order to "SUBJ_PRONOUN og NOUN".
      • At most ONE flip per call; governed by ``flip_prob``.

    :param dk_model: A Danish spaCy model.
    :param sentence: The input sentence.
    :param flip_prob: Probability of flipping the order.
    :return: ``(single_corruption_done, corrupted_sentence)``.
    """
    # ── Token look-up for valid noun-pronoun pairs ──────────────────────────────────
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False

    # We look for a NOUN followed by the conjunction "og" and then a subject pronoun.
    for i, token in enumerate(doc):
        if single_corruption_done:  # only one corruption per call
            break

        # Check if current token is a noun and the next token is "og" and the token after that is a subject pronoun
        if token.pos_ in ("NOUN", "PROPN") and i + 2 < len(doc):
            next_token = doc[i + 1]
            following_token = doc[i + 2]

            if next_token.text.lower() == "og" and following_token.pos_ == "PRON" and following_token.dep_ in ("nsubj", "nsubj:pass"):
                # There's a "NOUN og SUBJ_PRONOUN" construction, so flip the order
                if random.random() < flip_prob:
                    # Swap the noun and the subject pronoun
                    tokens_out[i] = following_token.text
                    tokens_out[i + 2] = token.text
                    single_corruption_done = True

    # Re-assemble the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]
    return single_corruption_done, join_tokens(corrupted_words)


# Excluded (not used) as only 12 sentences in UD can be corrupted for now
def flip_af_ad_hardcode(dk_model: Language, sentence: str, flip_prob: float = 1.0) -> (bool, str):
    """
    Corrupt Danish sentences by swapping "af" and "ad" particles.

    This corruption targets the common Danish error where "af" and "ad" particles are confused.
    The particles are homophones (pronounced the same) but have different grammatical functions:
    - "ad" indicates movement along paths or through openings, or used in metaphorical contexts
    - "af" indicates movement away from something, or expresses causation/material/method

    :param dk_model: A Danish spaCy model.
    :param sentence: The input sentence.
    :param flip_prob: Probability of flipping each eligible particle.
    :return: ``(single_corruption_done, corrupted_sentence)``.
    """
    # ── Particle look-up tables ────────────────────────────────────────────────────
    # Words that should use "af" (will be flipped to incorrect "ad")
    AF_CORRECT_WORDS = [
        "blegne", "bygge", "dø", "gløde", "hensyn",
        "ked", "lugte", "læsse", "skumme", "træt"
    ]

    # Words that should use "ad" (will be flipped to incorrect "af")
    AD_CORRECT_WORDS = [
        "gang", "helvede", "hen", "ind", "kimse",
        "læne", "omveje", "pege", "skille"
    ]

    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False

    for i, token in enumerate(doc):
        if single_corruption_done:
            break

        # Check if the current token is "af" or "ad"
        if token.text.lower() in ["af", "ad"]:
            current_particle = token.text.lower()
            context_window = 3  # Look 3 tokens before and after

            # Check surrounding context for relevant lemmas
            for j in range(max(0, i - context_window), min(len(doc), i + context_window + 1)):
                if j == i or single_corruption_done:  # Skip the particle itself
                    continue

                context_lemma = doc[j].lemma_.lower()

                # Check if "af" is correct but we currently have "af" (can flip to "ad")
                if context_lemma in AF_CORRECT_WORDS and current_particle == "af":
                    # Check if they are grammatically related or close in the sentence
                    if token.head == doc[j] or doc[j].head == token or abs(i - j) <= 2:
                        if random.random() < flip_prob:
                            tokens_out[i] = flip_preserving_caps(token.text, "ad")
                            single_corruption_done = True
                            break
                # Check if "ad" is correct but we currently have "ad" (can flip to "af")
                elif context_lemma in AD_CORRECT_WORDS and current_particle == "ad":
                    # Check if they are grammatically related or close in the sentence
                    if token.head == doc[j] or doc[j].head == token or abs(i - j) <= 2:
                        if random.random() < flip_prob:
                            tokens_out[i] = flip_preserving_caps(token.text, "af")
                            single_corruption_done = True
                            break

    # Re-assemble the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]
    return single_corruption_done, join_tokens(corrupted_words)


# Excluded (not used) as only 18 sentences in UD can be corrupted for now
def flip_ad_af_simple(dk_model: Language, sentence: str, flip_prob: float = 1.0) -> (bool, str):
    """
    Corrupt Danish sentences by replacing the first "ad" with "af".

    This is a simplified version that only replaces the first "ad" that is an adposition (ADP)
    with "af", without considering specific grammatical rules.

    :param dk_model: A Danish spaCy model.
    :param sentence: The input sentence.
    :param flip_prob: Probability of flipping the particle.
    :return: ``(single_corruption_done, corrupted_sentence)``.
    """
    doc = dk_model(sentence)
    tokens_out = list(doc)
    single_corruption_done = False

    for i, token in enumerate(doc):
        if single_corruption_done:  # only one corruption per call
            break

        # Check if the current token is "ad" and is an adposition
        if token.text.lower() == "ad" and token.pos_ == "ADP":
            if random.random() < flip_prob:
                tokens_out[i] = flip_preserving_caps(token.text, "af")
                single_corruption_done = True
                break

    # Re-assemble the sentence
    corrupted_words = [
        token if isinstance(token, str) else token.text for token in tokens_out
    ]
    return single_corruption_done, join_tokens(corrupted_words)