"""Legacy POS-constrained token operations (language independent)."""
import random
from typing import List, Tuple, Union
from .text import join_tokens

def delete(tokens: List[str], pos_tags: List[str], rng=random) -> Union[str, None]:
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
    rnd_idx = rng.choice(indices)

    # Delete the token at the index
    new_tokens.pop(rnd_idx)

    # Join up the new tokens and return the string
    return join_tokens(new_tokens)

def flip_neighbours(tokens: List[str], pos_tags: List[str], rng=random) -> Union[str, None]:
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
    rnd_fst_idx = rng.choice(indices)

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
    elif rng.random() > 0.5:
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
    tokens: List[str], pos_tags: List[str], num_corruptions: int = 1, rng=random, operators=None
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
        available = {'flip_neighbours': flip_neighbours, 'delete': delete}
        corruption_fn = rng.choice([available[name] for name in (operators or ['flip_neighbours', 'delete'])])

        # Corrupt the tokens
        corruption = corruption_fn(tokens, pos_tags, rng=rng)

        # If the corruption succeeded, and that we haven't already performed the same
        # corruption, then add the corruption to the list of corruptions
        if corruption not in corruptions and corruption is not None:
            corruptions.append((corruption, corruption_fn.__name__))

    # Return the list of corruptions
    return corruptions
