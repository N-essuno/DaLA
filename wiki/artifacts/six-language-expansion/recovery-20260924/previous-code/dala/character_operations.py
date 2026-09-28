"""Bounded character operations; language-specific parameters come from rule data.

Candidates are not certified errors. Production requires independent lexical
screening in every configured dialect before contextual screening.
"""


def replacements(word, rule):
    word = word.lower()
    if (not word.isalpha() or (not rule.get('unicode_letters', False) and not word.isascii())
            or not rule['min_length'] <= len(word) <= rule['max_length']):
        return ()
    result = set()
    consonants = set(rule.get('consonants', ''))
    for i in range(1, len(word) - 1):
        if rule['operation'] == 'delete_internal':
            result.add(word[:i] + word[i + 1:])
        elif rule['operation'] == 'substitute_character':
            for char in rule.get('substitutions', {}).get(word[i], []):
                result.add(word[:i] + char + word[i + 1:])
        elif rule['operation'] == 'transpose_internal' and i < len(word) - 2 and word[i] != word[i + 1]:
            result.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
        elif rule['operation'] == 'delete_doubled_consonant' and word[i] in consonants and word[i] == word[i - 1]:
            if word[i + 1:] in rule.get('protected_doubling_suffixes', []):
                continue
            result.add(word[:i] + word[i + 1:])
        elif rule['operation'] == 'duplicate_consonant' and word[i] in consonants and word[i] != word[i - 1] and word[i] != word[i + 1]:
            if word[i + 1:] in rule.get('protected_doubling_suffixes', []):
                continue
            result.add(word[:i] + word[i] + word[i:])
    return tuple(sorted(result))
