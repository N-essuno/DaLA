"""Explicit language configuration; unsupported languages fail early."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    treebank: str
    prefix: str
    revision: str = "r2.17"


LANGUAGES = {
    "da": LanguageConfig("da", "UD_Danish-DDT", "da_ddt"),
    "en": LanguageConfig("en", "UD_English-EWT", "en_ewt"),
}


def get_language(code):
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"Unsupported language {code!r}; choose from {', '.join(LANGUAGES)}") from None
