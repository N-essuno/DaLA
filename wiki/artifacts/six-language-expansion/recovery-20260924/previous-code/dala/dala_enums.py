"""Compatibility enum populated from the Danish input pack."""
from enum import Enum
from .profiles import load_profile

_rule = next(r for r in load_profile('da')['rules'] if r['operator'] == 'genitive')
GenitiveTypeEnum = Enum('GenitiveTypeEnum', {f'TYPE_{i + 1}': suffix for i, suffix in enumerate(_rule['suffixes'])})
