"""Compatibility names for Danish's declarative rule pack.

No Danish vocabulary, rule conditions or source choices live in this module.
"""
from .profiles import load_profile
from .rule_engine import ConfiguredRules, parser_model, run_rule
from .token_operations import corrupt_basic, delete, flip_neighbours


def SpacyModelSingleton(model_name=None):
    return parser_model(model_name or load_profile('da')['parser'])


def corrupt_dala(df):
    return ConfiguredRules(load_profile('da'), SpacyModelSingleton()).corrupt(df)


def _wrapper(rule):
    def apply(model, sentence, flip_prob=1.0, token_comparison=False):
        return run_rule(rule, model, sentence, flip_prob, token_comparison)
    apply.__name__ = rule['id']
    return apply


for _rule in load_profile('da')['rules']:
    if _rule['operator'] != 'token_fallback':
        globals()[_rule['id']] = _wrapper(_rule)


def get_corruption_functions():
    return [globals()[r['id']] for r in load_profile('da')['rules']]
