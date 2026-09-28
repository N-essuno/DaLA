"""Language-neutral operators for declarative compatibility rule packs.

Operator behavior includes selection and RNG timing; language-specific strings,
morphology, mappings and branch order belong exclusively to input JSON.
"""
import random
from functools import lru_cache

from .text import join_tokens, flip_preserving_caps
from .profiles import load_profile


def matches(token, condition):
    fields = {'pos': token.pos_, 'dep': token.dep_, 'lower': token.lower_, 'lemma': token.lemma_.lower(), 'tag': token.tag_}
    for field, value in condition.items():
        if field in fields and fields[field] not in value:
            return False
        if field == 'exclude_lemma' and fields['lemma'] in value:
            return False
        if field == 'suffix' and not token.lower_.endswith(value):
            return False
        if field == 'suffix_any' and not any(token.lower_.endswith(s) for s in value):
            return False
        if field == 'min_length' and len(token.text) < value:
            return False
        if field == 'morph' and not all(v in token.morph for v in value):
            return False
        if field == 'morph_exact' and not all(token.morph.get(k) == v for k, v in value.items()):
            return False
        if field not in {*fields, 'exclude_lemma', 'suffix', 'suffix_any', 'min_length', 'morph', 'morph_exact'}:
            raise ValueError(f'Unknown token condition: {field}')
    return True


def run_rule(rule, model, sentence, flip_prob=1.0, token_comparison=False, rng=random):
    doc = model(sentence)
    output = [t.text for t in doc]
    done, before, after = False, None, None
    op = rule['operator']
    for token in doc:
        if done:
            break
        word = token.text
        replacement = None
        if op == 'lookup':
            for branch in rule['branches']:
                if token.lower_ in branch['mapping'] and (rule.get('draw_before_condition') or matches(token, branch['when'])):
                    if rng.random() < flip_prob and matches(token, branch['when']):
                        replacement = flip_preserving_caps(word, branch['mapping'][token.lower_])
                    break
        elif op == 'suffix':
            if matches(token, rule['gate']) and rng.random() < flip_prob:
                for branch in rule['branches']:
                    if matches(token, branch['when']):
                        stem = word[:-branch['remove']]
                        suffix = branch['append']
                        if rule['case'] == 'preserve_stem':
                            replacement = stem + (suffix.upper() if word.isupper() else suffix)
                        else:
                            replacement = (stem.capitalize() if word[0].isupper() else stem) + suffix
                            if rule['case'] == 'capitalize_stem_then_upper' and word.isupper():
                                replacement = replacement.upper()
                        break
        elif op == 'child_lookup':
            if matches(token, rule['parent']):
                # Preserve the historical inner-loop behavior for multiple children.
                for child in token.children:
                    if matches(child, rule['child']) and child.lower_ in rule['mapping'] and rng.random() < flip_prob:
                        output[child.i] = flip_preserving_caps(child.text, rule['mapping'][child.lower_])
                        before = token.text if rule.get('legacy_report_parent') else child.text
                        after, done = output[child.i], True
        elif op == 'next_nominal':
            if token.lower_ in rule['mapping'] and rng.random() < flip_prob:
                following = next((t for t in doc[token.i + 1:] if t.pos_ in rule['pos']), None)
                negative = any(rule['negation_feature'] in t.morph or t.lower_ in rule['negations'] for t in doc)
                question = any(t.text == rule['question'] for t in doc)
                if following is not None and rule['number'][token.lower_] in following.morph.get('Number'):
                    if token.lower_ not in rule['positive_only'] or not (negative or question):
                        replacement = flip_preserving_caps(word, rule['mapping'][token.lower_])
        elif op == 'antecedent':
            if token.lower_ in rule['pronouns'] and any(ent.end <= token.i for ent in doc.ents):
                if rng.random() < flip_prob:
                    replacement = flip_preserving_caps(word, rule['replacement'])
                else:
                    # Historical function has undefined local variables on this path.
                    raise UnboundLocalError('Legacy antecedent corruption requires successful probability draw')
        elif op == 'genitive':
            order = rule['detect_order'] if rule['morph_feature'] in token.morph else rule['fallback_detect_order'] if token.pos_ in rule['pos'] else []
            kind = next((i for i in order if word.endswith(rule['suffixes'][i])), None)
            if kind is not None and rng.random() < flip_prob:
                stem = word[:-len(rule['suffixes'][kind])]
                choices = list(range(len(rule['suffixes'])))
                if any(stem.endswith(s) for s in rule['exclude_if_stem_ends']):
                    choices.remove(rule['exclude_suffix'])
                new = rng.choice([i for i in choices if i != kind])
                replacement = stem + rule['suffixes'][new]
        else:
            raise ValueError(f'Unknown rule operator: {op}')
        if replacement is not None:
            output[token.i] = replacement
            before, after, done = word, replacement, True
    result = join_tokens(output)
    if op == 'antecedent' and not done:
        result = ''
    return (done, result, before, after) if token_comparison else (done, result)


@lru_cache(maxsize=8)
def parser_model(name):
    import spacy
    return spacy.load(name)


class ConfiguredRules:
    def __init__(self, profile, model=None):
        self.profile = load_profile(profile) if isinstance(profile, (str, bytes)) else profile
        if self.profile['selection']['strategy'] != 'first_success' or self.profile['selection']['max_errors'] != 1:
            raise ValueError('Compatibility mode requires first_success and one error')
        self.model = model or parser_model(self.profile['parser'])
        self.rules = self.profile['rules']

    def corrupt(self, frame, rng=random):
        from .token_operations import corrupt_basic
        results = []
        for row in frame.to_dict('records'):
            if row['doc'] is None:
                continue
            for rule in self.rules:
                if rule['operator'] == 'token_fallback':
                    results.append(corrupt_basic(row['tokens'], row['pos_tags'], 1, rng=rng, operators=rule['operators'])[0])
                    break
                done, result = run_rule(rule, self.model, row['doc'], self.profile['selection']['flip_probability'], rng=rng)
                if done:
                    results.append((result, rule['id']))
                    break
            else:
                raise ValueError('No applicable rule; compatibility output requires one result per input')
        return results
