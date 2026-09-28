"""Compatibility helpers; language-specific values are supplied by the pack."""
from .profiles import load_profile
from .text import join_tokens, flip_preserving_caps
from .dala_enums import GenitiveTypeEnum


def _rule(operator):
    return next(r for r in load_profile('da')['rules'] if r['operator'] == operator)


def is_negative(doc):
    rule = _rule('next_nominal')
    return any(rule['negation_feature'] in t.morph or t.lower_ in rule['negations'] for t in doc)


def is_question(doc):
    return any(t.text == _rule('next_nominal')['question'] for t in doc)


def is_genitive(token):
    rule = _rule('genitive')
    order = rule['detect_order'] if rule['morph_feature'] in token.morph else rule['fallback_detect_order'] if token.pos_ in rule['pos'] else []
    kind = next((i for i in order if token.text.endswith(rule['suffixes'][i])), None)
    return (False, None) if kind is None else (True, list(GenitiveTypeEnum)[kind])


def has_antecedent_before(doc):
    rule = _rule('antecedent')
    for token in doc:
        if token.lower_ in rule['pronouns']:
            entities = [ent for ent in doc.ents if ent.end <= token.i]
            if entities:
                entities.sort(key=lambda ent: token.i - ent.start)
                return True, dict(pronoun=token.text, pronoun_position=token.i,
                                  potential_referents=[ent.text for ent in entities])
    return False, {'message': 'No valid pronoun found in the sentence'}
