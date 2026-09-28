"""English syntax guards around shared, span-preserving token operations."""
from ..edits import Corruption


def candidates(sent, rules):
    out = []
    for rule in rules:
        if rule.get('operator') != 'guarded_token' or not rule.get('enabled', True):
            continue
        for t in sent:
            if t.i + 1 >= sent.end or t.ent_type_ or t.pos_ == 'PROPN':
                continue
            following = t.doc[t.i + 1]
            if not t.is_alpha or not following.is_alpha or following.ent_type_ or following.pos_ == 'PROPN':
                continue
            # Restrict to simple whitespace and retain exact source offsets.
            space = t.whitespace_
            if not space or not space.isspace() or '\n' in space:
                continue
            allowed = False
            if rule['operation'] == 'delete_progressive_auxiliary':
                head = t.head
                subjects = [x for x in head.children if x.dep_ == 'nsubj']
                allowed = (t.lower_ in rule['auxiliaries'] and t.dep_ == 'aux'
                           and following == head and head.dep_ == 'ROOT' and head.tag_ == 'VBG'
                           and len(subjects) == 1 and subjects[0].i < t.i
                           and not any(x.dep_ in {'conj', 'cc', 'auxpass'} or
                                       x.dep_ == 'aux' and x != t for x in head.children)
                           and not any(x.tag_ in {'VBD', 'VBP', 'VBZ', 'MD'} and x != t for x in sent))
                replacement = following.text
            elif rule['operation'] == 'swap_article_noun':
                head = t.head
                allowed = (t.lower_ in rule['articles'] and t.dep_ == 'det' and following == head
                           and head.pos_ == 'NOUN' and head.dep_ in rule['noun_dependencies']
                           and not any(x.dep_ in {'compound', 'conj', 'cc', 'poss'} for x in head.children)
                           and t.i > sent.start and head.i + 1 < sent.end
                           and (head.doc[head.i + 1].is_punct or head.doc[head.i + 1].pos_ == 'ADP'))
                replacement = following.text + space + t.text
            else:
                raise ValueError('Unknown English fallback operation')
            if allowed:
                start, end = t.idx - sent.start_char, following.idx + len(following.text) - sent.start_char
                # Include the surviving neighbour in deletion spans so checker
                # evidence and inverse offsets have an explicit nonempty anchor.
                out.append(Corruption(rule['id'], rule['family'], start, end,
                                      sent.text[start:end], replacement, tuple(x.i for x in sent)))
    return out
