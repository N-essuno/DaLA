"""Compile language-supplied constraints and morphology; no language inventories."""
from collections import defaultdict


def compile_inflections(language, morphology, definitions, options=None):
    options = options or {}
    groups = defaultdict(list)
    for entry in morphology:
        groups[(entry['lemma'], entry['pos'])].append(entry)
    rules = []
    relevant = {'Gender','Number','Case','Definite','Degree','Animacy','Person','VerbForm','Tense','Mood','Voice','Polarity','PronType','Poss'}
    if options.get('preserve_all_features'):
        relevant |= {k for entry in morphology for k in entry['features']}
    for definition in definitions:
        rule = dict(definition, id=language+'_'+definition['family']+'_'+definition.get('suffix','ud'),
                    operator='dictionary_inflection', sources=['morphology','ud_syntax'], evidence_kind='grammar_constraint_with_attested_forms',
                    audit_status='pending', mappings=defaultdict(set), pairs=defaultdict(list))
        feature = rule['feature']
        for (lemma,pos), analyses in groups.items():
            if pos not in rule['pos'] or rule.get('lemmas') and lemma not in rule['lemmas']: continue
            original_forms = defaultdict(set)
            buckets = defaultdict(list)
            ignored = {feature}
            if feature in options.get('ignore_gender_animacy_features', ['Gender','Number','Definite']): ignored |= {'Gender','Animacy'}
            # Nordic plural adjectives do not distinguish definiteness. The
            # all-analysis syncretism veto still excludes valid replacements.
            if feature == 'Number' and options.get('number_ignores_definite'): ignored.add('Definite')
            applicability = rule.get('feature_applicability', {})
            ignored |= set(applicability)
            def compatible_endpoints(a, b):
                for key, condition in applicability.items():
                    left, right = a['features'].get(key), b['features'].get(key)
                    if left == right: continue
                    # Only an explicitly inapplicable missing feature may differ.
                    if left is not None and right is not None: return False
                    missing = a if left is None else b
                    if not all(missing['features'].get(k) in values for k, values in condition.items()): return False
                return True
            def signature(a):
                return () if feature == 'VerbForm' else tuple(sorted((k,v) for k,v in a['features'].items() if k in relevant and k not in ignored))
            for c in analyses:
                value = c['features'].get(feature)
                values = value.split(',') if value else [None]
                for alternative in values:
                    original_forms[alternative].update(c['forms'])
                buckets[signature(c)].append(c)
            for a in analyses:
                if not a.get('generation_eligible', True): continue
                before = {k:v for k,v in a['features'].items() if k in relevant}
                if any(before.get(k) in values for k, values in rule.get('excluded_features', {}).items()): continue
                value = before.get(feature)
                if not value or (',' in value and not rule.get('set_valued_feature')) or not set(value.split(',')) <= set(rule.get('from_values', value.split(','))): continue
                if any(before.get(k) not in values for k, values in rule.get('required_features', {}).items()): continue
                if rule['context'] == 'finite_agreement' and before.get('VerbForm') != 'Fin': continue
                for b in buckets[signature(a)]:
                    if not b.get('generation_eligible', True) or not compatible_endpoints(a, b): continue
                    after = {k:v for k,v in b['features'].items() if k in relevant}
                    other = after.get(feature)
                    if not other or (',' in other and not rule.get('set_valued_feature')) or set(other.split(',')) & set(value.split(',')) or not set(other.split(',')) <= set(rule.get('to_values', other.split(','))): continue
                    # Reject syncretism: replacement has an attested valid original-feature analysis.
                    # An underspecified analysis may also license the original
                    # value; absence of a feature is not negative evidence.
                    unknown_forms=original_forms[None]
                    if rule['context']=='finite_agreement':
                        # An infinitive with no person/number is not a licensed
                        # finite form merely because those features are absent.
                        unknown_forms={w for c in analyses if feature not in c['features']
                            and c['features'].get('VerbForm') in {None,'Fin'} for w in c['forms']}
                    valid_original = set().union(*(original_forms[v] for v in value.split(','))) | unknown_forms
                    def minimum_attestation(entry, form):
                        lexical = options.get('allow_lexical_evidence') and entry.get('evidence_kind') in options.get('lexical_evidence_kinds', ['unimorph_descriptive_paradigm'])
                        default = 3 if options.get('allow_unattested') or entry.get('normative') or lexical else 0
                        return entry.get('attestations', {}).get(form, default)
                    for source in a['forms']:
                        if minimum_attestation(a, source) < 3: continue
                        for replacement in b['forms']:
                            if minimum_attestation(b, replacement) < 3: continue
                            if replacement in valid_original or replacement == source: continue
                            rule['mappings'][source].add(replacement)
                            pair=dict(replacement=replacement,lemma=lemma,before=before,after=after)
                            if pair not in rule['pairs'][source]: rule['pairs'][source].append(pair)
        rule['mappings'] = {k:sorted(v) for k,v in sorted(rule['mappings'].items())}
        rule['pairs'] = dict(sorted(rule['pairs'].items()))
        # Retain empty rule definitions for honest coverage reporting.
        rule['active'] = bool(rule['mappings'])
        rules.append(rule)
    return rules
