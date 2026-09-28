"""One public build entry point; language packs select resources and policies."""
from .profiles import load_profile


def build(profile, **kwargs):
    profile = load_profile(profile) if isinstance(profile, str) else profile
    if profile['mode'] == 'legacy':
        from .legacy_pipeline import build_legacy
        return build_legacy(profile, **kwargs)
    from .pair_pipeline import build as build_pairs
    kwargs.setdefault('max_errors', profile['selection']['max_errors'])
    kwargs.setdefault('checker_mode', profile['checker']['mode'])
    return build_pairs(profile=profile, **kwargs)
