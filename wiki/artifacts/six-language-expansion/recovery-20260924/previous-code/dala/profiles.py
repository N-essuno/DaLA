"""Language packs are validated JSON inputs, resolved relative to their file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_profile(path_or_language):
    path = Path(path_or_language)
    if not path.is_file():
        path = ROOT / 'config/languages' / f'{path_or_language}.json'
    profile = json.loads(path.read_text())
    for key in ('schema_version', 'language', 'mode', 'parser', 'sources', 'selection'):
        if key not in profile:
            raise ValueError(f'Missing language-pack field: {key}')
    if profile['schema_version'] != 1 or profile['mode'] not in {'legacy', 'pairs'}:
        raise ValueError('Unsupported language-pack schema or mode')
    profile['_path'] = str(path.resolve())
    return profile


def resource(profile, key):
    return (Path(profile['_path']).parent / profile[key]).resolve()
