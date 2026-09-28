"""Check this project's OKF 0.2 structural conventions and internal links."""
from datetime import datetime
from pathlib import Path
import re

import yaml


def validate(root=Path('wiki')):
    count = 0
    for p in root.rglob('*.md'):
        # Archived upstream cards are evidence, not authored OKF concepts.
        if p.relative_to(root).parts[0] == 'artifacts':
            continue
        text = p.read_text(encoding='utf-8')
        if p.name not in {'index.md', 'log.md'}:
            if not text.startswith('---\n'):
                raise ValueError(f'{p}: missing frontmatter')
            meta = yaml.safe_load(text.split('---', 2)[1])
            if not isinstance(meta.get('type'), str) or not meta['type'].strip():
                raise ValueError(f'{p}: missing concept type')
            if meta.get('status', 'stable') not in {'stable', 'draft', 'deprecated'}:
                raise ValueError(f'{p}: invalid lifecycle status')
            generated = meta.get('generated')
            if generated:
                if not generated.get('by'):
                    raise ValueError(f'{p}: generated.by missing')
                timestamp = generated.get('at')
                parsed = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if parsed.tzinfo is None:
                    raise ValueError(f'{p}: timestamp needs UTC offset')
            sources = meta.get('sources', [])
            if any(not source.get('resource') for source in sources):
                raise ValueError(f'{p}: source resource missing')
            keys = {s.get('id') for s in sources}
            for footnote in re.findall(r'^\[\^([^\]]+)\]:', text, re.M):
                if footnote not in keys:
                    raise ValueError(f'{p}: footnote has no matching source ID: {footnote}')
            count += 1
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0]
            resolved = root / target.lstrip('/') if target.startswith('/') else p.parent / target
            if not resolved.exists():
                raise ValueError(f'{p}: broken bundle link {target}')
    return count


if __name__ == '__main__':
    print(f'Validated {validate()} OKF concepts and bundle links')
