"""Language-independent edit representation and exact application."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Corruption:
    rule_id: str
    corruption_type: str
    start: int
    end: int
    original: str
    replacement: str
    protected_tokens: tuple

    def record(self):
        return asdict(self)

def apply_edits(text, edits):
    """Offsets always refer to the original sentence, even for multiple edits."""
    end = -1
    for e in sorted(edits, key=lambda e: e.start):
        if e.start < end or e.start < 0 or e.end <= e.start or text[e.start:e.end] != e.original:
            raise ValueError('Overlapping edit or source-span mismatch')
        end = e.end
    for e in sorted(edits, key=lambda e: e.start, reverse=True):
        text = text[:e.start] + e.replacement + text[e.end:]
    return text
