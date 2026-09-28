"""Read gold CoNLL-U annotations and align surface tokens to original text."""
import hashlib

import pandas as pd
import requests

from .languages import get_language


def attributes(value):
    return dict(item.split("=", 1) for item in value.split("|") if "=" in item)


def parse_conllu(text):
    records = []
    text = text.replace("\r\n", "\n")
    for block in text.strip().split("\n\n"):
        words, surface, doc, sent_id = [], [], "", ""
        covered = set()
        for line in block.splitlines():
            if line.startswith("# text = "):
                doc = line[9:]
            elif line.startswith("# sent_id = "):
                sent_id = line[12:]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError(f"Invalid CoNLL-U row: {line!r}")
                idx, form, lemma, upos, xpos, feats, head, deprel, _, misc = fields
                if "." in idx:
                    continue  # Empty nodes are not surface words.
                if "-" in idx:
                    first, last = map(int, idx.split("-"))
                    covered.update(range(first, last + 1))
                    surface.append((None, form))
                    continue
                word = dict(id=idx, form=form, lemma=lemma, upos=upos, xpos=xpos,
                            feats=attributes(feats), head=head, deprel=deprel,
                            misc=attributes(misc), span=None)
                words.append(word)
                if int(idx) not in covered:
                    surface.append((word, form))
        if not words:
            continue
        cursor, aligned = 0, bool(doc)
        for word, form in surface:
            while cursor < len(doc) and doc[cursor].isspace():
                cursor += 1
            if doc[cursor:cursor + len(form)] != form:
                aligned = False
                break
            if word is not None:
                word["span"] = (cursor, cursor + len(form))
            cursor += len(form)
        aligned = aligned and not doc[cursor:].strip()
        records.append(dict(doc=doc, sent_id=sent_id, words=words, aligned=aligned,
                            ids=[w["id"] for w in words], tokens=[w["form"] for w in words],
                            pos_tags=[w["upos"] for w in words]))
    return pd.DataFrame(records, columns=["doc", "sent_id", "words", "aligned", "ids", "tokens", "pos_tags"])


def load_annotated_ud(language="en", data_dir=None):
    """Load a pinned UD release, or local files named like en_ewt-ud-dev.conllu."""
    from pathlib import Path
    config = get_language(language)
    splits = {}
    for split, suffix in (("train", "train"), ("val", "dev"), ("test", "test")):
        filename = f"{config.prefix}-ud-{suffix}.conllu"
        if data_dir is not None:
            content = (Path(data_dir) / filename).read_text(encoding="utf-8")
        else:
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{config.treebank}/{config.revision}/{filename}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content.decode("utf-8")
        splits[split] = parse_conllu(content)
        splits[split].attrs["source_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return splits
