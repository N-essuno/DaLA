"""Local dataset creation with language-specific corruption policies."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from .english import RULES, candidates, choose_edit
from .languages import LANGUAGES, get_language
from .ud_annotations import load_annotated_ud


AUDIT_COLUMNS = ["pair_id", "sent_id", "original_text", "corrupted_text",
                 "corruption_type", "token_id", "start", "end", "original", "replacement",
                 "original_acceptable", "corrupted_unacceptable", "single_error", "review_notes"]


def prepare_english(df, seed=4242):
    """Keep both members of eligible pairs; record all candidate-type coverage."""
    samples, audit = [], []
    eligible, selected = Counter(), Counter()
    for row in df.to_dict("records"):
        edits = candidates(row)
        eligible.update({edit.corruption_type for edit in edits})
        edit = choose_edit(row, seed)
        if edit is None:
            continue
        pair_id = hashlib.sha256(row["doc"].encode()).hexdigest()[:20]
        corrupted = edit.apply(row["doc"])
        selected[edit.corruption_type] += 1
        samples.extend([
            dict(text=row["doc"], corruption_type=None, label="correct"),
            dict(text=corrupted, corruption_type=edit.corruption_type, label="incorrect"),
        ])
        audit.append(dict(pair_id=pair_id, sent_id=row["sent_id"],
                          original_text=row["doc"], corrupted_text=corrupted, **asdict(edit)))
    data = pd.DataFrame(samples, columns=["text", "corruption_type", "label"])
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    report = dict(input_sentences=len(df), pairs=len(audit), abstained=len(df) - len(audit),
                  eligible_by_type={rule: eligible[rule] for rule in RULES},
                  selected_by_type={rule: selected[rule] for rule in RULES})
    return data, pd.DataFrame(audit, columns=AUDIT_COLUMNS), report


def create_english(output_dir="la_output", seed=4242, data_dir=None):
    """Preserve UD splits, removing repeated sources and cross-split collisions."""
    config = get_language("en")
    source = load_annotated_ud("en", data_dir)
    outputs, audits, reports = {}, {}, {}
    seen = set()
    # Give held-out data priority when a source occurs in several splits.
    for split in ("test", "val", "train"):
        df = source[split]
        raw_count = len(df)
        df = df.drop_duplicates("doc")
        df = df[~df.doc.isin(seen)]
        seen.update(df.doc)
        df = df[df.tokens.map(len).gt(5) & df.doc.str.len().between(2, 5000)]
        outputs[split], audits[split], reports[split] = prepare_english(df, seed)
        reports[split]["raw_sentences"] = raw_count
        reports[split]["source_sha256"] = source[split].attrs.get("source_sha256")
    # A corruption can itself equal another source or corruption. Keep all text
    # split-disjoint, dropping an entire pair rather than upsetting label balance.
    seen_texts = set()
    for split in ("test", "val", "train"):
        audit = audits[split]
        keep = []
        for row in audit.to_dict("records"):
            texts = {row["original_text"], row["corrupted_text"]}
            keep.append(not bool(texts & seen_texts))
            if keep[-1]:
                seen_texts.update(texts)
        reports[split]["collision_pairs_removed"] = len(audit) - sum(keep)
        audit = audit.loc[keep].reset_index(drop=True)
        audits[split] = audit
        samples = []
        for row in audit.to_dict("records"):
            samples.extend([
                dict(text=row["original_text"], corruption_type=None, label="correct"),
                dict(text=row["corrupted_text"], corruption_type=row["corruption_type"], label="incorrect"),
            ])
        outputs[split] = pd.DataFrame(samples, columns=["text", "corruption_type", "label"]).sample(
            frac=1, random_state=seed).reset_index(drop=True)
        reports[split]["exported_pairs"] = len(audit)
        reports[split]["exported_by_type"] = {rule: int((audit.corruption_type == rule).sum()) for rule in RULES}
    if any(frame.empty for frame in outputs.values()):
        raise ValueError("No eligible English pairs in at least one split; inspect source annotations and rule coverage")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        outputs[split].to_csv(output / f"dala_en_{split}.csv", index=False)
        audits[split].to_csv(output / f"dala_en_{split}_audit.csv", index=False)
    # A manageable review queue, stratified by split and error type.
    review_samples = []
    for split, audit in audits.items():
        for _, group in audit.groupby("corruption_type"):
            review_samples.append(group.sample(n=min(30, len(group)), random_state=seed).assign(split=split))
    pd.concat(review_samples, ignore_index=True).to_csv(output / "dala_en_review.csv", index=False)
    manifest = dict(language="en", treebank=config.treebank, revision=None if data_dir else config.revision,
                    source="local" if data_dir else "UniversalDependencies",
                    seed=seed, validation_status="unvalidated_candidates", splits=reports)
    (output / "dala_en_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return outputs


def cli():
    import random
    from .pipeline import build
    from .profiles import load_profile
    parser = argparse.ArgumentParser(description='Build DaLA from a language input pack')
    parser.add_argument('--language', default='da', help='Built-in language-pack code')
    parser.add_argument('--profile', help='Custom language-pack JSON; overrides --language')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--source', choices=['common-pile','ewt'], help='EWT is a historical English-only pilot')
    parser.add_argument('--data-dir', help='Local CoNLL-U files for the EWT pilot')
    parser.add_argument('--max-errors', type=int, choices=[1,2,3])
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--split-proportions', action='store_true', help='Use configured proportions in compatibility mode')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--checker', choices=['local','off'])
    parser.add_argument('--push-to-hub', metavar='DATASET_ID')
    args=parser.parse_args()
    if args.source=='ewt':
        if args.language!='en' or args.profile:
            parser.error('The historical EWT pilot requires --language en without --profile')
        frames=create_english(args.output_dir or 'la_output/english_ewt',args.seed,args.data_dir)
        if args.push_to_hub:
            from datasets import Dataset,DatasetDict
            DatasetDict({k:Dataset.from_pandas(v,preserve_index=False) for k,v in frames.items()}).push_to_hub(args.push_to_hub,private=True)
        return
    if args.data_dir:parser.error('--data-dir is only supported by the historical EWT pilot')
    profile=load_profile(args.profile or args.language)
    random.seed(args.seed)
    if profile['mode']=='legacy':
        if args.source:parser.error('Choose the source in the compatibility language pack')
        build(profile,use_split_proportions=args.split_proportions,output_dir=args.output_dir or 'la_output',dataset_id=args.push_to_hub)
    else:
        output=Path(args.output_dir or f"la_output/{profile['language']}_common_pile")
        build(profile,output_dir=output,seed=args.seed,max_errors=args.max_errors or profile['selection']['max_errors'],
              max_documents=args.max_documents,offline=args.offline,checker_mode=args.checker or profile['checker']['mode'])
        if args.push_to_hub:
            from datasets import Dataset,DatasetDict
            from .pair_pipeline import SPLITS
            from .dataset_review import load_jsonl
            for view in ('acceptability','acceptability_it','correction_it'):
                datasets={split:Dataset.from_list(load_jsonl(output/split/f'{view}.jsonl')) for split in SPLITS}
                DatasetDict(datasets).push_to_hub(args.push_to_hub,config_name=view,private=True)


if __name__ == '__main__':
    cli()
