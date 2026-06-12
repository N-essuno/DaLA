#!/usr/bin/env python3
"""Print summary statistics for local DaLA TV2R instruction-tuning datasets."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "data" / "dalas_tv2r_it"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print statistics for datasets under it_version/data/dalas_tv2r_it."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing the converted dataset folders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of most common corruption types to print.",
    )
    return parser.parse_args()


def load_split(path: Path) -> tuple[str, list[dict]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("direction", ""), payload.get("samples", [])


def discover_datasets(base_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir() and any((path / f"{split}.json").exists() for split in SPLITS)
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"


def format_counter(counter: Counter, top_k: int) -> str:
    if not counter:
        return "-"

    parts = []
    total = counter.total()
    for key, count in counter.most_common(top_k):
        label = "None" if key is None else str(key)
        percentage = count / total * 100 if total else 0
        parts.append(f"{label}: {count:,} ({percentage:.1f}%)")
    return ", ".join(parts)


def format_responses(counter: Counter, top_k: int) -> str:
    if not counter:
        return "-"

    max_response_len = max(len(str(response)) for response in counter)
    if len(counter) <= top_k and max_response_len <= 40:
        return format_counter(counter, top_k)

    repeated_values = sum(1 for count in counter.values() if count > 1)
    repeated_rows = sum(count for count in counter.values() if count > 1)
    return (
        f"{len(counter):,} unique free-text responses; "
        f"{repeated_values:,} repeated responses covering {repeated_rows:,} rows"
    )


def length_stats(values: list[int]) -> str:
    if not values:
        return "min=0 mean=0.00 p50=0.00 p95=0.00 max=0"

    return (
        f"min={min(values):,} "
        f"mean={mean(values):,.2f} "
        f"p50={percentile(values, 0.50):,.2f} "
        f"p95={percentile(values, 0.95):,.2f} "
        f"max={max(values):,}"
    )


def count_words(text: str) -> int:
    return len(text.split())


def summarize_samples(samples: Iterable[dict], top_k: int) -> dict:
    samples = list(samples)
    contents = [sample.get("content") or "" for sample in samples]
    responses = [sample.get("response") or "" for sample in samples]
    corruption_types = Counter(sample.get("corruption_type") for sample in samples)
    response_labels = Counter(responses)
    exact_copies = sum(content == response for content, response in zip(contents, responses))

    affected_token_pairs = Counter(
        (
            sample.get("affected_token_1"),
            sample.get("affected_token_2"),
        )
        for sample in samples
        if "affected_token_1" in sample or "affected_token_2" in sample
    )

    return {
        "num_samples": len(samples),
        "unique_content": len(set(contents)),
        "unique_response": len(set(responses)),
        "exact_copies": exact_copies,
        "content_chars": [len(text) for text in contents],
        "content_words": [count_words(text) for text in contents],
        "response_chars": [len(text) for text in responses],
        "response_words": [count_words(text) for text in responses],
        "corruption_types": corruption_types,
        "response_labels": response_labels,
        "affected_token_pairs": affected_token_pairs,
        "top_k": top_k,
    }


def merge_counters(summaries: list[dict], key: str) -> Counter:
    merged = Counter()
    for summary in summaries:
        merged.update(summary[key])
    return merged


def merge_lists(summaries: list[dict], key: str) -> list[int]:
    merged = []
    for summary in summaries:
        merged.extend(summary[key])
    return merged


def print_summary(dataset_dir: Path, top_k: int) -> None:
    print("=" * 100)
    print(f"Dataset: {dataset_dir.name}")
    print(f"Path: {dataset_dir}")

    split_summaries = []
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.json"
        if not split_path.exists():
            print(f"\n[{split}] missing: {split_path}")
            continue

        direction, samples = load_split(split_path)
        summary = summarize_samples(samples, top_k)
        split_summaries.append(summary)

        exact_copy_rate = (
            summary["exact_copies"] / summary["num_samples"] * 100
            if summary["num_samples"]
            else 0
        )

        print(f"\n[{split}]")
        print(f"  Samples: {summary['num_samples']:,}")
        print(f"  Direction chars: {len(direction):,}")
        print(f"  Unique content: {summary['unique_content']:,}")
        print(f"  Unique responses: {summary['unique_response']:,}")
        print(f"  Exact content==response: {summary['exact_copies']:,} ({exact_copy_rate:.1f}%)")
        print(f"  Content chars: {length_stats(summary['content_chars'])}")
        print(f"  Content words: {length_stats(summary['content_words'])}")
        print(f"  Response chars: {length_stats(summary['response_chars'])}")
        print(f"  Response words: {length_stats(summary['response_words'])}")
        print(f"  Responses: {format_responses(summary['response_labels'], top_k)}")
        print(f"  Corruption types: {format_counter(summary['corruption_types'], top_k)}")

    if not split_summaries:
        return

    total_samples = sum(summary["num_samples"] for summary in split_summaries)
    total_exact_copies = sum(summary["exact_copies"] for summary in split_summaries)
    total_exact_copy_rate = total_exact_copies / total_samples * 100 if total_samples else 0

    print("\n[all]")
    print(f"  Samples: {total_samples:,}")
    print(f"  Exact content==response: {total_exact_copies:,} ({total_exact_copy_rate:.1f}%)")
    print(f"  Content chars: {length_stats(merge_lists(split_summaries, 'content_chars'))}")
    print(f"  Content words: {length_stats(merge_lists(split_summaries, 'content_words'))}")
    print(f"  Response chars: {length_stats(merge_lists(split_summaries, 'response_chars'))}")
    print(f"  Response words: {length_stats(merge_lists(split_summaries, 'response_words'))}")
    print(f"  Responses: {format_responses(merge_counters(split_summaries, 'response_labels'), top_k)}")
    print(f"  Corruption types: {format_counter(merge_counters(split_summaries, 'corruption_types'), top_k)}")


def main() -> None:
    args = parse_args()
    if not args.base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.base_dir}")

    dataset_dirs = discover_datasets(args.base_dir)
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders with JSON splits found in: {args.base_dir}")

    for dataset_dir in dataset_dirs:
        print_summary(dataset_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
