#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DIRECTION = (
    "Givet følgende sætning på dansk, afgør om den er grammatisk korrekt eller ukorrekt.\n"
    "- Hvis den er ukorrekt, skal du outputte den rettede version.\n"
    "- Hvis den er korrekt, skal du outputte den originale sætning.\n"
    "- Output kun den rettede eller originale sætning — intet andet.\n\n"
    "Regler:\n"
    "- Svar kun på dansk.\n"
    "- Bevar samme formatering og store/små bogstaver som i den originale sætning.\n"
    "- Output ingen forklaring.\n"
    "- Sig ikke, om sætningen er korrekt eller ukorrekt.\n"
)

SPLITS = ("train", "val", "test")

# HF_DATASET = "giannor/dala_gen_large_v3"
# OUTPUT_FOLDER = "output/dala_gen_large_v3_it"

# HF_DATASET = "giannor/dala_gen_large_v3_ci"
# OUTPUT_FOLDER = "output/dala_gen_large_v3_ci_it"

HF_DATASET = "giannor/dala_gen_tv2r"
OUTPUT_FOLDER = "output/dala_gen_tv2r_it"
HF_UPLOAD_ID = "dala_gen_tv2r_it"


def convert_sample(sample: dict) -> dict:
    return {
        "content": sample["corrupted"],
        "response": sample["original"],
        "corruption_type": sample["corruption_type"],
        "affected_token_1": sample["affected_token_1"],
        "affected_token_2": sample["affected_token_2"],
    }


def convert_split(dataset, split_name: str) -> dict:
    return {
        "direction": DIRECTION,
        "samples": [convert_sample(sample) for sample in dataset[split_name]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download giannor/dala_gen_v3 from Hugging Face and export "
            "train/val/test splits as JSON files."
        )
    )
    parser.add_argument(
        "--dataset",
        default=HF_DATASET,
        help="Hugging Face dataset identifier to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Directory where train.json, val.json, and test.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}

    dataset = load_dataset(args.dataset)

    for split_name in SPLITS:
        output_path = args.output_dir / f"{split_name}.json"
        payload = convert_split(dataset, split_name)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
            f.write("\n")
        
        sizes[split_name] = len(payload["samples"])

        print(f"Wrote {output_path}")
    
    print("Sizes:", sizes)


if __name__ == "__main__":
    main()
