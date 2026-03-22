#!/usr/bin/env python3
"""
Download jackhhao/jailbreak-classification Dataset
====================================================
Downloads the train and test splits from HuggingFace and saves them
to the 'raw datasets/' folder.

Usage:
    python download_jackhhao.py
    python download_jackhhao.py --split test
    python download_jackhhao.py --output-dir "../raw datasets"

Dependencies:
    pip install pandas
"""

import argparse
from pathlib import Path

import pandas as pd

DATASET_SPLITS = {
    "train": "hf://datasets/jackhhao/jailbreak-classification/balanced/jailbreak_dataset_train_balanced.csv",
    "test":  "hf://datasets/jackhhao/jailbreak-classification/balanced/jailbreak_dataset_test_balanced.csv",
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "raw datasets"


def download_split(split: str, output_dir: Path):
    url = DATASET_SPLITS[split]
    print(f"Downloading {split} split: {url}")
    df = pd.read_csv(url)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"jackhhao_{split}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    n_jailbreak = len(df[df["type"] == "jailbreak"])
    n_benign = len(df[df["type"] == "benign"])
    print(f"  Saved -> {out_path}")
    print(f"  {len(df)} total rows ({n_jailbreak} jailbreak, {n_benign} benign)")


def main():
    parser = argparse.ArgumentParser(
        description="Download jackhhao/jailbreak-classification dataset"
    )
    parser.add_argument("--split", nargs="+", choices=["train", "test"],
                        default=["train", "test"],
                        help="Which split(s) to download (default: both)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../raw datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    for split in args.split:
        download_split(split, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
