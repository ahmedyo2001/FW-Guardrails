#!/usr/bin/env python3
"""
Download SafeBench (ForbidQI) Dataset
=====================================
Downloads the SafeBench dataset from the FigStep GitHub repository.
This is the dataset used in the Jigsaw Puzzle (JSP) attack paper.

Filters out non-harmful categories (Legal Opinion, Financial Advice,
Health Consultation) and saves the remaining 350 prompts.

Usage:
    python download_safebench.py
    python download_safebench.py --output-dir "../raw datasets"

Dependencies:
    pip install pandas
"""

import argparse
from pathlib import Path

import pandas as pd

DATASET_URL = "https://raw.githubusercontent.com/CryptoAILab/FigStep/main/data/question/safebench.csv"

EXCLUDE_CATEGORIES = {
    "Legal Opinion",
    "Financial Advice",
    "Health Consultation",
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "raw datasets"


def download(output_dir: Path):
    print(f"Downloading SafeBench dataset: {DATASET_URL}")
    df = pd.read_csv(DATASET_URL)
    print(f"  Downloaded {len(df)} rows")

    # Filter out non-harmful categories
    before = len(df)
    df = df[~df["category_name"].isin(EXCLUDE_CATEGORIES)]
    print(f"  Excluded {before - len(df)} rows from: {', '.join(EXCLUDE_CATEGORIES)}")
    print(f"  Remaining: {len(df)} rows")

    # Print per-category breakdown
    print("\n  Categories:")
    for cat, count in df["category_name"].value_counts().sort_index().items():
        print(f"    {cat}: {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "safebench.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SafeBench (ForbidQI) dataset"
    )
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../raw datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    download(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
