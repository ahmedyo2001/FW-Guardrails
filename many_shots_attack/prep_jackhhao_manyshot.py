#!/usr/bin/env python3
"""
Prep jackhhao Dataset for Many-Shot Attack
============================================
Reads the jackhhao/jailbreak-classification CSV (downloaded by download_jackhhao.py),
extracts jailbreak prompts, and saves them as a JSON list for the many-shot notebook.

Usage:
    python prep_jackhhao_manyshot.py
    python prep_jackhhao_manyshot.py --dataset "../raw datasets/jackhhao_train.csv" --limit 50
    python prep_jackhhao_manyshot.py --output objectives.json

Prerequisites:
    python ../dataset_downloaders/download_jackhhao.py   (to download the dataset first)

Dependencies:
    pip install pandas
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "raw datasets" / "jackhhao_test.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "objectives.json"


def main():
    parser = argparse.ArgumentParser(
        description="Extract jailbreak prompts from jackhhao CSV for many-shot attack"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="Path to jackhhao CSV (default: ../raw datasets/jackhhao_test.csv)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output JSON file (default: ../many_shots_attack/objectives.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of prompts to extract (default: all)")
    args = parser.parse_args()

    print(f"Reading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    prompts = df[df["type"] == "jailbreak"]["prompt"].dropna().tolist()
    print(f"  {len(prompts)} jailbreak prompts (of {len(df)} total rows)")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"  Capped to {args.limit}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(prompts)} objectives -> {out_path}")
    print(f"Set OBJECTIVES_FILE={out_path} in .env or notebook to use.")


if __name__ == "__main__":
    main()
