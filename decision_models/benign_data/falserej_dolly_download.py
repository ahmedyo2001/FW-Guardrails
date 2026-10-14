#!/usr/bin/env python3
"""
Download benign datasets for FPR evaluation.

Downloads:
- FalseReject (AmazonScience): every other prompt per category (44 categories, borderline benign)
- Dolly (Databricks): 50 prompts per category (8 categories, clearly benign)
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


def download_falsereject(output_file: str) -> None:
    """
    Download FalseReject test set and sample every other prompt per category.

    Args:
        output_file: Path to output JSON file
    """
    print("Downloading FalseReject (AmazonScience)...")
    dataset = load_dataset("AmazonScience/FalseReject", split="test")
    print(f"Total FalseReject test prompts: {len(dataset)}")

    # Get all unique categories
    categories = sorted(set(dataset["category"]))
    print(f"Found {len(categories)} categories\n")

    results = []

    for category in categories:
        # Filter by category
        category_subset = dataset.filter(lambda x: x["category"] == category)
        total = len(category_subset)

        # Take every other prompt (indices 0, 2, 4, ...)
        sampled_indices = list(range(0, total, 2))
        sampled = category_subset.select(sampled_indices)

        for idx, item in enumerate(sampled):
            results.append({
                "prompt_id": f"falsereject_{category}_{idx+1:03d}",
                "prompt": item["prompt"],
                "category": category,
                "dataset": "falsereject",
                "label": "borderline_benign",
            })

        print(f"  [{category}] {len(sampled)}/{total} prompts sampled")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(results)} FalseReject prompts to {output_file}\n")


def download_dolly(output_file: str, n_per_category: int = 50) -> None:
    """
    Download Dolly and sample n prompts per category.

    Args:
        output_file: Path to output JSON file
        n_per_category: Number of prompts per category
    """
    print("Downloading Dolly (Databricks)...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"Total Dolly prompts: {len(dataset)}\n")

    results = []

    # Get all unique categories present in the dataset
    categories = sorted(set(dataset["category"]))

    for category in categories:
        category_subset = dataset.filter(lambda x: x["category"] == category)
        available = len(category_subset)
        sample_size = min(n_per_category, available)

        sampled = category_subset.shuffle(seed=42).select(range(sample_size))

        for idx, item in enumerate(sampled):
            results.append({
                "prompt_id": f"dolly_{category.replace(' ', '_')}_{idx+1:03d}",
                "prompt": item["instruction"],
                "category": category,
                "dataset": "dolly",
                "label": "benign",
            })

        print(f"  [{category}] Sampled {sample_size}/{available} prompts")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(results)} Dolly prompts to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download FalseReject and Dolly benign datasets"
    )
    parser.add_argument(
        "--falsereject-output",
        default="falsereject_borderline.json",
        help="Output file for FalseReject (default: falsereject_borderline.json)"
    )
    parser.add_argument(
        "--dolly-output",
        default="dolly_benign.json",
        help="Output file for Dolly (default: dolly_benign.json)"
    )
    parser.add_argument(
        "--dolly-n",
        type=int,
        default=50,
        help="Prompts per Dolly category (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    falsereject_path = output_dir / args.falsereject_output
    dolly_path = output_dir / args.dolly_output

    download_falsereject(str(falsereject_path))
    download_dolly(str(dolly_path), n_per_category=args.dolly_n)

    print("Done.")
    print(f"  FalseReject → {falsereject_path}")
    print(f"  Dolly       → {dolly_path}")


if __name__ == "__main__":
    main()