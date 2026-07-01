#!/usr/bin/env python3
"""
Download benign hard-negative datasets for FPR evaluation.

Downloads:
- OR-Bench: 50 prompts per category (500 total across 10 categories)
- XSTest: full dataset (450 prompts)
"""

import json
import os
import argparse
from pathlib import Path
from datasets import load_dataset


def _load_env(path: str = ".env") -> None:
    """Load .env file into os.environ, searching from script dir up to project root."""
    search = Path(__file__).resolve().parent
    for _ in range(4):  # walk up at most 4 levels
        candidate = search / path
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
            print(f"Loaded .env from {candidate}")
            return
        search = search.parent
    print("No .env file found.")

_load_env()


OR_BENCH_CATEGORIES = [
    "deception",
    "harassment",
    "harmful",
    "hate",
    "illegal",
    "privacy",
    "self-harm",
    "sexual",
    "unethical",
    "violence",
]


def download_or_bench(output_file: str, n_per_category: int = 50) -> None:
    """
    Download OR-Bench and sample n prompts per category.

    Args:
        output_file: Path to output JSON file
        n_per_category: Number of prompts to sample per category
    """
    print("Downloading OR-Bench...")
    dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train", token=os.environ.get("HF_TOKEN"))
    print(f"Total OR-Bench prompts: {len(dataset)}")

    results = []

    for category in OR_BENCH_CATEGORIES:
        # Filter by category
        category_subset = dataset.filter(lambda x: x["category"] == category)

        available = len(category_subset)
        sample_size = min(n_per_category, available)

        if available == 0:
            print(f"  [{category}] No prompts found — skipping")
            continue

        # Sample n prompts
        sampled = category_subset.shuffle(seed=42).select(range(sample_size))

        for idx, item in enumerate(sampled):
            results.append({
                "prompt_id": f"orbench_{category}_{idx+1:03d}",
                "prompt": item["prompt"],
                "category": category,
                "dataset": "or-bench",
                "label": "benign",
            })

        print(f"  [{category}] Sampled {sample_size}/{available} prompts")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nOK Saved {len(results)} OR-Bench prompts to {output_file}\n")


def download_xstest(output_file: str) -> None:
    """
    Download XSTest full dataset.

    Args:
        output_file: Path to output JSON file
    """
    print("Downloading XSTest...")
    dataset = load_dataset("Paul/XSTest", split="train", token=os.environ.get("HF_TOKEN"))
    print(f"Total XSTest prompts: {len(dataset)}")

    results = []

    for idx, item in enumerate(dataset):
        results.append({
            "prompt_id": f"xstest_{idx+1:04d}",
            "prompt": item["prompt"],
            "category": item.get("type", "unknown"),
            "dataset": "xstest",
            "label": "benign",
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"OK Saved {len(results)} XSTest prompts to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download OR-Bench and XSTest benign hard-negative datasets"
    )
    parser.add_argument(
        "--or-bench-output",
        default="or_bench_benign.json",
        help="Output file for OR-Bench (default: or_bench_benign.json)"
    )
    parser.add_argument(
        "--xstest-output",
        default="xstest_benign.json",
        help="Output file for XSTest (default: xstest_benign.json)"
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=50,
        help="Number of prompts per OR-Bench category (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    or_bench_path = output_dir / args.or_bench_output
    xstest_path = output_dir / args.xstest_output

    download_or_bench(str(or_bench_path), n_per_category=args.n_per_category)
    download_xstest(str(xstest_path))

    print("Done.")
    print(f"  OR-Bench -> {or_bench_path}")
    print(f"  XSTest   -> {xstest_path}")


if __name__ == "__main__":
    main()