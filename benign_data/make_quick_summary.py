#!/usr/bin/env python3
"""
Build benign_quick_summary.json aggregating all 4 benign result folders.

Reads all benign_prompt_detail_<model>_<ts>.json files in each folder and
computes per-model acceptance rate (= 1 - false rejection rate).

Output: benign_data/benign_quick_summary.json

Usage:
    python make_quick_summary.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FOLDERS = [
    "dolly_results",
    "falsereject_results",
    "or_bench_results",
    "xstest_results",
]
OUTPUT_FILE = SCRIPT_DIR / "benign_quick_summary.json"

MODEL_ORDER = [
    "meta-llama/Llama-Guard-3-1B",
    "meta-llama/Llama-Guard-3-8B",
    "ibm-granite/granite-guardian-3.3-8b",
    "ibm-granite/granite-guardian-3.2-3b-a800m",
    "google/shieldgemma-9b",
    "Qwen/Qwen3Guard-Gen-8B",
]


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%"


def summarise_folder(folder: Path) -> dict:
    detail_files = sorted(folder.glob("benign_prompt_detail_*.json"))
    if not detail_files:
        raise FileNotFoundError(f"No detail files found in {folder}")

    guardrails: dict[str, dict] = {}
    total_prompts: int | None = None

    for fpath in detail_files:
        with open(fpath, encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            print(f"  WARNING: {fpath.name} is empty, skipping")
            continue

        model_id: str = records[0]["guardrail_model_id"]
        n = len(records)

        if total_prompts is None:
            total_prompts = n
        elif total_prompts != n:
            print(f"  WARNING: {fpath.name} has {n} records, expected {total_prompts}")

        rejected = sum(1 for r in records if r["flagged_original"])
        accepted = n - rejected

        guardrails[model_id] = {
            "total":           n,
            "accepted":        {"count": accepted,  "pct": pct(accepted, n)},
            "rejected":        {"count": rejected,  "pct": pct(rejected, n)},
            "acceptance_rate": pct(accepted, n),
            "false_rejection_rate": pct(rejected, n),
        }

        print(
            f"  {model_id}: {n} prompts | "
            f"accepted={accepted} ({pct(accepted, n)}) | "
            f"false-rejected={rejected} ({pct(rejected, n)})"
        )

    ordered = {m: guardrails[m] for m in MODEL_ORDER if m in guardrails}
    ordered.update({m: v for m, v in guardrails.items() if m not in ordered})

    return {
        "dataset":       folder.name,
        "total_prompts": total_prompts,
        "guardrails":    ordered,
    }


def main():
    all_datasets = []

    for folder_name in RESULT_FOLDERS:
        folder = SCRIPT_DIR / folder_name
        print(f"\n=== {folder_name} ===")
        all_datasets.append(summarise_folder(folder))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_datasets, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUTPUT_FILE.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
