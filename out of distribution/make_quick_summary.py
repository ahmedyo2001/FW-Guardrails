#!/usr/bin/env python3
"""
Build ood_quick_summary.json for the out-of-distribution results folder.

Reads all ood_prompt_detail_<model>_<ts>.json files and computes per-model
detection rate (correctly flagged jailbreaks) and miss rate.

Usage:
    python make_quick_summary.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FOLDER = SCRIPT_DIR / "results"
OUTPUT_FILE = SCRIPT_DIR / "ood_quick_summary.json"

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


def main():
    detail_files = sorted(RESULTS_FOLDER.glob("ood_prompt_detail_*.json"))
    if not detail_files:
        raise FileNotFoundError(f"No ood_prompt_detail_*.json files in {RESULTS_FOLDER}")

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

        detected = sum(1 for r in records if r["flagged_original"])
        missed   = n - detected

        guardrails[model_id] = {
            "total":          n,
            "detected":       {"count": detected, "pct": pct(detected, n)},
            "missed":         {"count": missed,   "pct": pct(missed, n)},
            "detection_rate": pct(detected, n),
            "miss_rate":      pct(missed, n),
        }

        print(
            f"  {model_id}: {n} prompts | "
            f"detected={detected} ({pct(detected, n)}) | "
            f"missed={missed} ({pct(missed, n)})"
        )

    ordered = {m: guardrails[m] for m in MODEL_ORDER if m in guardrails}
    ordered.update({m: v for m, v in guardrails.items() if m not in ordered})

    summary = {
        "dataset":       "jailbreak_ood",
        "total_prompts": total_prompts,
        "guardrails":    ordered,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUTPUT_FILE.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
