#!/usr/bin/env python3
"""
Merge jackhhao manyshot_summary files.

- Base (Apr 16): manyshot_summary_1776354597.json - 3 guardrails (Llama-1B, Llama-8B, Qwen), 517 prompts
- Source (Apr 12): manyshot_summary.json - 6 guardrails, 517 prompts

For each prompt+TC entry in the base, adds the 3 missing guardrails
(Granite-3.3-8B, Granite-3.2-3B, ShieldGemma) from the Apr 12 file.
Matching is done by (original_prompt, filler_tokens) text — no IDs used.

Output: manyshot_summary_merged.json
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results_jackhhao"
BASE_FILE   = RESULTS_DIR / "manyshot_summary_1776354597.json"   # Apr 16, 3 models
SOURCE_FILE = RESULTS_DIR / "manyshot_summary.json"               # Apr 12, 6 models
OUTPUT_FILE = RESULTS_DIR / "manyshot_summary_merged.json"

MISSING_MODELS = {
    "ibm-granite/granite-guardian-3.3-8b",
    "ibm-granite/granite-guardian-3.2-3b-a800m",
    "google/shieldgemma-9b",
}


def main():
    print(f"Loading base  : {BASE_FILE.name}")
    with open(BASE_FILE, encoding="utf-8") as f:
        base = json.load(f)

    print(f"Loading source: {SOURCE_FILE.name}")
    with open(SOURCE_FILE, encoding="utf-8") as f:
        source = json.load(f)

    # Build lookup from source: (prompt_text, filler_tokens) -> guardrail_results dict
    source_lookup: dict[tuple[str, int], dict[str, dict]] = {}
    for entry in source["results"]:
        key = (entry["original_prompt"], entry["filler_tokens"])
        source_lookup[key] = {g["model_id"]: g for g in entry["guardrail_results"]}

    print(f"Base entries  : {len(base['results'])}")
    print(f"Source entries: {len(source['results'])}")

    # Merge
    not_found = 0
    for entry in base["results"]:
        key = (entry["original_prompt"], entry["filler_tokens"])
        if key not in source_lookup:
            not_found += 1
            continue
        source_guardrails = source_lookup[key]
        for model_id in MISSING_MODELS:
            if model_id in source_guardrails:
                entry["guardrail_results"].append(source_guardrails[model_id])
            else:
                print(f"  WARNING: {model_id} missing in source for TC={entry['filler_tokens']}")

    if not_found:
        print(f"WARNING: {not_found} base entries had no match in source (by prompt+TC text)")
    else:
        print("All entries matched successfully.")

    # Update run_config guardrail list
    all_models = sorted({g["model_id"] for e in base["results"] for g in e["guardrail_results"]})
    base["run_config"]["guardrails"] = all_models
    print(f"Guardrails in merged file: {all_models}")

    # Verify counts
    for entry in base["results"]:
        n = len(entry["guardrail_results"])
        if n != 6:
            print(f"  WARNING: entry TC={entry['filler_tokens']} prompt={entry['original_prompt'][:40]!r} has {n} guardrails")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False)

    print(f"\nSaved -> {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
