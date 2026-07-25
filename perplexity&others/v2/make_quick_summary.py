#!/usr/bin/env python3
"""
Build jigsaw_quick_summary.json and obfuscation_quick_summary.json for the v2
perplexity/embedding-based approaches, matching the both_flagged / both_passed /
bypassed / worse / asr schema used by obfuscation/make_quick_summary.py.

v2 stores per-approach prediction jsonl files (one per test_file) rather than a
single detail file with both flagged_original and flagged_modified per record, so
original vs. attack-modified verdicts are paired by line index instead.

Usage:
    python make_quick_summary.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = SCRIPT_DIR / "test_results" / "predictions"

APPROACHES = [
    "approach1_ppl_threshold",
    "approach2_lgbm",
    "approach3_embed_char",
    "approach3_embed_sentence",
    "approach3_embed_full",
]

DATASETS = ["jackhhao", "safebench"]

BASELINE_SUFFIX = "jigsaw_original"  # shared baseline for both attack types
ATTACK_MODIFIED_SUFFIX = {
    "jigsaw": "jigsaw_obfuscated",
    "obfuscation": "obfuscation_obfuscated",
}


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%"


def load_predictions(approach: str, dataset: str, suffix: str) -> list[dict]:
    fpath = PREDICTIONS_DIR / approach / f"{dataset}_{suffix}.jsonl"
    with open(fpath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarise_approach(approach: str, dataset: str, modified_suffix: str) -> dict:
    original = load_predictions(approach, dataset, BASELINE_SUFFIX)
    modified = load_predictions(approach, dataset, modified_suffix)

    if len(original) != len(modified):
        print(
            f"  WARNING: {approach}/{dataset}: original has {len(original)} records, "
            f"modified has {len(modified)}"
        )

    n = min(len(original), len(modified))

    both_flagged = both_passed = bypassed = worse = 0
    for orig, mod in zip(original[:n], modified[:n]):
        flagged_orig = bool(orig["predicted"])
        flagged_mod = bool(mod["predicted"])
        if flagged_orig and flagged_mod:
            both_flagged += 1
        elif not flagged_orig and not flagged_mod:
            both_passed += 1
        elif flagged_orig and not flagged_mod:
            bypassed += 1
        else:
            worse += 1

    orig_flagged_total = both_flagged + bypassed
    asr = bypassed / orig_flagged_total if orig_flagged_total else None

    return {
        "both_flagged":                      {"count": both_flagged, "pct": pct(both_flagged, n)},
        "both_passed":                       {"count": both_passed,  "pct": pct(both_passed, n)},
        "flagged_orig_passed_mod (bypassed)": {"count": bypassed,    "pct": pct(bypassed, n)},
        "passed_orig_flagged_mod (worse)":   {"count": worse,        "pct": pct(worse, n)},
        "asr":                                None if asr is None else f"{asr * 100:.1f}%",
    }, n


def summarise_dataset(attack_type: str, dataset: str) -> dict:
    modified_suffix = ATTACK_MODIFIED_SUFFIX[attack_type]

    guardrails: dict[str, dict] = {}
    total_prompts: int | None = None

    for approach in APPROACHES:
        result, n = summarise_approach(approach, dataset, modified_suffix)
        guardrails[approach] = result
        total_prompts = n

        bypassed = result["flagged_orig_passed_mod (bypassed)"]
        asr_str = result["asr"] or "N/A"
        print(f"  {approach}: bypassed={bypassed['count']} ({bypassed['pct']}) | ASR={asr_str}")

    return {
        "total_prompts": total_prompts,
        "guardrails":    guardrails,
    }


def main():
    for attack_type in ATTACK_MODIFIED_SUFFIX:
        print(f"\n=== {attack_type} ===")

        summary = {
            "attack_type": attack_type,
            "datasets":    {},
        }
        for dataset in DATASETS:
            print(f" -- {dataset} --")
            summary["datasets"][dataset] = summarise_dataset(attack_type, dataset)

        out_path = SCRIPT_DIR / "test_results" / f"{attack_type}_quick_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  Saved -> {out_path.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
