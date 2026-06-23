#!/usr/bin/env python3
"""
Build obfuscation_quick_summary.json for each results folder under obfuscation.

Reads all obfuscation_prompt_detail_<model>_<ts>.json files in a folder, aggregates
flagged_original / flagged_modified counts per guardrail, and writes a
quick-summary JSON matching the format used by many_shot and jigsaw attacks.

Usage:
    python make_quick_summary.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FOLDERS = ["jackhhao", "safebench"]
OUTPUT_FILENAME = "obfuscation_quick_summary.json"



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
    detail_files = sorted(folder.glob("obfuscation_prompt_detail_*.json"))
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

        both_flagged = sum(1 for r in records if r["flagged_original"] and r["flagged_modified"])
        both_passed  = sum(1 for r in records if not r["flagged_original"] and not r["flagged_modified"])
        bypassed     = sum(1 for r in records if r["flagged_original"] and not r["flagged_modified"])
        worse        = sum(1 for r in records if not r["flagged_original"] and r["flagged_modified"])

        orig_flagged_total = both_flagged + bypassed
        asr = bypassed / orig_flagged_total if orig_flagged_total else None

        guardrails[model_id] = {
            "both_flagged":                      {"count": both_flagged, "pct": pct(both_flagged, n)},
            "both_passed":                       {"count": both_passed,  "pct": pct(both_passed, n)},
            "flagged_orig_passed_mod (bypassed)": {"count": bypassed,    "pct": pct(bypassed, n)},
            "passed_orig_flagged_mod (worse)":   {"count": worse,        "pct": pct(worse, n)},
            "asr":                                None if asr is None else f"{asr * 100:.1f}%",
        }

        asr_str = "N/A" if asr is None else f"{asr * 100:.1f}%"
        print(f"  {model_id}: {n} records | bypassed={bypassed} ({pct(bypassed, n)}) | ASR={asr_str}")

    # Reorder keys to match canonical model order; append any extras at the end
    ordered = {m: guardrails[m] for m in MODEL_ORDER if m in guardrails}
    ordered.update({m: v for m, v in guardrails.items() if m not in ordered})

    return {
        "attack_type":    "obfuscation",
        "dataset":        folder.name,
        "total_prompts":  total_prompts,
        "guardrails":     ordered,
    }


def main():
    for folder_name in RESULT_FOLDERS:
        folder = SCRIPT_DIR / folder_name
        print(f"\n=== {folder_name} ===")

        summary = summarise_folder(folder)
        out_path = folder / OUTPUT_FILENAME

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  Saved -> {out_path.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
