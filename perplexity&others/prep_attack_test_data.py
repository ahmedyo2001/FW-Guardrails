"""
Converts manyshot, jigsaw, obfuscation, out-of-distribution, and benign
result/source files into .jsonl test files compatible with the --test
argument of all three guardrail approaches.

Output: perplexity&others/attack_tests/
  jackhhao_manyshot_0tokens.jsonl
  jackhhao_manyshot_2000tokens.jsonl
  jackhhao_manyshot_4000tokens.jsonl
  jackhhao_manyshot_6000tokens.jsonl
  jackhhao_manyshot_8000tokens.jsonl
  safebench_manyshot_0tokens.jsonl
  safebench_manyshot_2000tokens.jsonl
  safebench_manyshot_4000tokens.jsonl
  safebench_manyshot_6000tokens.jsonl
  safebench_manyshot_8000tokens.jsonl
  jackhhao_jigsaw_original.jsonl
  jackhhao_jigsaw_obfuscated.jsonl
  safebench_jigsaw_original.jsonl
  safebench_jigsaw_obfuscated.jsonl
  jackhhao_obfuscation_original.jsonl
  jackhhao_obfuscation_obfuscated.jsonl
  safebench_obfuscation_original.jsonl
  safebench_obfuscation_obfuscated.jsonl
  ood_jailbreak.jsonl
  benign_dolly.jsonl
  benign_falsereject.jsonl
  benign_xstest.jsonl
"""

import json
import os
from collections import defaultdict

REPO_ROOT  = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "attack_tests")

MANYSHOT_SOURCES = {
    "jackhhao": [
        os.path.join(REPO_ROOT, "many_shots_attack", "results_jackhhao",
                     "manyshot_detail_meta-llama_Llama-Guard-3-1B_1776348815.json"),
        os.path.join(REPO_ROOT, "many_shots_attack", "results_jackhhao",
                     "manyshot_detail_meta-llama_Llama-Guard-3-1B_4k8k.json"),
    ],
    "safebench": [
        os.path.join(REPO_ROOT, "many_shots_attack", "results_safebench",
                     "manyshot_detail_meta-llama_Llama-Guard-3-1B_1776355649.json"),
        os.path.join(REPO_ROOT, "many_shots_attack", "results_safebench",
                     "manyshot_detail_meta-llama_Llama-Guard-3-1B_s4k8k.json"),
    ],
}

JIGSAW_SOURCES = {
    "jackhhao": os.path.join(REPO_ROOT, "jigsaw_attack", "results",
                             "jigsaw_detail_meta-llama_Llama-Guard-3-1B_1776310393.json"),
    "safebench": os.path.join(REPO_ROOT, "jigsaw_attack", "results_safebench",
                              "jigsaw_detail_meta-llama_Llama-Guard-3-1B_1776309115.json"),
}

OBFUSCATION_SOURCES = {
    "jackhhao": os.path.join(REPO_ROOT, "obfuscation", "jackhhao",
                             "obfuscation_prompt_detail_meta-llama_Llama-Guard-3-1B_1782071479.json"),
    "safebench": os.path.join(REPO_ROOT, "obfuscation", "safebench",
                              "obfuscation_prompt_detail_meta-llama_Llama-Guard-3-1B_1782067701.json"),
}

# All entries are jailbreak prompts (label=1) sourced from outside the
# manyshot/jigsaw/obfuscation attack families — tests generalization.
OOD_SOURCE = os.path.join(REPO_ROOT, "out of distribution",
                          "jailbreak_original_or_alchemy_full.json")

# Benign prompt sets — every entry is label=0, used to measure false-refusal /
# false-positive rate of the guardrail approaches.
BENIGN_SOURCES = {
    "dolly":       os.path.join(REPO_ROOT, "benign_data", "dolly_benign.json"),
    "falsereject": os.path.join(REPO_ROOT, "benign_data", "falsereject_benign.json"),
    "xstest":      os.path.join(REPO_ROOT, "benign_data", "xstest_benign.json"),
}


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {os.path.basename(path)}: {len(records)} records")


def process_manyshot(dataset_name, source_files):
    groups = defaultdict(list)
    seen = set()

    for path in source_files:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            key = (entry["combined_prompt"], entry["filler_tokens"])
            if key in seen:
                continue
            seen.add(key)
            groups[entry["filler_tokens"]].append(entry)

    for filler_tokens, entries in sorted(groups.items()):
        records = [
            {
                "text":   e["combined_prompt"],
                "label":  1,
                "source": f"{dataset_name}_manyshot_{filler_tokens}tokens",
            }
            for e in entries
        ]
        out_path = os.path.join(OUT_DIR, f"{dataset_name}_manyshot_{filler_tokens}tokens.jsonl")
        write_jsonl(out_path, records)


def process_jigsaw(dataset_name, source_file):
    with open(source_file, encoding="utf-8") as f:
        entries = json.load(f)

    seen_obj = set()
    originals   = []
    obfuscated  = []

    for entry in entries:
        obj = entry["objective"]
        if obj in seen_obj:
            continue
        seen_obj.add(obj)
        originals.append({
            "text":   obj,
            "label":  1,
            "source": f"{dataset_name}_jigsaw_original",
        })
        obfuscated.append({
            "text":   entry["jsp_prompt_raw"],
            "label":  1,
            "source": f"{dataset_name}_jigsaw_obfuscated",
        })

    write_jsonl(os.path.join(OUT_DIR, f"{dataset_name}_jigsaw_original.jsonl"),   originals)
    write_jsonl(os.path.join(OUT_DIR, f"{dataset_name}_jigsaw_obfuscated.jsonl"), obfuscated)


def process_obfuscation(dataset_name, source_file):
    with open(source_file, encoding="utf-8") as f:
        entries = json.load(f)

    seen_obj   = set()
    originals  = []
    obfuscated = []

    for entry in entries:
        obj = entry["objective"]
        if obj in seen_obj:
            continue
        seen_obj.add(obj)
        originals.append({
            "text":   obj,
            "label":  1,
            "source": f"{dataset_name}_obfuscation_original",
        })
        obfuscated.append({
            "text":   entry["combined_prompt"],
            "label":  1,
            "source": f"{dataset_name}_obfuscation_obfuscated",
        })

    write_jsonl(os.path.join(OUT_DIR, f"{dataset_name}_obfuscation_original.jsonl"),   originals)
    write_jsonl(os.path.join(OUT_DIR, f"{dataset_name}_obfuscation_obfuscated.jsonl"), obfuscated)


def process_ood(source_file):
    with open(source_file, encoding="utf-8") as f:
        entries = json.load(f)

    seen    = set()
    records = []

    for entry in entries:
        text = entry["text"]
        if text in seen:
            continue
        seen.add(text)
        records.append({
            "text":   text,
            "label":  1,
            "source": "ood_jailbreak",
        })

    write_jsonl(os.path.join(OUT_DIR, "ood_jailbreak.jsonl"), records)


def process_benign(dataset_name, source_file):
    with open(source_file, encoding="utf-8") as f:
        entries = json.load(f)

    seen    = set()
    records = []

    for entry in entries:
        text = entry["prompt"]
        if text in seen:
            continue
        seen.add(text)
        records.append({
            "text":   text,
            "label":  0,
            "source": f"benign_{dataset_name}",
        })

    write_jsonl(os.path.join(OUT_DIR, f"benign_{dataset_name}.jsonl"), records)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Manyshot - Jackhhao ===")
    process_manyshot("jackhhao", MANYSHOT_SOURCES["jackhhao"])

    print("\n=== Manyshot - Safebench ===")
    process_manyshot("safebench", MANYSHOT_SOURCES["safebench"])

    print("\n=== Jigsaw - Jackhhao ===")
    process_jigsaw("jackhhao", JIGSAW_SOURCES["jackhhao"])

    print("\n=== Jigsaw - Safebench ===")
    process_jigsaw("safebench", JIGSAW_SOURCES["safebench"])

    print("\n=== Obfuscation - Jackhhao ===")
    process_obfuscation("jackhhao", OBFUSCATION_SOURCES["jackhhao"])

    print("\n=== Obfuscation - Safebench ===")
    process_obfuscation("safebench", OBFUSCATION_SOURCES["safebench"])

    print("\n=== Out of distribution ===")
    process_ood(OOD_SOURCE)

    print("\n=== Benign ===")
    for name, path in BENIGN_SOURCES.items():
        process_benign(name, path)

    print(f"\nAll files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
