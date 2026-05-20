"""
Dataset loader for PPL-based jailbreak detector (Alon & Kamfonas 2023)

Loads adversarial (label=1) and benign (label=0) prompts from multiple
sources, deduplicates, then produces a stratified 80/20 train/test split.

Outputs
-------
  data/train.jsonl   — {"text": "...", "label": 0|1, "source": "..."}
  data/test.jsonl    — same format
  data/stats.json    — per-source counts + split sizes

Usage
-----
  pip install datasets scikit-learn
  python load_dataset.py
  python load_dataset.py --max-per-source 500 --out-dir my_data
"""

import argparse
import csv
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OUT_DIR        = "data"
RANDOM_SEED            = 42


# ── helpers ───────────────────────────────────────────────────────────────────

@dataclass
class Source:
    name:  str
    label: int          # 1 = adversarial, 0 = benign
    fn:    Callable     # () -> list[str]
    texts: list[str] = field(default_factory=list, repr=False)

    def load(self, max_n: int | None = None) -> int:
        try:
            raw = self.fn()
            raw = [t.strip() for t in raw if t and t.strip()]
            self.texts = raw[:max_n] if max_n is not None else raw
            print(f"  ✓ [{self.name}]: {len(self.texts)}")
            return len(self.texts)
        except Exception as e:
            print(f"  ✗ [{self.name}]: {e}")
            return 0


def dedup(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_jsonl(path: str, records: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── adversarial sources (label = 1) ──────────────────────────────────────────

def make_adversarial_sources() -> list[Source]:

    # 1. GCG machine-generated suffixes (local JSONL)
    def gcg():
        path = "gcg_attacks.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Generate with "
                "https://github.com/llm-attacks/llm-attacks "
                "and save each prompt as {\"prompt\": \"...\"}  per line."
            )
        with open(path, encoding="utf-8") as f:
            return [json.loads(l)["prompt"] for l in f]

    # 2. rubend18 — human GPT-4 jailbreaks (paper original, ~79 prompts)
    def rubend18():
        ds  = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        col = "Prompt" if "Prompt" in ds.column_names else ds.column_names[0]
        return [ex[col] for ex in ds]

    # 3a. TrustAIRLab in-the-wild — May 2023 snapshot
    def itw_may():
        ds  = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                           "jailbreak_2023_05_07", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        return list({ex[col] for ex in ds})       # set → deduplicate within source

    # 3b. TrustAIRLab in-the-wild — Dec 2023 snapshot
    def itw_dec():
        ds  = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                           "jailbreak_2023_12_25", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        return list({ex[col] for ex in ds})

    # 4. JailbreakBench JBB-Behaviors (NeurIPS 2024, 100 curated behaviors)
    def jbb():
        ds  = load_dataset("JailbreakBench/JBB-Behaviors",
                           "behaviors", split="train")
        col = "Goal" if "Goal" in ds.column_names else ds.column_names[0]
        return [ex[col] for ex in ds if ex[col]]

    # 5. HarmBench behaviors — raw CSV from GitHub (510 text behaviors)
    def harmbench():
        url = (
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
            "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
        )
        with urllib.request.urlopen(url, timeout=20) as r:
            content = r.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        return [row["Behavior"] for row in reader if row.get("Behavior", "").strip()]

    # 6. WildJailbreak adversarial-harmful (WildTeaming, AllenAI)
    #    Multi-tactic, fluent — the hardest attack type for a PPL filter
    def wildjailbreak_adv():
        ds = load_dataset("allenai/wildjailbreak", "train", split="train")
        return [
            ex["adversarial"] for ex in ds
            if ex.get("data_type") == "adversarial_harmful"
            and ex.get("adversarial", "").strip()
        ]

    # 7. AdvBench harmful behaviors (Zou et al. 2023 classic benchmark)
    def advbench():
        ds  = load_dataset("walledai/AdvBench", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        return [ex[col] for ex in ds if ex[col]]

    return [
        Source("2_rubend18_gpt4_jb",       1, rubend18),
        Source("4_jailbreakbench",         1, jbb),
        Source("5_harmbench",              1, harmbench),
        Source("6_wildjailbreak_adv",      1, wildjailbreak_adv),
        Source("7_advbench",               1, advbench),
    ]


# ── benign sources (label = 0) ────────────────────────────────────────────────

def make_benign_sources() -> list[Source]:

    # 1. DocRED — multi-sentence passages (paper original)
    def docred():
        ds = load_dataset("docred", split="validation", trust_remote_code=True)
        return [" ".join(ex.get("sents", [""])) for ex in ds]

    # 2. BoolQ — instruction + passage (paper original)
    def boolq():
        ds = load_dataset("super_glue", "boolq", split="validation")
        return [
            f"Read the following passage and answer the question:\n"
            f"{ex['question']}\n{ex['passage']}"
            for ex in ds
        ]

    # 3. WildJailbreak vanilla-benign
    #    Contrastive: same style as adversarial queries but harmless.
    #    Critical for reducing false positives on sensitive-looking benign prompts.
    def wildjailbreak_benign():
        ds = load_dataset("allenai/wildjailbreak", "train", split="train")
        return [
            ex["vanilla"] for ex in ds
            if ex.get("data_type") == "vanilla_benign"
            and ex.get("vanilla", "").strip()
        ]

    # 4. OpenOrca — diverse real user instructions (benign)
    # NOTE: OpenOrca is a streaming dataset (4M+ rows), so we must set a hard
    # stop. When no --max-per-source is given, Source.load() will not slice,
    # so we cap here at 50_000 to avoid an infinite loop. Pass --max-per-source
    # to override this to a smaller number.
    OPENORCA_HARD_CAP = 50_000

    def openorca():
        ds  = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        col = "question" if "question" in ds.features else "input"
        out = []
        for ex in ds:
            q = ex.get(col, "").strip()
            if q:
                out.append(q)
            if len(out) >= OPENORCA_HARD_CAP:
                break
        return out

    return [
        Source("1_docred",                 0, docred),
        Source("2_boolq",                  0, boolq),
        Source("3_wildjailbreak_benign",   0, wildjailbreak_benign),
        Source("4_openorca",               0, openorca),
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Max prompts from any single source. "
                             "Omit to load full datasets.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory for train/test JSONL files")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for test (default 0.2)")
    args = parser.parse_args()

    M = args.max_per_source

    # ── load all sources ──────────────────────────────────────────────────────
    print("\n=== Adversarial sources (label=1) ===")
    adv_sources = make_adversarial_sources()
    for s in adv_sources:
        s.load(M)

    print("\n=== Benign sources (label=0) ===")
    ben_sources = make_benign_sources()
    for s in ben_sources:
        s.load(M)

    # ── build record list ─────────────────────────────────────────────────────
    all_sources = adv_sources + ben_sources
    records = []
    stats   = {}

    for s in all_sources:
        for t in s.texts:
            records.append({"text": t, "label": s.label, "source": s.name})
        stats[s.name] = {"label": s.label, "count": len(s.texts)}

    # ── global dedup ──────────────────────────────────────────────────────────
    before = len(records)
    records = dedup(records)
    after  = len(records)
    print(f"\n=== Deduplication: {before} → {after} "
          f"({before - after} duplicates removed) ===")

    # ── label summary ─────────────────────────────────────────────────────────
    n_adv = sum(1 for r in records if r["label"] == 1)
    n_ben = sum(1 for r in records if r["label"] == 0)
    print(f"\n  Adversarial (1): {n_adv}")
    print(f"  Benign      (0): {n_ben}")
    print(f"  Total:           {len(records)}")

    if n_adv == 0 or n_ben == 0:
        raise RuntimeError("One class is empty — check dataset loading errors above.")

    # ── stratified 80/20 split ────────────────────────────────────────────────
    labels = [r["label"] for r in records]
    train_records, test_records = train_test_split(
        records,
        test_size=args.test_size,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    print(f"\n=== Split (stratified {int((1-args.test_size)*100)}/"
          f"{int(args.test_size*100)}) ===")
    print(f"  Train: {len(train_records)}  "
          f"(adv={sum(r['label']==1 for r in train_records)}, "
          f"ben={sum(r['label']==0 for r in train_records)})")
    print(f"  Test:  {len(test_records)}  "
          f"(adv={sum(r['label']==1 for r in test_records)}, "
          f"ben={sum(r['label']==0 for r in test_records)})")

    # ── write outputs ─────────────────────────────────────────────────────────
    train_path = os.path.join(args.out_dir, "train.jsonl")
    test_path  = os.path.join(args.out_dir, "test.jsonl")
    stats_path = os.path.join(args.out_dir, "stats.json")

    write_jsonl(train_path, train_records)
    write_jsonl(test_path,  test_records)

    stats["_split"] = {
        "train": len(train_records),
        "test":  len(test_records),
        "test_fraction": args.test_size,
        "seed": RANDOM_SEED,
        "total_before_dedup": before,
        "total_after_dedup":  after,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved → {train_path}")
    print(f"  Saved → {test_path}")
    print(f"  Saved → {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()