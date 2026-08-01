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
  pip install datasets scikit-learn rapidfuzz
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
from pathlib import Path
from typing import Callable

from datasets import load_dataset
from rapidfuzz import fuzz, process
from sklearn.model_selection import train_test_split


def _load_env(path: str = ".env") -> None:
    search = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = search / path
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
            return
        search = search.parent

_load_env()

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OUT_DIR        = "v3/training_data"
RANDOM_SEED            = 42


# ── helpers ───────────────────────────────────────────────────────────────────
# dataclass is a wrapper that allows us to write the code below as a class
@dataclass
class Source:
    name:  str
    label: int          # 1 = adversarial, 0 = benign
    fn:    Callable     # () -> list[str]
    # field default factory is used to create a new array for each source class so it doesn't use the same array
    texts: list[str] = field(default_factory=list, repr=False)


    # runs the fn and loads the data
    def load(self, max_n: int | None = None) -> int:
        try:
            raw = self.fn()
            raw = [t.strip() for t in raw if t and t.strip()]
            self.texts = raw[:max_n] if max_n is not None else raw
            print(f"  OK [{self.name}]: {len(self.texts)}")
            return len(self.texts)
        except Exception as e:
            print(f"  FAIL [{self.name}]: {e}")
            return 0

# many-shot attack objectives are drawn from the same itw jailbreak pool as
# 3B_itw_dec; exact matches would let the classifier train on prompts it's
# later "attacked" with, so they're excluded at load time (see FW-Guardrails
# many_shots_attack contamination check).
def _load_manyshot_objectives() -> list[str]:
    path = Path(__file__).resolve().parent.parent / "many_shots_attack" / "objectives.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        objectives = json.load(f)
    # dedupe while preserving order
    return list(dict.fromkeys(" ".join(o.strip().split()) for o in objectives))


# dedup data
def dedup(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

#writes json
def write_jsonl(path: str, records: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── adversarial sources (label = 1) ──────────────────────────────────────────

def make_adversarial_sources() -> list[Source]:

    # 1. GCG machine-generated suffixes (local JSONL)
    # def gcg():
    #     path = "gcg_attacks.jsonl"
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(
    #             f"{path} not found. Generate with "
    #             "https://github.com/llm-attacks/llm-attacks "
    #             "and save each prompt as {\"prompt\": \"...\"}  per line."
    #         )
    #     with open(path, encoding="utf-8") as f:
    #         return [json.loads(l)["prompt"] for l in f]

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
        return list({ex[col] for ex in ds})       # set -> deduplicate within source

    # 3b. TrustAIRLab in-the-wild — Dec 2023 snapshot
    def itw_dec():
        ds  = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                           "jailbreak_2023_12_25", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        prompts = list({ex[col] for ex in ds})

        objectives = _load_manyshot_objectives()
        if objectives:
            objectives_set = set(objectives)
            kept, exact_removed, fuzzy_removed = [], 0, 0
            for p in prompts:
                norm_p = " ".join(p.strip().split())
                if norm_p in objectives_set:
                    exact_removed += 1
                    continue
                match = process.extractOne(norm_p, objectives, scorer=fuzz.QRatio)
                if match and match[1] >= 95:
                    fuzzy_removed += 1
                    continue
                kept.append(p)
            prompts = kept
            if exact_removed or fuzzy_removed:
                print(f"    [3B_itw_dec] excluded {exact_removed + fuzzy_removed} prompts "
                      f"found in many_shots_attack/objectives.json "
                      f"({exact_removed} exact, {fuzzy_removed} fuzzy>=95%)")
        return prompts

    # 4. JailbreakBench JBB-Behaviors (NeurIPS 2024, 100 curated behaviors)
    def jbb():
        ds  = load_dataset("JailbreakBench/JBB-Behaviors",
                           "behaviors", split="harmful")
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
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset("allenai/wildjailbreak", "train", split="train",
                          token=token, streaming=True)
        out = []
        for ex in ds:
            if ex.get("data_type") == "adversarial_harmful" and ex.get("adversarial", "").strip():
                out.append(ex["adversarial"])
                if len(out) >= 8273:  # 10% of ~82,728
                    break
        return out

    # 7. AdvBench harmful behaviors (Zou et al. 2023 classic benchmark)
    def advbench():
        ds  = load_dataset("walledai/AdvBench", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        return [ex[col] for ex in ds if ex[col]]

    return [
        Source("2_rubend18_gpt4_jb",       1, rubend18),
        Source("3B_itw_dec",       1, itw_dec),
        Source("4_jailbreakbench",         1, jbb),
        Source("5_harmbench",              1, harmbench),
        Source("6_wildjailbreak_adv",      1, wildjailbreak_adv),
        Source("7_advbench",               1, advbench),
    ]


# ── benign sources (label = 0) ────────────────────────────────────────────────

def make_benign_sources() -> list[Source]:

    # 1. DocRED — multi-sentence passages (paper original)
    def docred():
        ds = load_dataset("thunlp/docred", split="train", trust_remote_code=True)
        rows = []
        for ex in ds:
            sents = ex.get("sents", [])
            if sents and isinstance(sents[0], list):
                rows.append(" ".join(" ".join(s) for s in sents))
            else:
                rows.append(" ".join(sents))
        return rows

    # 2. BoolQ — instruction + passage (paper original)
    def boolq():
        ds = load_dataset("google/boolq", split="train[:10%]")
        return [
            f"Read the following passage and answer the question:\n"
            f"{ex['question']}\n{ex['passage']}"
            for ex in ds
        ]

    # 3. SQuAD-v2 — reading comprehension questions (paper original)
    def squad_v2():
        ds = load_dataset("rajpurkar/squad_v2", split="train[:10%]")
        return [
            f"{ex['question']}\n{ex['context']}"
            for ex in ds if ex.get("question") and ex.get("context")
        ]

    # 4. Platypus — instruction prompts (paper original)
    def platypus():
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train[:10%]")
        col = "instruction" if "instruction" in ds.column_names else ds.column_names[0]
        return [ex[col] for ex in ds if ex.get(col, "").strip()]

    # 5. WildJailbreak vanilla-benign
    def wildjailbreak_benign():
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset("allenai/wildjailbreak", "train", split="train",
                          token=token, streaming=True)
        out = []
        for ex in ds:
            if ex.get("data_type") == "vanilla_benign" and ex.get("vanilla", "").strip():
                out.append(ex["vanilla"])
                if len(out) >= 5005:  # 10% of ~50,050
                    break
        return out

    return [
        Source("1_docred",                 0, docred),
        Source("2_boolq",                  0, boolq),
        Source("3_squad_v2",               0, squad_v2),
        Source("4_platypus",               0, platypus),
        Source("5_wildjailbreak_benign",   0, wildjailbreak_benign),
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
    print(f"\n=== Deduplication: {before} -> {after} "
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

    print(f"\n  Saved -> {train_path}")
    print(f"  Saved -> {test_path}")
    print(f"  Saved -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()