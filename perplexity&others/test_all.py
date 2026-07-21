"""
Test All — inference-only evaluation of the saved v2 guardrail models
=======================================================================
Loads the artifacts already produced by train_all.py (no retraining, no
re-tuning of thresholds) and scores every *.jsonl file under the attack
test data directory (produced by prep_attack_test_data.py) against each
of the 5 model variants:

  approach1_ppl_threshold     v2/results/approach1_results.json          (threshold)
  approach2_lgbm              v2/models/approach2_lgbm.pkl               (booster + threshold)
  approach3_embed_char        v2/models/approach3_source_embeddings_char{N}.pkl
                               v2/results/approach3_results_char.json     (threshold)
  approach3_embed_sentence    v2/models/approach3_source_embeddings_sentence.pkl
                               v2/results/approach3_results_sentence.json (threshold)
  approach3_embed_full        v2/models/approach3_source_embeddings_full.pkl
                               v2/results/approach3_results_full.json     (threshold)

A variant is silently skipped (with a warning) if its artifacts aren't
found, so you can test a subset without having trained everything.

Output
------
  <out-dir>/summary.json                                  one row per (approach, test file)
  <out-dir>/summary.csv                                   same, as a table
  <out-dir>/predictions/<approach>/<test_file>.jsonl       per-sample predictions

Metrics reported per (approach, test file):
  detection_rate       = recall on the positive class — the meaningful number
                          for attack-only test files (manyshot, jigsaw, ...)
  false_positive_rate   = FP / (FP + TN) — the meaningful number for
                          benign-only test files
  precision / f1 / accuracy — reported when both classes are present

Setup
-----
  1. Train once:
       python train_all.py --train v2/training_data/train.jsonl \\
                            --test  v2/training_data/test.jsonl \\
                            --out-dir v2/results --model-dir v2/models
  2. Build the attack test files:
       python prep_attack_test_data.py
  3. Score everything against the trained models:
       python test_all.py

Usage
-----
  python test_all.py
  python test_all.py --v2-dir v2 --test-dir v2/attack_test_data --out-dir v2/test_results
  python test_all.py --skip 3char 3full     # only approach1, approach2, approach3-sentence
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

import thresholding_approach as a1
import LightGBM_approach as a2
import embeddings_gr as a3

ALL_VARIANTS = ["1", "2", "3char", "3sentence", "3full"]

APPROACH_NAMES = {
    "1":         "approach1_ppl_threshold",
    "2":         "approach2_lgbm",
    "3char":     "approach3_embed_char",
    "3sentence": "approach3_embed_sentence",
    "3full":     "approach3_embed_full",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_test_files(test_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(test_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {test_dir}")
    return paths


# ── artifact loaders ──────────────────────────────────────────────────────────

def load_approach1(v2_dir: str):
    path = os.path.join(v2_dir, "results", "approach1_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return {"threshold": results["threshold"]}


def load_approach2(v2_dir: str):
    path = os.path.join(v2_dir, "models", "approach2_lgbm.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return {"booster": obj["booster"], "threshold": obj["threshold"]}


def load_approach3(v2_dir: str, chunk_strategy: str, chunk_size: int):
    chunk_tag = f"char{chunk_size}" if chunk_strategy == "char" else chunk_strategy
    lib_path  = os.path.join(v2_dir, "models",  f"approach3_source_embeddings_{chunk_tag}.pkl")
    res_path  = os.path.join(v2_dir, "results", f"approach3_results_{chunk_strategy}.json")
    if not (os.path.exists(lib_path) and os.path.exists(res_path)):
        return None
    with open(lib_path, "rb") as f:
        source_lib = pickle.load(f)
    with open(res_path, encoding="utf-8") as f:
        results = json.load(f)
    return {"source_lib": source_lib, "threshold": results["threshold"]}


# ── scoring per approach (reuses the scoring/caching logic already in each
#    approach module — only the trained artifact + threshold come from disk) ──

def score_approach1(records, artifact, tokenizer, model, cache_path):
    ppls  = a1.score_records(records, tokenizer, model, cache_path)
    preds = (ppls >= artifact["threshold"]).astype(int)
    return ppls, preds


def score_approach2(records, artifact, tokenizer, model, cache_path):
    feats = a2.extract_features(records, tokenizer, model, cache_path)
    probs = artifact["booster"].predict(feats)
    preds = (probs >= artifact["threshold"]).astype(int)
    return probs, preds


def score_approach3(records, artifact, embedder, cache_path):
    dists = a3.score_prompts(records, artifact["source_lib"], embedder, cache_path)
    preds = (dists < artifact["threshold"]).astype(int)
    return dists, preds


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds) -> dict:
    labels = np.asarray(labels)
    preds  = np.asarray(preds)

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))

    n_pos, n_neg = tp + fn, tn + fp
    recall    = tp / n_pos if n_pos else None
    fpr       = fp / n_neg if n_neg else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1        = (2 * precision * recall / (precision + recall)
                 if precision and recall else None)
    accuracy  = (tp + tn) / len(labels) if len(labels) else None

    return {
        "n": len(labels), "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "detection_rate":      recall,
        "false_positive_rate": fpr,
        "precision":           precision,
        "f1":                  f1,
        "accuracy":            accuracy,
    }


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score saved v2 guardrail models against all attack/OOD/benign test files")
    parser.add_argument("--v2-dir",   default="v2")
    parser.add_argument("--test-dir", default=None,
                        help="Default: <v2-dir>/attack_test_data")
    parser.add_argument("--out-dir",  default=None,
                        help="Default: <v2-dir>/test_results")
    parser.add_argument("--a1-cache-dir", default="ppl_cache")
    parser.add_argument("--a2-cache-dir", default="ppl_cache_approach2")
    parser.add_argument("--a3-cache-dir", default="embed_cache_approach3")
    parser.add_argument("--a3-chunk-size", type=int, default=30,
                        help="Must match the chunk size used for the 'char' "
                             "variant in train_all.py (default: 30)")
    parser.add_argument("--skip", nargs="*", default=[], choices=ALL_VARIANTS,
                        metavar="VARIANT",
                        help="Variants to skip, e.g. --skip 3char 3full")
    args = parser.parse_args()

    test_dir  = args.test_dir or os.path.join(args.v2_dir, "attack_test_data")
    out_dir   = args.out_dir  or os.path.join(args.v2_dir, "test_results")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    skip = set(args.skip)

    test_files = discover_test_files(test_dir)
    print(f"Found {len(test_files)} test files in {test_dir}")

    # ── load whichever artifacts exist ────────────────────────────────────────
    artifacts = {}
    if "1" not in skip:
        artifacts["1"] = load_approach1(args.v2_dir)
        if artifacts["1"] is None:
            print("  [skip] approach1: no results/approach1_results.json found")
    if "2" not in skip:
        artifacts["2"] = load_approach2(args.v2_dir)
        if artifacts["2"] is None:
            print("  [skip] approach2: no models/approach2_lgbm.pkl found")
    for strat in ["char", "sentence", "full"]:
        key = f"3{strat}"
        if key not in skip:
            artifacts[key] = load_approach3(args.v2_dir, strat, args.a3_chunk_size)
            if artifacts[key] is None:
                print(f"  [skip] {key}: missing source-embeddings/results artifact")

    active = {k: v for k, v in artifacts.items() if v is not None}
    if not active:
        raise RuntimeError(f"No usable model artifacts found under {args.v2_dir}")
    print(f"Active variants: {', '.join(APPROACH_NAMES[k] for k in active)}")

    # ── load shared models once ───────────────────────────────────────────────
    tokenizer = model = embedder = None
    if "1" in active or "2" in active:
        print("\n=== Loading LLaMA 3.2 1B ===")
        tokenizer, model = a1.load_llama()
    if any(k.startswith("3") for k in active):
        print("\n=== Loading sentence-transformers embedder ===")
        embedder = SentenceTransformer(a3.EMBED_MODEL)

    cache_paths = {
        "1":         os.path.join(args.a1_cache_dir, "attack_tests_ppl.json"),
        "2":         os.path.join(args.a2_cache_dir, "attack_tests_features.json"),
        "3char":     os.path.join(args.a3_cache_dir, "attack_tests_char_scores.json"),
        "3sentence": os.path.join(args.a3_cache_dir, "attack_tests_sentence_scores.json"),
        "3full":     os.path.join(args.a3_cache_dir, "attack_tests_full_scores.json"),
    }
    score_fields = {
        "1": "ppl", "2": "probability",
        "3char": "min_distance", "3sentence": "min_distance", "3full": "min_distance",
    }

    # ── score every test file against every active variant ───────────────────
    summary_rows = []

    for test_path in test_files:
        test_name = os.path.splitext(os.path.basename(test_path))[0]
        records   = load_jsonl(test_path)
        labels    = [r["label"] for r in records]
        print(f"\n--- {test_name} ({len(records)} records) ---")

        for key, artifact in active.items():
            approach_name = APPROACH_NAMES[key]
            cache_path    = cache_paths[key]

            if key == "1":
                scores, preds = score_approach1(records, artifact, tokenizer, model, cache_path)
            elif key == "2":
                scores, preds = score_approach2(records, artifact, tokenizer, model, cache_path)
            else:
                scores, preds = score_approach3(records, artifact, embedder, cache_path)

            metrics = compute_metrics(labels, preds)
            metrics.update({
                "test_file": test_name,
                "approach":  approach_name,
                "threshold": artifact["threshold"],
            })
            summary_rows.append(metrics)

            print(f"  {approach_name:28s}  detect={fmt(metrics['detection_rate'])}  "
                  f"fpr={fmt(metrics['false_positive_rate'])}  acc={fmt(metrics['accuracy'])}")

            # per-sample predictions
            approach_pred_dir = os.path.join(preds_dir, approach_name)
            os.makedirs(approach_pred_dir, exist_ok=True)
            pred_path = os.path.join(approach_pred_dir, f"{test_name}.jsonl")
            score_field = score_fields[key]
            with open(pred_path, "w", encoding="utf-8") as f:
                for r, score, pred in zip(records, scores.tolist(), preds.tolist()):
                    f.write(json.dumps({
                        "text":      r["text"],
                        "source":    r.get("source", ""),
                        "label":     r["label"],
                        score_field: round(float(score), 4),
                        "predicted": int(pred),
                        "correct":   int(pred == r["label"]),
                    }, ensure_ascii=False) + "\n")

    # ── write aggregated summary ──────────────────────────────────────────────
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = ["test_file", "approach", "threshold", "n", "n_pos", "n_neg",
                  "tp", "fp", "tn", "fn", "detection_rate", "false_positive_rate",
                  "precision", "f1", "accuracy"]
    summary_csv_path = os.path.join(out_dir, "summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"\nSummary written    -> {summary_json_path}")
    print(f"Summary written    -> {summary_csv_path}")
    print(f"Predictions written -> {preds_dir}")


if __name__ == "__main__":
    main()
