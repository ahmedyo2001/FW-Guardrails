"""
Approach 2 — LightGBM Classifier (PPL + Token Length)
======================================================
Replicating Alon & Kamfonas (2023) - arXiv:2308.14132

Scoring:    LLaMA 3.2 1B perplexity on the full prompt in one forward pass.
            Token length is the full tokenized length before any truncation.
Features:   [ppl, token_length]  — exactly as in the paper
Classifier: LightGBM binary classifier
Threshold:  tuned on a validation split of training data to maximise F2 (β=2),
            then evaluated on held-out test data.

Input
-----
  data/train.jsonl   produced by load_dataset.py
  data/test.jsonl    produced by load_dataset.py

Output
------
  results/approach2_results.json
  results/approach2_predictions.jsonl
  models/approach2_lgbm.pkl

Usage
-----
  python approach2_lgbm.py
  python approach2_lgbm.py --train data/train.jsonl --test data/test.jsonl
"""

import argparse
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.2-1B"
BETA        = 2
RANDOM_SEED = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LGBM_PARAMS = {

    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.05,
    #min number of leafs
    "num_leaves":        31,
    #min numnber of samples in a leaf node
    "min_child_samples": 10,
    #supresses internal messages
    "verbose":           -1,
    "seed":              RANDOM_SEED,
}


# ── LLaMA model loader ────────────────────────────────────────────────────────

def load_llama():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else DEVICE}")
    return tokenizer, model

# computes perplexity like other code but without multiplying and dividing
def compute_ppl_and_length(text: str, tokenizer, model) -> tuple[float, int]:
    """
    Returns (perplexity, token_length).

    Single forward pass on the full prompt — no windowing, matching the paper.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len   = input_ids.size(1)

    if seq_len == 0:
        return float("inf"), 0

    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss

    ppl = torch.exp(loss).item()
    return ppl, seq_len


# ── data loading & feature extraction ────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# checks for cache, if not there computes 
def extract_features(records: list[dict], tokenizer, model,
                     cache_path: str | None = None) -> np.ndarray:
    """
    Returns (N, 2) feature matrix: [ppl, token_length] per record.
    Cache stores {text: [ppl, token_length]} to avoid rescoring on reruns.
    """
    cache = {}
    #loads already cached entries
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached entries from {cache_path}")

    rows  = []
    n_new = 0
    #computes new entries
    for i, r in enumerate(records):
        key = r["text"]
        if key in cache:
            rows.append(cache[key])
        else:
            ppl, length = compute_ppl_and_length(r["text"], tokenizer, model)
            cache[key]  = [ppl, length]
            rows.append([ppl, length])
            n_new += 1

        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(records)}  (new: {n_new})")
    #adds new entries to cache
    if cache_path and n_new > 0:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"  Cache saved ->{cache_path}")

    return np.array(rows, dtype=np.float32)


# ── threshold tuning ──────────────────────────────────────────────────────────
# finds best accept reject threshold 
def find_best_threshold(probs: np.ndarray, labels: np.ndarray,
                        beta: float = BETA) -> tuple[float, float]:
    """Sweep 200 thresholds on validation probs, maximise F-beta."""
    best_t, best_f = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        preds = (probs >= t).astype(int)
        f     = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ── evaluation ────────────────────────────────────────────────────────────────
# same eval
def evaluate(probs: np.ndarray, labels: np.ndarray,
             threshold: float) -> dict:
    preds  = (probs >= threshold).astype(int)
    f2     = fbeta_score(labels, preds, beta=BETA, zero_division=0)
    f1     = fbeta_score(labels, preds, beta=1,    zero_division=0)
    auc    = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")
    cm     = confusion_matrix(labels, preds)
    report = classification_report(labels, preds,
                                   labels=[0, 1],
                                   target_names=["benign", "adversarial"],
                                   output_dict=True, zero_division=0)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(preds.sum()))
    return {
        "threshold": float(threshold),
        "f2":        float(f2),
        "f1":        float(f1),
        "auc_roc":   float(auc),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "precision": report["adversarial"]["precision"],
        "recall":    report["adversarial"]["recall"],
        "report":    report,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    #arguments
    parser = argparse.ArgumentParser(
        description="Approach 2 — LightGBM on PPL + token length (full prompt)")
    parser.add_argument("--train",     default="v2/data/train.jsonl")
    parser.add_argument("--test",      default="v2/data/test.jsonl")
    parser.add_argument("--cache-dir", default="ppl_cache_approach2",
                        help="Cache directory for PPL scores (independent of approach 1)")
    parser.add_argument("--out-dir",   default="results")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_records = load_jsonl(args.train)
    test_records  = load_jsonl(args.test)
    print(f"  Train: {len(train_records)}  |  Test: {len(test_records)}")

    train_labels = np.array([r["label"] for r in train_records])
    test_labels  = np.array([r["label"] for r in test_records])

    # ── extract features ──────────────────────────────────────────────────────
    tokenizer, model = load_llama()

    train_cache = os.path.join(args.cache_dir, "train_features.json")
    test_cache  = os.path.join(args.cache_dir, "test_features.json")

    print("\n=== Extracting features — training set ===")
    X_train_full = extract_features(train_records, tokenizer, model, train_cache)

    print("\n=== Extracting features — test set ===")
    X_test = extract_features(test_records, tokenizer, model, test_cache)

    # ── split training into train / validation for threshold tuning ───────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, train_labels,
        test_size=0.2,
        stratify=train_labels,
        random_state=RANDOM_SEED,
    )
    print(f"\n  LightGBM train: {len(X_tr)}  |  Validation: {len(X_val)}")

    # ── train LightGBM ────────────────────────────────────────────────────────
    print("\n=== Training LightGBM ===")
    lgb_train = lgb.Dataset(X_tr,  label=y_tr,
                            feature_name=["ppl", "token_length"])
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    booster = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # ── tune threshold on validation set ──────────────────────────────────────
    print("\n=== Tuning threshold on validation set (maximising F2) ===")
    val_probs = booster.predict(X_val)
    threshold, val_f2 = find_best_threshold(val_probs, y_val)
    print(f"  Best threshold : {threshold:.3f}")
    print(f"  Validation F2  : {val_f2:.4f}")

    # ── evaluate on test set ──────────────────────────────────────────────────
    print("\n=== Evaluating on test set ===")
    test_probs = booster.predict(X_test)
    metrics    = evaluate(test_probs, test_labels, threshold)

    print(f"\n  Threshold  : {metrics['threshold']:.3f}")
    print(f"  F2 (test)  : {metrics['f2']:.4f}")
    print(f"  F1 (test)  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")

    print("\n" + classification_report(
        test_labels,
        (test_probs >= threshold).astype(int),
        labels=[0, 1],
        target_names=["benign", "adversarial"],
        zero_division=0
    ))

    # ── save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.model_dir, "approach2_lgbm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"booster": booster, "threshold": threshold}, f)
    print(f"\n  Model saved ->{model_path}")

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "approach":    "2_lgbm_ppl_token_length",
        "ppl_method":  "full_prompt_single_pass",
        "features":    ["ppl", "token_length"],
        "beta":        BETA,
        "train_size":  len(train_records),
        "test_size":   len(test_records),
        "val_f2":      float(val_f2),
        "lgbm_params": LGBM_PARAMS,
        **metrics,
    }

    results_path = os.path.join(args.out_dir, "approach2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved ->{results_path}")

    # per-sample predictions
    preds_path = os.path.join(args.out_dir, "approach2_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r, feats, prob, pred in zip(
            test_records,
            X_test.tolist(),
            test_probs.tolist(),
            (test_probs >= threshold).astype(int).tolist(),
        ):
            f.write(json.dumps({
                "text":         r["text"],
                "source":       r.get("source", ""),
                "label":        r["label"],
                "ppl":          round(feats[0], 2),
                "token_length": int(feats[1]),
                "probability":  round(prob, 4),
                "predicted":    pred,
                "correct":      int(pred == r["label"]),
            }, ensure_ascii=False) + "\n")
    print(f"  Predictions saved ->{preds_path}")


if __name__ == "__main__":
    main()