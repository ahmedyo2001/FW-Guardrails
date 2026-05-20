"""
Approach 1 — Windowed PPL Threshold Detector
=============================================
Scoring:   GPT-2 perplexity with sliding window (1024 tokens, stride 512)
           following HuggingFace Transformers documentation.
Detection: single PPL threshold, tuned on training data to maximise F2 (β=2),
           then evaluated on held-out test data.

This is the simplest perplexity-based guardrail — no classifier, just one number.

Input
-----
  data/train.jsonl   produced by load_dataset.py
  data/test.jsonl    produced by load_dataset.py

Output
------
  results/approach1_results.json   threshold + full metrics
  results/approach1_predictions.jsonl  per-sample predictions on test set

Usage
-----
  python approach1_ppl_threshold.py
  python approach1_ppl_threshold.py --train data/train.jsonl --test data/test.jsonl
  python approach1_ppl_threshold.py --cache-dir ppl_cache   # reuse scored prompts
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, precision_recall_curve,
                             roc_auc_score)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"
MAX_TOKENS  = 1024          # GPT-2 context window
STRIDE      = 512           # half-window stride (HuggingFace recommendation)
BETA        = 2             # F-beta score β — penalises false negatives more
RANDOM_SEED = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── PPL scorer ────────────────────────────────────────────────────────────────

def load_gpt2():
    print(f"Loading GPT-2 on {DEVICE}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


def compute_ppl(text: str, tokenizer, model) -> float:
    """
    Windowed perplexity following HuggingFace Transformers documentation:
    https://huggingface.co/docs/transformers/en/perplexity

    Window = 1024 tokens (GPT-2 context limit)
    Stride = 512 tokens  (half-window overlap)

    Each token is scored with at least 512 tokens of left context,
    giving a closer approximation to the true autoregressive likelihood
    than non-overlapping chunking.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len   = input_ids.size(1)

    if seq_len == 0:
        return float("inf")

    nlls      = []
    prev_end  = 0

    for begin in range(0, seq_len, STRIDE):
        end        = min(begin + MAX_TOKENS, seq_len)
        target_len = end - prev_end      # tokens being scored this window

        # mask context tokens so only new tokens contribute to the loss
        target_ids              = input_ids[:, begin:end].clone()
        target_ids[:, :-target_len] = -100   # -100 is ignored by CrossEntropyLoss

        with torch.no_grad():
            loss = model(input_ids[:, begin:end],
                         labels=target_ids).loss
        nlls.append(loss * target_len)

        prev_end = end
        if end == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / seq_len).item()


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── PPL scoring with disk cache ───────────────────────────────────────────────

def score_records(records: list[dict], tokenizer, model,
                  cache_path: str | None = None) -> np.ndarray:
    """
    Returns a 1-D array of PPL values aligned with records.
    If cache_path is given, previously scored texts are reused.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached PPL scores from {cache_path}")

    ppls    = []
    n_new   = 0
    for i, r in enumerate(records):
        key = r["text"]
        if key in cache:
            ppls.append(cache[key])
        else:
            ppl = compute_ppl(r["text"], tokenizer, model)
            cache[key] = ppl
            ppls.append(ppl)
            n_new += 1

        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(records)}  (new: {n_new})")

    if cache_path and n_new > 0:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"  Cache saved → {cache_path}  ({len(cache)} total entries)")

    return np.array(ppls, dtype=np.float64)


# ── threshold tuning ──────────────────────────────────────────────────────────

def find_best_threshold(ppls: np.ndarray, labels: np.ndarray,
                        beta: float = BETA) -> tuple[float, float]:
    """
    Sweep 500 candidate thresholds across the PPL range of the training set.
    Pick the one that maximises F-beta score.
    Returns (best_threshold, best_f_beta).
    """
    candidates = np.linspace(ppls.min(), ppls.max(), 500)
    best_t, best_f = 0.0, 0.0

    for t in candidates:
        preds = (ppls >= t).astype(int)
        f     = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t

    return best_t, best_f


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(ppls: np.ndarray, labels: np.ndarray,
             threshold: float) -> dict:
    preds = (ppls >= threshold).astype(int)

    f2    = fbeta_score(labels, preds, beta=BETA,  zero_division=0)
    f1    = fbeta_score(labels, preds, beta=1,     zero_division=0)
    auc   = roc_auc_score(labels, ppls)   # PPL itself as the score
    cm    = confusion_matrix(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=["benign", "adversarial"],
                                   output_dict=True)

    tn, fp, fn, tp = cm.ravel()
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
    parser = argparse.ArgumentParser(
        description="Approach 1 — Windowed PPL threshold guardrail")
    parser.add_argument("--train",     default="data/train.jsonl")
    parser.add_argument("--test",      default="data/test.jsonl")
    parser.add_argument("--cache-dir", default="ppl_cache",
                        help="Directory to cache PPL scores (speeds up reruns)")
    parser.add_argument("--out-dir",   default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_records = load_jsonl(args.train)
    test_records  = load_jsonl(args.test)
    print(f"  Train: {len(train_records)}  |  Test: {len(test_records)}")

    train_labels = np.array([r["label"] for r in train_records])
    test_labels  = np.array([r["label"] for r in test_records])

    # ── score with GPT-2 windowed PPL ─────────────────────────────────────────
    tokenizer, model = load_gpt2()

    train_cache = os.path.join(args.cache_dir, "train_ppl.json")
    test_cache  = os.path.join(args.cache_dir, "test_ppl.json")

    print("\n=== Scoring training set ===")
    train_ppls = score_records(train_records, tokenizer, model, train_cache)

    print("\n=== Scoring test set ===")
    test_ppls  = score_records(test_records,  tokenizer, model, test_cache)

    # ── find threshold on training data ───────────────────────────────────────
    print("\n=== Tuning threshold on training set (maximising F2) ===")
    threshold, train_f2 = find_best_threshold(train_ppls, train_labels)
    print(f"  Best threshold : {threshold:.2f}")
    print(f"  Train F2       : {train_f2:.4f}")

    # ── evaluate on test set ──────────────────────────────────────────────────
    print("\n=== Evaluating on test set ===")
    metrics = evaluate(test_ppls, test_labels, threshold)

    print(f"\n  Threshold  : {metrics['threshold']:.2f}")
    print(f"  F2 (test)  : {metrics['f2']:.4f}")
    print(f"  F1 (test)  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")

    print("\n" + classification_report(
        test_labels,
        (test_ppls >= threshold).astype(int),
        target_names=["benign", "adversarial"]
    ))

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "approach":        "1_windowed_ppl_threshold",
        "window":          MAX_TOKENS,
        "stride":          STRIDE,
        "beta":            BETA,
        "train_size":      len(train_records),
        "test_size":       len(test_records),
        "train_f2":        float(train_f2),
        **metrics,
    }

    results_path = os.path.join(args.out_dir, "approach1_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {results_path}")

    # per-sample predictions
    preds_path = os.path.join(args.out_dir, "approach1_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r, ppl, pred in zip(test_records,
                                 test_ppls.tolist(),
                                 (test_ppls >= threshold).astype(int).tolist()):
            f.write(json.dumps({
                "text":       r["text"],
                "source":     r.get("source", ""),
                "label":      r["label"],
                "ppl":        round(ppl, 2),
                "predicted":  pred,
                "correct":    int(pred == r["label"]),
            }, ensure_ascii=False) + "\n")
    print(f"  Saved → {preds_path}")


if __name__ == "__main__":
    main()