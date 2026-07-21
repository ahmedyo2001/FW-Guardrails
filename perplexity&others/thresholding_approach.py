"""
Approach 1 — Windowed PPL Threshold Detector
=============================================
Scoring:   LLaMA 3.2 1B perplexity with sliding window (2048 tokens, stride 1024)
Detection: single PPL threshold, tuned on training data to maximise F2 (β=2),
           then evaluated on held-out test data.

This is the simplest perplexity-based guardrail — no classifier, just one number.



Windowed perplexity following the HuggingFace Transformers documentation:
https://huggingface.co/docs/transformers/en/perplexity

Paper:BASELINE DEFENSES FOR ADVERSARIAL ATTACKS
AGAINST ALIGNED LANGUAGE MODELS



Input
-----
  data/train.jsonl   produced by load_dataset.py
  data/test.jsonl    produced by load_dataset.py

Output
------
  results/approach1_results.json           threshold + full metrics
  results/approach1_predictions.jsonl      per-sample predictions on test set

Usage
-----
  pip install transformers torch scikit-learn numpy accelerate

  # requires HuggingFace login for LLaMA access:
  huggingface-cli login

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
                             fbeta_score, roc_auc_score)
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.2-1B"
MAX_TOKENS  = 2048          # LLaMA 3.2 supports up to 128k, but 2048 is enough
STRIDE      = 1024          # half-window stride
BETA        = 1            # F1 score — equal weight to precision and recall
RANDOM_SEED = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── model loader ──────────────────────────────────────────────────────────────

def load_llama():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    print("  (first run will download ~2.5 GB — this may take a few minutes)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   # half precision — saves ~50% VRAM
        device_map="auto",           # auto places layers on GPU/CPU as available
    )
    model.eval()

    # LLaMA tokenizer has no default pad token — set it to eos to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else DEVICE}")
    return tokenizer, model


# ── PPL scorer ────────────────────────────────────────────────────────────────

# computes avg perplexity accross the prompt
def compute_ppl(text: str, tokenizer, model) -> float:
    """
    Windowed perplexity following the HuggingFace Transformers documentation:
    https://huggingface.co/docs/transformers/en/perplexity

    Window = 2048 tokens
    Stride = 1024 tokens (half-window overlap)

    Each token is scored with at least 1024 tokens of left context,
    giving a close approximation to the true autoregressive likelihood.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len   = input_ids.size(1)

    if seq_len == 0:
        return float("inf")

    nlls     = []
    prev_end = 0
    
    for begin in range(0, seq_len, STRIDE):
        # end of window considered
        end        = min(begin + MAX_TOKENS, seq_len)
        # length of scoring window, at first it is 2048 then 1024
        target_len = end - prev_end      

        # 2d array due to batch dim, clones beginning to end 
        target_ids                      = input_ids[:, begin:end].clone()
        # mask context tokens — -100 is ignored by CrossEntropyLoss 
        # makes the tokens considered before ignored in new run
        target_ids[:, :-target_len]     = -100


        # the model calculates the loss on the input with the output as the ground truth
        # any tokens with target id = -100 is ignored
        with torch.no_grad():
            loss = model(
                input_ids[:, begin:end],
                labels=target_ids
            ).loss
        # the part above outputs the avg loss so we multiply by the target len to get the loss
        nlls.append(loss * target_len)
        prev_end = end

        if end == seq_len:
            break
    # here we sum the NLL then divide by seq length to get the avg then we take exp to get the perplexity
    
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len).item()
    return ppl


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── PPL scoring with disk cache ───────────────────────────────────────────────
# computes perplexity for prompts and caches them
def score_records(records: list[dict], tokenizer, model,
                  cache_path: str | None = None) -> np.ndarray:
    """
    Returns a 1-D float array of PPL values aligned with records.
    Caches results to disk so reruns skip already-scored prompts.
    """
    #load ppl if it was already cached
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached PPL scores from {cache_path}")

    ppls  = []
    n_new = 0

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
        print(f"  Cache saved -> {cache_path}  ({len(cache)} total entries)")

    return np.array(ppls, dtype=np.float64)


# ── threshold tuning ──────────────────────────────────────────────────────────
# tunes threshold on full data since there is no training or DB here
def find_best_threshold(ppls: np.ndarray, labels: np.ndarray,
                        beta: float = BETA) -> tuple[float, float]:
    """
    Sweep 500 candidate thresholds across the PPL range of the training set.
    Pick the one that maximises F-beta score (default β=2).
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
#TODO: check eval part
def evaluate(ppls: np.ndarray, labels: np.ndarray,
             threshold: float) -> dict:
    preds  = (ppls >= threshold).astype(int)
    f2     = fbeta_score(labels, preds, beta=BETA, zero_division=0)
    f1     = fbeta_score(labels, preds, beta=1,    zero_division=0)
    auc    = roc_auc_score(labels, ppls) if len(set(labels.tolist())) > 1 else float("nan")
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
    parser = argparse.ArgumentParser(
        description="Approach 1 — Windowed PPL threshold guardrail (LLaMA 3.2 1B)")
    parser.add_argument("--train",     default="v2/data/train.jsonl")
    parser.add_argument("--test",      default="v2/data/test.jsonl")
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

    # ── load LLaMA 3.2 1B ────────────────────────────────────────────────────
    tokenizer, model = load_llama()

    train_cache = os.path.join(args.cache_dir, "train_ppl.json")
    test_cache  = os.path.join(args.cache_dir, "test_ppl.json")

    # ── score with windowed PPL ───────────────────────────────────────────────
    print("\n=== Scoring training set ===")
    train_ppls = score_records(train_records, tokenizer, model, train_cache)

    print("\n=== Scoring test set ===")
    test_ppls  = score_records(test_records,  tokenizer, model, test_cache)

    # ── tune threshold on training data ───────────────────────────────────────
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
        labels=[0, 1],
        target_names=["benign", "adversarial"],
        zero_division=0
    ))

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "approach":   "1_windowed_ppl_threshold",
        "model":      MODEL_NAME,
        "window":     MAX_TOKENS,
        "stride":     STRIDE,
        "beta":       BETA,
        "train_size": len(train_records),
        "test_size":  len(test_records),
        "train_f2":   float(train_f2),
        **metrics,
    }

    results_path = os.path.join(args.out_dir, "approach1_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {results_path}")

    # ── per-sample predictions ────────────────────────────────────────────────
    preds_path = os.path.join(args.out_dir, "approach1_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r, ppl, pred in zip(
            test_records,
            test_ppls.tolist(),
            (test_ppls >= threshold).astype(int).tolist()
        ):
            f.write(json.dumps({
                "text":      r["text"],
                "source":    r.get("source", ""),
                "label":     r["label"],
                "ppl":       round(ppl, 2),
                "predicted": pred,
                "correct":   int(pred == r["label"]),
            }, ensure_ascii=False) + "\n")
    print(f"  Saved -> {preds_path}")


if __name__ == "__main__":
    main()