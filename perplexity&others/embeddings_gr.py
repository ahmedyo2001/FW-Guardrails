"""
Approach 3 — Arize Dataset Embeddings Guardrail
================================================
Based on: https://guardrailsai.com/hub/validator/arize-ai/dataset_embeddings_guardrails
GitHub:   https://github.com/Arize-ai/dataset-embeddings-guardrails

Mechanism:
  1. Embed all adversarial prompts from the training set (label=1) → source library
  2. For each test prompt, compute cosine distance to every source embedding
  3. Take the minimum cosine distance as the similarity score
  4. If min_distance < threshold → flag as adversarial (1), else benign (0)

Threshold is tuned on a validation split of the training set to maximise F2 (β=2).

Embeddings use sentence-transformers/all-MiniLM-L6-v2 — free, local, no API key.

Input
-----
  data/train.jsonl   produced by load_dataset.py
  data/test.jsonl    produced by load_dataset.py

Output
------
  results/approach3_results.json
  results/approach3_predictions.jsonl
  models/approach3_source_embeddings.pkl   (cached source embeddings)

Usage
-----
  pip install sentence-transformers scikit-learn numpy
  python approach3_embeddings.py
  python approach3_embeddings.py --threshold 0.2   # skip tuning, use Arize default
"""

import argparse
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# ── config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, 80MB
BETA          = 2
RANDOM_SEED   = 42
CHUNK_SIZE    = 30       # characters per chunk (Arize default)
CHUNK_OVERLAP = 5        # overlap between chunks (Arize default)
BATCH_SIZE    = 256      # sentence-transformers batch size


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Mirrors the default chunking strategy in ArizeDatasetEmbeddings.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks if chunks else [text]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity."""
    return 1.0 - float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    )


# ── source library builder ────────────────────────────────────────────────────

def build_source_library(adv_texts: list[str], embedder: SentenceTransformer,
                         cache_path: str | None = None) -> np.ndarray:
    """
    Chunk all adversarial training prompts, embed each chunk,
    and return an (N_chunks, embed_dim) matrix — the source library.
    Mirrors what ArizeDatasetEmbeddings does at construction time.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            lib = pickle.load(f)
        print(f"  Loaded source library ({lib.shape[0]} chunks) from {cache_path}")
        return lib

    print(f"  Chunking {len(adv_texts)} adversarial prompts...")
    all_chunks = []
    for text in adv_texts:
        all_chunks.extend(chunk_text(text))
    print(f"  Total chunks: {len(all_chunks)}")

    print("  Embedding chunks...")
    embeddings = embedder.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → dot product = cosine sim
    )
    lib = embeddings.astype(np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(lib, f)
        print(f"  Source library saved → {cache_path}")

    return lib


# ── scoring ───────────────────────────────────────────────────────────────────

def score_prompts(records: list[dict], source_lib: np.ndarray,
                  embedder: SentenceTransformer,
                  cache_path: str | None = None) -> np.ndarray:
    """
    For each prompt, embed it and compute minimum cosine distance
    to any chunk in the source library.

    Lower score = more similar to known jailbreaks = more suspicious.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached scores from {cache_path}")

    # separate cached from uncached
    min_distances    = [None] * len(records)
    texts_to_embed   = []
    indices_to_embed = []

    for i, r in enumerate(records):
        if r["text"] in cache:
            min_distances[i] = cache[r["text"]]
        else:
            texts_to_embed.append(r["text"])
            indices_to_embed.append(i)

    # batch embed all uncached prompts
    if texts_to_embed:
        print(f"  Embedding {len(texts_to_embed)} prompts...")
        new_embeddings = embedder.encode(
            texts_to_embed,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
            # since embeddings are normalized, cosine distance = 1 - dot product
            sims     = source_lib @ emb          # (N_chunks,)
            min_dist = float(1.0 - sims.max())   # max similarity → min distance
            cache[text]          = min_dist
            min_distances[idx]   = min_dist

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"  Score cache saved → {cache_path}")

    return np.array(min_distances, dtype=np.float64)


# ── threshold tuning ──────────────────────────────────────────────────────────

def find_best_threshold(min_distances: np.ndarray, labels: np.ndarray,
                        beta: float = BETA) -> tuple[float, float]:
    """
    Sweep 500 thresholds on min cosine distance.
    Predict adversarial (1) when min_distance < threshold.
    Maximise F-beta.
    """
    candidates = np.linspace(min_distances.min(), min_distances.max(), 500)
    best_t, best_f = 0.2, 0.0

    for t in candidates:
        preds = (min_distances < t).astype(int)
        f     = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t

    return best_t, best_f


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(min_distances: np.ndarray, labels: np.ndarray,
             threshold: float) -> dict:
    preds  = (min_distances < threshold).astype(int)
    f2     = fbeta_score(labels, preds, beta=BETA, zero_division=0)
    f1     = fbeta_score(labels, preds, beta=1,    zero_division=0)
    scores = 1.0 - min_distances    # similarity — higher = more adversarial
    auc    = roc_auc_score(labels, scores)
    cm     = confusion_matrix(labels, preds)
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
        description="Approach 3 — Arize Dataset Embeddings guardrail (local embeddings)")
    parser.add_argument("--train",     default="data/train.jsonl")
    parser.add_argument("--test",      default="data/test.jsonl")
    parser.add_argument("--cache-dir", default="embed_cache_approach3")
    parser.add_argument("--out-dir",   default="results")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fix cosine distance threshold instead of tuning. "
                             "Arize default is 0.2.")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── load embedding model ──────────────────────────────────────────────────
    print(f"\n=== Loading embedding model: {EMBED_MODEL} ===")
    embedder = SentenceTransformer(EMBED_MODEL)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_records = load_jsonl(args.train)
    test_records  = load_jsonl(args.test)
    print(f"  Train: {len(train_records)}  |  Test: {len(test_records)}")

    train_labels = np.array([r["label"] for r in train_records])
    test_labels  = np.array([r["label"] for r in test_records])

    # ── build source library from adversarial training prompts ────────────────
    print("\n=== Building source library (adversarial training prompts) ===")
    adv_train_texts = [r["text"] for r in train_records if r["label"] == 1]
    print(f"  Adversarial training prompts: {len(adv_train_texts)}")

    lib_cache  = os.path.join(args.model_dir, "approach3_source_embeddings.pkl")
    source_lib = build_source_library(adv_train_texts, embedder, lib_cache)
    print(f"  Source library shape: {source_lib.shape}")

    # ── score training set ────────────────────────────────────────────────────
    print("\n=== Scoring training set ===")
    train_cache     = os.path.join(args.cache_dir, "train_scores.json")
    train_distances = score_prompts(train_records, source_lib, embedder, train_cache)

    # ── tune threshold on validation split of training set ────────────────────
    if args.threshold is not None:
        threshold = args.threshold
        val_f2    = None
        print(f"\n=== Using fixed threshold: {threshold} (Arize default=0.2) ===")
    else:
        print("\n=== Tuning threshold on validation split of training set ===")
        indices   = np.arange(len(train_records))
        tr_idx, val_idx = train_test_split(
            indices, test_size=0.2,
            stratify=train_labels,
            random_state=RANDOM_SEED,
        )
        threshold, val_f2 = find_best_threshold(
            train_distances[val_idx], train_labels[val_idx]
        )
        print(f"  Best threshold : {threshold:.4f}")
        print(f"  Validation F2  : {val_f2:.4f}")

    # ── score and evaluate test set ───────────────────────────────────────────
    print("\n=== Scoring test set ===")
    test_cache     = os.path.join(args.cache_dir, "test_scores.json")
    test_distances = score_prompts(test_records, source_lib, embedder, test_cache)

    print("\n=== Evaluating on test set ===")
    metrics = evaluate(test_distances, test_labels, threshold)

    print(f"\n  Threshold  : {metrics['threshold']:.4f}  "
          f"(lower distance = more adversarial)")
    print(f"  F2 (test)  : {metrics['f2']:.4f}")
    print(f"  F1 (test)  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")

    print("\n" + classification_report(
        test_labels,
        (test_distances < threshold).astype(int),
        target_names=["benign", "adversarial"]
    ))

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "approach":        "3_arize_dataset_embeddings",
        "embed_model":     EMBED_MODEL,
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "n_source_chunks": int(source_lib.shape[0]),
        "n_adv_sources":   len(adv_train_texts),
        "beta":            BETA,
        "train_size":      len(train_records),
        "test_size":       len(test_records),
        "val_f2":          float(val_f2) if val_f2 is not None else None,
        "threshold_tuned": args.threshold is None,
        **metrics,
    }

    results_path = os.path.join(args.out_dir, "approach3_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    preds_path = os.path.join(args.out_dir, "approach3_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r, dist, pred in zip(
            test_records,
            test_distances.tolist(),
            (test_distances < threshold).astype(int).tolist(),
        ):
            f.write(json.dumps({
                "text":         r["text"],
                "source":       r.get("source", ""),
                "label":        r["label"],
                "min_distance": round(dist, 4),
                "predicted":    pred,
                "correct":      int(pred == r["label"]),
            }, ensure_ascii=False) + "\n")
    print(f"  Predictions saved → {preds_path}")


if __name__ == "__main__":
    main()