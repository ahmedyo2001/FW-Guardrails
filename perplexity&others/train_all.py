"""
Runs all 5 steps sequentially and prints a timing summary.

  Step 1 — Approach 1: Windowed PPL Threshold
  Step 2 — Approach 2: LightGBM on PPL + Length
  Step 3 — Approach 3: Embedding Distance (char chunks, 30 chars)
  Step 4 — Approach 3: Embedding Distance (sentence chunks)
  Step 5 — Approach 3: Embedding Distance (full prompt)

Shared args (apply to all steps):
  --train           path to train JSONL         (default: v2/training_data/train.jsonl)
  --test            path to test JSONL          (default: v2/training_data/test.jsonl)
  --out-dir         results output directory    (default: results)
  --model-dir       saved models directory      (default: models)

Approach 1 tuning:
  --a1-cache-dir    PPL cache directory         (default: ppl_cache)

Approach 2 tuning:
  --a2-cache-dir    PPL cache directory         (default: ppl_cache_approach2)

Approach 3 tuning (applies to both step 3 and 4):
  --a3-cache-dir    embedding cache directory   (default: embed_cache_approach3)
  --a3-threshold    fix cosine threshold instead of tuning (e.g. 0.25)
  --a3-chunk-size   character chunk size for step 3   (default: 50)
  --a3-chunk-overlap character chunk overlap for step 3 (default: 5)

Runner:
  --skip N [N ...]  skip step numbers, e.g. --skip 3 4

Examples:
  python run_all.py
  python run_all.py --skip 1 2
  python run_all.py --a3-chunk-size 30 --a3-threshold 0.25
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_argv(pairs: list[tuple[str, object]]) -> list[str]:
    argv = []
    for flag, value in pairs:
        if value is not None:
            argv += [flag, str(value)]
    return argv


def run_step(step: int, name: str, script: str, extra_argv: list[str]) -> tuple[bool, float]:
    print(f"\n{'='*70}")
    print(f"  STEP {step}: {name}")
    if extra_argv:
        print(f"  ARGS: {' '.join(extra_argv)}")
    print(f"{'='*70}\n")

    start  = time.time()
    result = subprocess.run(
        [sys.executable, str(HERE / script)] + extra_argv,
        cwd=HERE,
    )
    elapsed = time.time() - start

    ok     = result.returncode == 0
    status = "DONE" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] Step {step}  —  {elapsed/60:.1f} min ({elapsed:.0f}s)")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run all 5 steps (approaches 1, 2, 3×char, 3×sentence, 3×full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--train",     default=None, metavar="PATH")
    parser.add_argument("--test",      default=None, metavar="PATH")
    parser.add_argument("--out-dir",   default=None, metavar="DIR")
    parser.add_argument("--model-dir", default=None, metavar="DIR")

    # ── approach 1 ────────────────────────────────────────────────────────────
    parser.add_argument("--a1-cache-dir", default=None, metavar="DIR")

    # ── approach 2 ────────────────────────────────────────────────────────────
    parser.add_argument("--a2-cache-dir", default=None, metavar="DIR")

    # ── approach 3 (shared between step 3 and 4) ─────────────────────────────
    parser.add_argument("--a3-cache-dir",     default=None, metavar="DIR")
    parser.add_argument("--a3-threshold",     default=None, type=float, metavar="F")
    parser.add_argument("--a3-chunk-size",    default=30,   type=int,   metavar="N",
                        help="Char chunk size for step 3 (default: 30)")
    parser.add_argument("--a3-chunk-overlap", default=None, type=int,   metavar="N",
                        help="Char chunk overlap for step 3 (default: 5)")

    # ── runner ────────────────────────────────────────────────────────────────
    parser.add_argument("--skip", nargs="*", type=int, default=[], metavar="N",
                        help="Step numbers to skip, e.g. --skip 3 4")

    args = parser.parse_args()
    skip = set(args.skip or [])

    shared = build_argv([
        ("--train",   args.train),
        ("--test",    args.test),
        ("--out-dir", args.out_dir),
    ])

    steps = [
        (
            1,
            "Approach 1 — Windowed PPL Threshold",
            "thresholding_approach.py",
            shared + build_argv([("--cache-dir", args.a1_cache_dir)]),
        ),
        (
            2,
            "Approach 2 — LightGBM on PPL + Length",
            "LightGBM_approach.py",
            shared + build_argv([
                ("--cache-dir", args.a2_cache_dir),
                ("--model-dir", args.model_dir),
            ]),
        ),
        (
            3,
            f"Approach 3 — Embedding Distance (char, {args.a3_chunk_size} chars)",
            "embeddings_gr.py",
            shared + build_argv([
                ("--cache-dir",      args.a3_cache_dir),
                ("--model-dir",      args.model_dir),
                ("--threshold",      args.a3_threshold),
                ("--chunk-strategy", "char"),
                ("--chunk-size",     args.a3_chunk_size),
                ("--chunk-overlap",  args.a3_chunk_overlap),
            ]),
        ),
        (
            4,
            "Approach 3 — Embedding Distance (sentence chunks)",
            "embeddings_gr.py",
            shared + build_argv([
                ("--cache-dir",      args.a3_cache_dir),
                ("--model-dir",      args.model_dir),
                ("--threshold",      args.a3_threshold),
                ("--chunk-strategy", "sentence"),
            ]),
        ),
        (
            5,
            "Approach 3 — Embedding Distance (full prompt)",
            "embeddings_gr.py",
            shared + build_argv([
                ("--cache-dir",      args.a3_cache_dir),
                ("--model-dir",      args.model_dir),
                ("--threshold",      args.a3_threshold),
                ("--chunk-strategy", "full"),
            ]),
        ),
    ]

    summary    = []
    wall_start = time.time()

    for step, name, script, argv in steps:
        if step in skip:
            print(f"\n  [SKIPPED] Step {step} — {name}")
            summary.append((step, name, None, None))
            continue

        ok, elapsed = run_step(step, name, script, argv)
        summary.append((step, name, ok, elapsed))

    total = time.time() - wall_start

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for step, name, ok, elapsed in summary:
        if ok is None:
            print(f"  SKIPPED  Step {step}: {name}")
        elif ok:
            print(f"  OK       Step {step}: {name}  —  {elapsed/60:.1f} min")
        else:
            print(f"  FAILED   Step {step}: {name}  —  {elapsed/60:.1f} min")
    print(f"\n  Total wall time: {total/60:.1f} min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
