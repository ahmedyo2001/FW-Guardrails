#!/usr/bin/env python3
"""
Deduplicate Apr 12 jackhhao detail files.

Each of the three Apr 12 files has 527 entries per token count, but only
517 unique objectives — 10 objectives appear twice. This script removes the
duplicate entries (keeping the first occurrence) and overwrites each file
in-place, leaving exactly 517 entries per token count (1551 total).

Similarity threshold: 0.99 via difflib.SequenceMatcher.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results_jackhhao"
TARGET_FILES = [
    "manyshot_detail_ibm-granite_granite-guardian-3.3-8b_1776034221.json",
    "manyshot_detail_ibm-granite_granite-guardian-3.2-3b-a800m_1776036754.json",
    "manyshot_detail_google_shieldgemma-9b_1776047942.json",
]
EXPECTED_PER_TC = 517


def dedup_group(entries: list) -> tuple[list, list]:
    """Return (kept, removed) by exact objective string deduplication."""
    seen = set()
    kept, removed = [], []
    for entry in entries:
        obj = entry["objective"]
        if obj in seen:
            removed.append(entry)
        else:
            seen.add(obj)
            kept.append(entry)
    return kept, removed


def process_file(fpath: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {fpath.name}")

    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    token_counts = sorted(set(d["filler_tokens"] for d in data))
    print(f"  Token counts: {token_counts}")
    print(f"  Total entries before: {len(data)}")

    all_kept = []
    all_removed = []

    for tc in token_counts:
        group = [d for d in data if d["filler_tokens"] == tc]
        kept, removed = dedup_group(group)
        all_kept.extend(kept)
        all_removed.extend(removed)

        if removed:
            print(f"  TC={tc}: {len(group)} -> {len(kept)} entries ({len(removed)} removed)")
        else:
            print(f"  TC={tc}: {len(group)} entries — no duplicates")

    # Validate
    for tc in token_counts:
        count = sum(1 for d in all_kept if d["filler_tokens"] == tc)
        assert count == EXPECTED_PER_TC, (
            f"Expected {EXPECTED_PER_TC} entries for TC={tc}, got {count}"
        )

    print(f"  Total entries after:  {len(all_kept)}")

    if all_removed:
        print(f"\n  Removed objectives ({len(all_removed) // len(token_counts)} unique):")
        # Show unique removed objectives (deduplicated across TCs)
        seen = set()
        for entry in all_removed:
            obj_preview = entry["objective"][:80].replace("\n", " ").replace("\r", "")
            if obj_preview not in seen:
                print(f"    - {obj_preview!r}")
                seen.add(obj_preview)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(all_kept, f, ensure_ascii=False)

    print(f"  Saved (overwritten): {fpath.name}")


if __name__ == "__main__":
    for fname in TARGET_FILES:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping.")
            continue
        process_file(fpath)

    print("\nDone. Verifying final counts:")
    for fname in TARGET_FILES:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        for tc in [0, 2000, 6000]:
            count = sum(1 for d in data if d["filler_tokens"] == tc)
            status = "OK" if count == EXPECTED_PER_TC else f"ERROR (expected {EXPECTED_PER_TC})"
            print(f"  {fname[:50]}... TC={tc}: {count} [{status}]")
