#!/usr/bin/env python3
"""
Stage 3: Pass Bypassed Prompts to Uncensored LLMs
==================================================
Reads standardized summary file(s) from Stage 2 (attack notebooks),
filters to prompts where at least 1 guardrail was bypassed on the
modified prompt, and probes uncensored LLMs with both the original
and modified prompts.

Usage:
    python pass_to_uncensored.py \
        --input results/jigsaw_summary_*.json \
        --output results/uncensored_jigsaw_puzzle.json \
        --models huihui-ai/Qwen3-8B-abliterated huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated \
        --device auto --cache-dir ./models_cache

Dependencies:
    pip install torch transformers accelerate
"""

import argparse
import gc
import glob
import json
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pass_to_uncensored")


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_summaries(input_patterns: list) -> list:
    """Load and merge all Stage 2 summary files matching the input patterns."""
    all_entries = []
    source_files = []
    for pattern in input_patterns:
        for path in sorted(glob.glob(pattern)):
            logger.info(f"Loading: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != "1.0":
                logger.warning(f"  Unexpected schema_version in {path}, skipping")
                continue
            source_files.append(path)
            for entry in data["results"]:
                entry["_source_file"] = path
                entry["_attack_type"] = data.get("attack_type", "unknown")
                all_entries.append(entry)
    logger.info(f"Loaded {len(all_entries)} entries from {len(source_files)} file(s)")
    return all_entries, source_files


def filter_bypassed(entries: list) -> list:
    """Keep only entries where at least 1 guardrail was bypassed on modified prompt."""
    bypassed = []
    for entry in entries:
        bypassed_guardrails = [
            gr["model_id"]
            for gr in entry["guardrail_results"]
            if not gr["flagged_modified"]
        ]
        if bypassed_guardrails:
            entry["_bypassed_guardrails"] = bypassed_guardrails
            bypassed.append(entry)
    logger.info(f"Filtered to {len(bypassed)} entries with at least 1 bypass")
    return bypassed


def probe_with_model(model_id: str, entries: list, device: str, cache_dir: str,
                     max_new_tokens: int) -> list:
    """Load one uncensored model, probe all entries, return results."""
    logger.info(f"Loading uncensored model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, device_map=device, trust_remote_code=True,
    )
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device)

    detail_records = []

    for i, entry in enumerate(entries, 1):
        prompt_id = entry["prompt_id"]
        original = entry["original_prompt"]
        modified = entry["modified_prompt"]
        logger.info(f"  [{i}/{len(entries)}] {prompt_id}: {original[:50]}...")

        results_for_entry = []

        # Probe mode: original
        messages_orig = [{"role": "user", "content": original}]
        prompt_orig = tokenizer.apply_chat_template(messages_orig, tokenize=False, add_generation_prompt=True)
        output_orig = pipe(prompt_orig, max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=0.7, top_p=0.9, return_full_text=False)
        response_orig = output_orig[0]["generated_text"].strip()
        logger.info(f"    [original] {response_orig[:80]}...")

        results_for_entry.append({
            "model_id": model_id,
            "probe_mode": "original",
            "response": response_orig,
        })

        # Probe mode: modified
        messages_mod = [{"role": "user", "content": modified}]
        prompt_mod = tokenizer.apply_chat_template(messages_mod, tokenize=False, add_generation_prompt=True)
        output_mod = pipe(prompt_mod, max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=0.7, top_p=0.9, return_full_text=False)
        response_mod = output_mod[0]["generated_text"].strip()
        logger.info(f"    [modified] {response_mod[:80]}...")

        results_for_entry.append({
            "model_id": model_id,
            "probe_mode": "modified",
            "response": response_mod,
        })

        # Detailed record for per-model file
        detail_records.append({
            "prompt_id": prompt_id,
            "original_prompt": original,
            "modified_prompt": modified,
            "model_id": model_id,
            "response_original": response_orig,
            "response_modified": response_mod,
            "prompt_sent_original": prompt_orig,
            "prompt_sent_modified": prompt_mod,
        })

        # Attach to entry
        if "_responses" not in entry:
            entry["_responses"] = []
        entry["_responses"].extend(results_for_entry)

    logger.info(f"Unloading {model_id}")
    del pipe, model, tokenizer
    flush_memory()

    return detail_records


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Probe bypassed prompts with uncensored LLMs"
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="Stage 2 summary file(s) or glob pattern(s)")
    parser.add_argument("--output", required=True,
                        help="Output JSON file (input for judge.py)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Uncensored model IDs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="./models_cache")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load and filter
    entries, source_files = load_summaries(args.input)
    bypassed = filter_bypassed(entries)

    if not bypassed:
        print("No bypassed prompts found. Nothing to probe.")
        return

    print(f"\n{len(bypassed)} bypassed entries to probe with {len(args.models)} model(s)")
    print("-" * 60)

    # Probe with each model
    for model_id in args.models:
        detail_records = probe_with_model(
            model_id, bypassed, args.device, args.cache_dir, args.max_new_tokens
        )

        # Save per-model detail file
        model_short = model_id.split("/")[-1][:30]
        detail_path = f"results/uncensored_detail_{model_short}_{int(time.time())}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_records, f, indent=2, ensure_ascii=False)
        print(f"  Detail saved: {detail_path}")

    # Build main output (input for Stage 4)
    attack_type = bypassed[0].get("_attack_type", "unknown") if bypassed else "unknown"
    output_results = []
    for entry in bypassed:
        output_results.append({
            "prompt_id": entry["prompt_id"],
            "original_prompt": entry["original_prompt"],
            "modified_prompt": entry["modified_prompt"],
            "bypassed_guardrails": entry.get("_bypassed_guardrails", []),
            "responses": entry.get("_responses", []),
        })

    output = {
        "schema_version": "1.0",
        "timestamp": int(time.time()),
        "source_summaries": source_files,
        "attack_type": attack_type,
        "uncensored_models": args.models,
        "results": output_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done. {len(output_results)} entries probed.")
    print(f"Output saved to {args.output}")
    print(f"This file is the input for judge.py (Stage 4)")


if __name__ == "__main__":
    main()
