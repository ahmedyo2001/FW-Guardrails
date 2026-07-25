import json
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

TOKEN_COUNTS = [0, 2000, 4000, 6000, 8000]

MODEL_LABELS = {
    "meta-llama/Llama-Guard-3-1B":               "LlamaGuard-3\n1B",
    "meta-llama/Llama-Guard-3-8B":               "LlamaGuard-3\n8B",
    "ibm-granite/granite-guardian-3.3-8b":       "Granite-Guard\n3.3-8B",
    "ibm-granite/granite-guardian-3.2-3b-a800m": "Granite-Guard\n3.2-3B",
    "google/shieldgemma-9b":                     "ShieldGemma\n9B",
    "Qwen/Qwen3Guard-Gen-8B":                    "Qwen3Guard\n8B",
    "approach1_ppl_threshold":                   "PPL\nThreshold",
    "approach2_lgbm":                            "LGBM\n(ppl+stats)",
    "approach3_embed_char":                      "Embed\n(char)",
    "approach3_embed_sentence":                  "Embed\n(sentence)",
    "approach3_embed_full":                      "Embed\n(full)",
}

FAMILY_COLORS = {
    "meta-llama/Llama-Guard-3-1B":               "#1f77b4",
    "meta-llama/Llama-Guard-3-8B":               "#1f77b4",
    "ibm-granite/granite-guardian-3.3-8b":       "#2ca02c",
    "ibm-granite/granite-guardian-3.2-3b-a800m": "#2ca02c",
    "google/shieldgemma-9b":                     "#000000",
    "Qwen/Qwen3Guard-Gen-8B":                    "#000000",
    "approach1_ppl_threshold":                   "#d62728",
    "approach2_lgbm":                            "#9467bd",
    "approach3_embed_char":                      "#e377c2",
    "approach3_embed_sentence":                  "#e377c2",
    "approach3_embed_full":                      "#e377c2",
}

# v2 perplexity/embedding-based detector results live in a single aggregated
# summary.json (keyed by test_file + approach) rather than per-model detail files.
V2_SUMMARY_PATH = os.path.join("perplexity&others", "v2", "test_results", "summary.json")

COLORS = {
    0:    "#d0dce8",
    2000: "#8aafc8",
    4000: "#4682b4",
    6000: "#2a5080",
    8000: "#1a3a5c",
}
HATCH = {0: "", 2000: "", 4000: "", 6000: "", 8000: "//"}

DATASETS = [
    {
        "dir":        "many_shots_attack/results_jackhhao",
        "n_per_tc":   517,
        "v2_prefix":  "jackhhao",
        "title":      "Many-Shot Attack: Success Rate per Model and Context Length\n(jackhhao dataset)",
        "out":        "attack_success_rate_jackhhao_full.png",
    },
    {
        "dir":        "many_shots_attack/results_safebench",
        "n_per_tc":   350,
        "v2_prefix":  "safebench",
        "title":      "Many-Shot Attack: Success Rate per Model and Context Length\n(safebench dataset)",
        "out":        "attack_success_rate_safebench_full.png",
    },
]


def load_results(results_dir, n_per_tc):
    """Return {model_id: {tc: bypass_%}} for all 5 token counts."""
    results = {}

    # Old files (tc = 0, 2000, 6000)
    for fpath in glob.glob(os.path.join(results_dir, "manyshot_detail_*.json")):
        if re.search(r"4k|8k", os.path.basename(fpath)):
            continue
        data = json.load(open(fpath, encoding="utf-8"))
        model_id = data[0]["guardrail_model_id"]
        if model_id not in results:
            results[model_id] = {}
        for tc in [0, 2000, 6000]:
            subset = [d for d in data if d["filler_tokens"] == tc]
            safe = sum(1 for d in subset if not d["flagged_modified"])
            results[model_id][tc] = safe / n_per_tc * 100

    # New files (tc = 0, 4000, 8000) — use only 4000 and 8000; 0 comes from old file
    for fpath in glob.glob(os.path.join(results_dir, "manyshot_detail_*.json")):
        if not re.search(r"4k|8k", os.path.basename(fpath)):
            continue
        data = json.load(open(fpath, encoding="utf-8"))
        model_id = data[0]["guardrail_model_id"]
        if model_id not in results:
            results[model_id] = {}
        for tc in [4000, 8000]:
            subset = [d for d in data if d["filler_tokens"] == tc]
            safe = sum(1 for d in subset if not d["flagged_modified"])
            results[model_id][tc] = safe / n_per_tc * 100

    return results


def load_v2_results(summary_path, dataset_prefix):
    """Return {approach: {tc: bypass_%}} from the aggregated v2 summary.json.

    v2 has no explicit 0-filler-token run; the un-manyshotted "jigsaw_original"
    test file (raw attack prompts) stands in for tc=0.
    """
    with open(summary_path, encoding="utf-8") as f:
        rows = json.load(f)

    tc_by_test_file = {f"{dataset_prefix}_jigsaw_original": 0}
    for tc in [2000, 4000, 6000, 8000]:
        tc_by_test_file[f"{dataset_prefix}_manyshot_{tc}tokens"] = tc

    results = {}
    for row in rows:
        tc = tc_by_test_file.get(row["test_file"])
        if tc is None or row.get("detection_rate") is None:
            continue
        approach = row["approach"]
        bypass_pct = (1 - row["detection_rate"]) * 100
        results.setdefault(approach, {})[tc] = bypass_pct

    return results


def plot_dataset(results, title, out_path):
    model_ids = sorted(results.keys(), key=lambda m: results[m].get(8000, 0), reverse=True)
    labels = [MODEL_LABELS.get(m, m) for m in model_ids]

    n_models = len(model_ids)
    n_groups = len(TOKEN_COUNTS)
    bar_width = 0.15
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(13, n_models * 1.4), 6))

    for i, tc in enumerate(TOKEN_COUNTS):
        offsets = x + (i - (n_groups - 1) / 2) * bar_width
        heights = [results[m].get(tc, 0) for m in model_ids]
        bars = ax.bar(
            offsets, heights,
            width=bar_width,
            color=COLORS[tc],
            hatch=HATCH[tc],
            edgecolor="white",
            linewidth=0.6,
            label=f"{tc:,} tokens",
            zorder=3,
        )
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.6,
                    f"{h:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    for tick, m in zip(ax.get_xticklabels(), model_ids):
        tick.set_color(FAMILY_COLORS.get(m, "#333333"))
    ax.set_ylabel("Attack Success Rate\n(% bypassed)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    max_val = max(v for m in results.values() for v in m.values())
    ax.set_ylim(0, max_val * 1.20 + 5)
    ax.legend(title="Filler context", fontsize=9, title_fontsize=9, ncol=5)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


for ds in DATASETS:
    results = load_results(ds["dir"], ds["n_per_tc"])
    results.update(load_v2_results(V2_SUMMARY_PATH, ds["v2_prefix"]))
    plot_dataset(results, ds["title"], ds["out"])
