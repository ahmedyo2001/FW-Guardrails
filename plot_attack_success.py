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
}

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
        "title":      "Many-Shot Attack: Success Rate per Model and Context Length\n(jackhhao dataset)",
        "out":        "attack_success_rate_jackhhao_full.png",
    },
    {
        "dir":        "many_shots_attack/results_safebench",
        "n_per_tc":   350,
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


def plot_dataset(results, title, out_path):
    model_ids = sorted(results.keys(), key=lambda m: results[m].get(8000, 0), reverse=True)
    labels = [MODEL_LABELS.get(m, m) for m in model_ids]

    n_models = len(model_ids)
    n_groups = len(TOKEN_COUNTS)
    bar_width = 0.15
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(13, 6))

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
    plot_dataset(results, ds["title"], ds["out"])
