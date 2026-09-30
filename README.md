# FW-Guardrails

Evaluation of open-source LLM guardrail models against adversarial attacks, with lightweight perplexity/embedding-based detectors as a comparison.

> For authorized security testing and research purposes only.

## Guardrails evaluated

**LLM-based:** Llama Guard 3 (1B, 8B) · Granite Guardian (3.2 3B, 3.3 8B) · ShieldGemma 9B · Qwen3Guard Gen 8B

**Non-LLM (in `perplexity&others/`, perplexity from Llama 3.2 1B):**
- Windowed perplexity threshold
- LightGBM on perplexity + token length ([Alon & Kamfonas, 2023](https://arxiv.org/abs/2308.14132))
- Embedding similarity to known attack prompts, based on [Arize dataset embeddings guardrail](https://github.com/Arize-ai/dataset-embeddings-guardrails) (char chunks, sentence chunks, full prompt)

## Datasets

- **Malicious:** jackhhao jailbreak, SafeBench (`dataset_downloaders/`)
- **Benign (false refusals):** Dolly, XSTest, FalseReject (`benign_data/`)
- **OOD jailbreaks:** `out of distribution/`

## Repo layout

| Folder | What it does |
|---|---|
| `jigsaw_attack/` | Jigsaw Puzzle attack: harmful words split into fragments ([arXiv:2410.11459](https://arxiv.org/abs/2410.11459)) |
| `many_shots_attack/` | Many-shot attack: harmful query buried after 0–8k tokens of filler |
| `obfuscation/` | Obfuscated versions of the harmful prompts |
| `prompt_poisoning/`, `prompt_poisoning2/` | System prompt poisoning |
| `out of distribution/` | OOD jailbreak prompts |
| `benign_data/` | False refusal rate on benign prompts |
| `probing/` | Cross-trigger probing experiments |
| `perplexity&others/` | Non-LLM guardrails (see above): training, testing and results. Versions `v1`–`v3` |

Each attack folder has a `colab_*.ipynb` (run on a GPU in Colab), `results*/` with per-model JSON outputs, and a `make_quick_summary.py` that sums those results up.

## Quick start

```bash
# Guardrail attacks: open the colab_*.ipynb in each folder and run it on a GPU
# (needs a Hugging Face token with access to the gated models)

# Our detectors
pip install datasets transformers torch sentence-transformers scikit-learn lightgbm
cd "perplexity&others"
python prep_attack_test_data.py   # build attack test sets
python train_all.py               # train all 5 detector variants
python test_all.py                # evaluate them on the attack sets

# Attack success rate plots
python plot_attack_success.py
```

## Outputs

- `attack_success_rate_*.png`: attack success rate per guardrail and approach
- `main.pdf`: write-up
- `llm_guardrail_taxonomy_v9.svg`: attack taxonomy
