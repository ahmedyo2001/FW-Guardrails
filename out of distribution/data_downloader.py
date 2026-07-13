# ------------------------------------------------------------
# 1) Install (once)
# ------------------------------------------------------------
# !pip install -q datasets pandas

from datasets import load_dataset, concatenate_datasets
import pandas as pd

# ------------------------------------------------------------
# 2) Load ALL splits of the "full" config
# ------------------------------------------------------------
dataset = load_dataset(
    "neuralchemy/Prompt-injection-dataset",
    "full",            # config name
    split=None         # loads train + validation + test as a DatasetDict
)

print(dataset)  # sanity check: shows each split + row counts

# ------------------------------------------------------------
# 3) Same filter as before: category == "jailbreak" AND
#    source in {"original", "neuralchemy_v1"}
# ------------------------------------------------------------
def keep_row(row):
    return (row["category"] == "jailbreak") and (
        row["source"] in {"original", "neuralchemy_v1"}
    )

filtered = dataset.filter(keep_row, batched=False)  # filters each split in the DatasetDict

# ------------------------------------------------------------
# 4) Concatenate the filtered splits into one DataFrame
# ------------------------------------------------------------
df = concatenate_datasets(list(filtered.values())).to_pandas()

print(f"Rows kept: {len(df)}")

# ------------------------------------------------------------
# 5) Save to JSON
# ------------------------------------------------------------
output_path = "jailbreak_original_or_alchemy_full.json"
df.to_json(output_path, orient="records", indent=2, force_ascii=False)

print(f"Filtered data written to: {output_path}")