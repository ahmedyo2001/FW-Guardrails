import json

STEP = 517

# Load your data
with open("manyshot_detail_ibm-granite_granite-guardian-3.3-8b_1776034221.json", "r", encoding="utf-8") as f:
    data = json.load(f)

n = len(data)
new_data = []

# Reorder
for start in range(STEP):
    idx = start
    while idx < n:
        new_data.append(data[idx])
        idx += STEP

# Save result
with open("manyshot_detail_ibm-granite_granite-guardian-3.3-8b_1776034221reordered.json", "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Reordered {len(new_data)} items")