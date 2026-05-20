import json

STEP = 517

# Load files
with open("manyshot_detail_ibm-granite_granite-guardian-3.3-8b_1776034221.json", "r", encoding="utf-8") as f:
    original = json.load(f)

with open("manyshot_detail_ibm-granite_granite-guardian-3.3-8b_1776034221reordered.json", "r", encoding="utf-8") as f:
    reordered = json.load(f)

# Recompute expected order
expected = []
n = len(original)

for start in range(STEP):
    idx = start
    while idx < n:
        expected.append(original[idx])
        idx += STEP

# Check lengths first
if len(expected) != len(reordered):
    print(f"❌ Length mismatch: expected {len(expected)}, got {len(reordered)}")
    exit()

# Compare element by element
for i, (exp, got) in enumerate(zip(expected, reordered)):
    if exp != got:
        print(f"❌ Mismatch at index {i}")
        print("Expected:", exp)
        print("Got     :", got)
        break
else:
    print("✅ All good! The reordering is correct.")