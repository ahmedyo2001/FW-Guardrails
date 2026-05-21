import json
import glob

data = []

for filepath in glob.glob("jigsaw_detail_*"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = json.load(f)
        if isinstance(content, list):
            data.extend(content)
        else:
            data.append(content)

hashmap_bening: dict[str, int] = {}

hashmap_malicious: dict[str, int] = {}


for obj in data:
    prompt = obj["objective"]
    malprompt = obj["jsp_prompt_raw"]

    if not obj["flagged_original"]:
        hashmap_bening[prompt] = hashmap_bening.get(prompt, 0) + 1
    
    if not obj["flagged_modified"]:
        hashmap_malicious[malprompt] = hashmap_malicious.get(malprompt, 0) + 1

benign_6 = {p: c for p, c in hashmap_bening.items() if c >= 6}
malicious_6 = {p: c for p, c in hashmap_malicious.items() if c >= 6}
with open("benign_6.json", "w") as f:
    json.dump(benign_6, f, indent=4)

with open("malicious_6.json", "w") as f:
    json.dump(malicious_6, f, indent=4)

