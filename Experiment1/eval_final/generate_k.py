"""
generate_accuracy.py
Reads data/asdiv_eval_dataset.json and explodes each row into 4 individual
question prompts — one per question, no reward framing.
Output: data/asdiv_accuracy_dataset.json  (384 rows = 96 × 4)
"""
import json, os

with open("data/asdiv_eval_dataset.json") as f:
    rows = json.load(f)

out = []
for row in rows:
    for pos in range(1, 5):
        out.append({
            "source_permutation": row["permutation"],
            "position":           pos,
            "points":             row[f"q{pos}_points"],
            "question":           row[f"q{pos}_question"],
            "answer":             row[f"q{pos}_answer"],
            "prompt":             (
                f"{row[f'q{pos}_question']}\n\n"
                "Solve this math problem. Respond with the final numeric answer only."
            ),
        })

os.makedirs("data", exist_ok=True)
with open("data/asdiv_accuracy_dataset.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Saved {len(out)} rows → data/asdiv_accuracy_dataset.json")
# Quick sanity check
from collections import Counter
print("Points distribution:", dict(sorted(Counter(r["points"] for r in out).items())))