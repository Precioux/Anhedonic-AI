"""
Post-hoc fix: recover chosen_points from chosen_question + original prompt.
Run: python fix_points.py
"""

import csv
import re

INPUT_RESULTS  = "results_normal_mode.csv"
INPUT_PROMPTS  = "data/full_experiment_100_rows.csv"
OUTPUT_RESULTS = "results_normal_mode.csv"

# Load original prompts to build question→points mapping per prompt
with open(INPUT_PROMPTS, newline="", encoding="utf-8") as f:
    prompts = {row["ID"]: row["Full_Prompt"] for row in csv.DictReader(f)}

def extract_points_map(prompt_text):
    """Returns {question_number: points} for a given prompt."""
    mapping = {}
    for line in prompt_text.split("\n"):
        m = re.match(r"^(\d)\.\s.*?\((\d+)\s*points?\)", line)
        if m:
            mapping[int(m.group(1))] = int(m.group(2))
    return mapping

# Load results and patch chosen_points
with open(INPUT_RESULTS, newline="", encoding="utf-8") as f:
    results = list(csv.DictReader(f))

fixed = 0
for row in results:
    if row["chosen_points"] in ("None", "", None):
        q = row["chosen_question"]
        if q not in ("None", "", None):
            pts_map = extract_points_map(prompts[row["id"]])
            pts = pts_map.get(int(q))
            row["chosen_points"] = pts
            fixed += 1

with open(OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Fixed {fixed} rows → {OUTPUT_RESULTS}")

# Summary
from collections import Counter
valid  = [r for r in results if r["chosen_points"] not in ("None", "", None)]
counts = Counter(int(r["chosen_points"]) for r in valid)
print("\n── Point distribution ──")
for pts in sorted(counts, reverse=True):
    print(f"  {pts:>3} pts : {counts[pts]:>3}  ({100*counts[pts]/len(valid):.1f}%)")
print(f"\n  Mean points chosen: {sum(int(r['chosen_points']) for r in valid)/len(valid):.1f}")