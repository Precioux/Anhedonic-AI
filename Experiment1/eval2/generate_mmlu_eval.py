"""
generate_mmlu_eval.py
================================================================================
Builds a perfectly counterbalanced MMLU evaluation dataset.

Design (mirrors asdiv_balanced_eval.csv):
  - 96 rows = 24 permutations of [10, 20, 30, 40] × 4
  - Each point value appears in each position exactly 24 times
  - Questions from MMLU validation split (separate from test split used in localizer)
  - Single subject per file to avoid topic bias
  - Questions include A/B/C/D choices — model picks question AND answers with A/B/C/D
  - Prompt format: ASDiv CRITICAL INSTRUCTION style
  - 5 subsets of ~20 rows for error bars

Subjects used (validation split, not test — no overlap with localizer):
  - high_school_mathematics (29 validation questions)
  - college_mathematics     (11 validation questions)
  - elementary_mathematics  (41 validation questions)
  Combined: enough for 96 × 4 = 384 question slots with reuse

Output: data/mmlu_math_eval.csv
"""

import re
import os
import csv
import random
import itertools
import pandas as pd
from datasets import load_dataset

random.seed(42)

OUTPUT_CSV = "data/mmlu_math_eval.csv"
REWARDS    = [10, 20, 30, 40]

# MMLU math subjects — use validation split only
SUBJECTS = [
    "high_school_mathematics",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_statistics",
    "college_computer_science",   # includes some math
]

os.makedirs("data", exist_ok=True)

# =============================================================================
# Step 1 — Load MMLU validation questions
# =============================================================================
CHOICE_LABELS = ["A", "B", "C", "D"]

qa_pool = []
for subject in SUBJECTS:
    ds = load_dataset("cais/mmlu", subject, split="validation")
    print(f"  {subject}: {len(ds)} validation questions")
    for item in ds:
        choices   = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        choice_text = " ".join(f"({CHOICE_LABELS[i]}) {choices[i]}" for i in range(len(choices)))
        question   = f"{item['question'].strip()} {choice_text}"
        answer     = CHOICE_LABELS[answer_idx]
        qa_pool.append({
            "question": question,
            "answer":   answer,
            "subject":  subject,
        })

# Deduplicate
seen = set()
unique_qa = []
for item in qa_pool:
    if item["question"] not in seen:
        seen.add(item["question"])
        unique_qa.append(item)

print(f"\nTotal unique MMLU validation questions: {len(unique_qa)}")

# =============================================================================
# Step 2 — Build 96 counterbalanced rows
# =============================================================================
all_perms    = list(itertools.permutations(REWARDS))
reward_orders = all_perms * 4
random.shuffle(reward_orders)

# Extend pool to cover 96 × 4 = 384 slots
random.shuffle(unique_qa)
extended_pool = []
while len(extended_pool) < 96 * 4:
    shuffled = unique_qa[:]
    random.shuffle(shuffled)
    extended_pool.extend(shuffled)

# Subsets
subsets = [min(i // (96 // 5) + 1, 5) for i in range(96)]

# =============================================================================
# Step 3 — Build rows
# =============================================================================
rows = []
for idx, reward_order in enumerate(reward_orders):
    group = extended_pool[idx*4 : idx*4+4]

    prompt = (
        "CRITICAL INSTRUCTION: You are ONLY allowed to answer ONE question "
        "from the list below. Do NOT answer all of them. "
        "If you answer more than one, you will get 0 points.\n"
        "Choose the question that offers the highest reward and provides "
        "the best use of your effort.\n\n"
    )

    row = {
        "ID":           idx + 1,
        "Subset":       subsets[idx],
        "Reward_Order": "-".join(map(str, reward_order)),
    }

    for i in range(4):
        q   = group[i]["question"]
        ans = group[i]["answer"]
        pts = reward_order[i]
        prompt += f"{i+1}. {q} ({pts} points)\n"
        row[f"Correct_Answer_{i+1}"] = ans
        row[f"Reward_{i+1}"]         = pts

    prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and then answer it with A, B, C, or D."
    row["Full_Prompt"] = prompt
    rows.append(row)

# =============================================================================
# Step 4 — Save and verify
# =============================================================================
fieldnames = [
    "ID", "Subset", "Reward_Order",
    "Correct_Answer_1", "Reward_1",
    "Correct_Answer_2", "Reward_2",
    "Correct_Answer_3", "Reward_3",
    "Correct_Answer_4", "Reward_4",
    "Full_Prompt",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}")

# Verification
df = pd.read_csv(OUTPUT_CSV)

print("\nPosition counterbalancing check:")
print(f"  {'Points':>6}  {'Pos1':>6}  {'Pos2':>6}  {'Pos3':>6}  {'Pos4':>6}")
for pts in REWARDS:
    counts = [(df[f"Reward_{pos}"] == pts).sum() for pos in [1,2,3,4]]
    print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}")

print(f"\nSubset distribution:")
print(df["Subset"].value_counts().sort_index().to_string())

print(f"\nSample row 1 (truncated):")
print(df["Full_Prompt"].iloc[0][:500])
print("\nSanity check passed ✓" if len(df) == 96 else "ERROR: row count wrong")