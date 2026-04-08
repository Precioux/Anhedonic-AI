"""
generate_origin_eval.py
================================================================================
Builds a perfectly counterbalanced origin math evaluation dataset.

Design (mirrors asdiv_balanced_eval.csv):
  - 96 rows = 24 permutations of [10, 20, 30, 40] × 4
  - Each point value appears in each position exactly 24 times
  - Questions sourced from full_experiment_100_rows_v3.csv (original questions)
  - Prompt format: ASDiv CRITICAL INSTRUCTION style
  - 5 subsets of ~20 rows for error bars
  - Correct answers computed by evaluating the math expression

Output: data/origin_math_eval.csv
Columns: ID, Subset, Reward_Order,
         Correct_Answer_1..4, Reward_1..4,
         Full_Prompt
"""

import re
import os
import csv
import math
import random
import itertools
import pandas as pd

random.seed(42)

INPUT_CSV  = "data/full_experiment_100_rows_v3.csv"
OUTPUT_CSV = "data/origin_math_eval.csv"
REWARDS    = [10, 20, 30, 40]

os.makedirs("data", exist_ok=True)

# =============================================================================
# Step 1 — Parse and deduplicate questions from original CSV
# =============================================================================
df_orig = pd.read_csv(INPUT_CSV)

raw_questions = []
for _, row in df_orig.iterrows():
    prompt = row['Full_Prompt']
    matches = re.findall(r'[1-4]\.\s+(.*?)\s+\(\d+ points?\)', prompt)
    for q in matches:
        raw_questions.append(q.strip())

unique_questions = list(dict.fromkeys(raw_questions))
print(f"Total questions parsed:  {len(raw_questions)}")
print(f"Unique questions:        {len(unique_questions)}")

# =============================================================================
# Step 2 — Compute correct answers
# =============================================================================
def compute_answer(question: str) -> str:
    """Evaluate simple arithmetic from 'What is X op Y?' format."""
    expr_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', question)
    if not expr_match:
        return ""
    a, op, b = int(expr_match.group(1)), expr_match.group(2), int(expr_match.group(3))
    if op == '+': return str(a + b)
    if op == '-': return str(a - b)
    if op == '*': return str(a * b)
    if op == '/': return str(a // b)
    return ""

qa_pool = []
for q in unique_questions:
    ans = compute_answer(q)
    if ans:
        qa_pool.append({"question": q, "answer": ans})

print(f"Questions with valid answers: {len(qa_pool)}")

# =============================================================================
# Step 3 — Build 96 counterbalanced rows
# =============================================================================
all_perms = list(itertools.permutations(REWARDS))   # 24 permutations
assert len(all_perms) == 24

reward_orders = all_perms * 4   # 96 rows total
random.shuffle(reward_orders)

# Shuffle question pool and sample groups of 4
# We need 96 × 4 = 384 question slots, pool has ~200 — allow reuse
random.shuffle(qa_pool)
extended_pool = []
while len(extended_pool) < 96 * 4:
    shuffled = qa_pool[:]
    random.shuffle(shuffled)
    extended_pool.extend(shuffled)

# Assign subsets (5 subsets of ~20 rows)
subset_size = 96 // 5
subsets = []
for i in range(96):
    subsets.append(min(i // subset_size + 1, 5))

# =============================================================================
# Step 4 — Build rows
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

    prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and provide the final answer."
    row["Full_Prompt"] = prompt
    rows.append(row)

# =============================================================================
# Step 5 — Save and verify
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
    counts = [
        (df[f"Reward_{pos}"] == pts).sum()
        for pos in [1, 2, 3, 4]
    ]
    print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}")

print(f"\nSubset distribution:")
print(df["Subset"].value_counts().sort_index().to_string())

print(f"\nSample row 1:")
print(df["Full_Prompt"].iloc[0])
print("\nSanity check passed ✓" if len(df) == 96 else "ERROR: row count wrong")