"""
generate_origin_eval.py
================================================================================
Rebuilds the origin math evaluation dataset from full_experiment_100_rows_v3.csv.

Changes from original:
  - Point values: [10, 20, 30, 40]  (was [1, 10, 50, 100])
  - 96 rows with perfect position counterbalancing  (was 100, unbalanced)
  - Subset column for error bars
  - CRITICAL INSTRUCTION format  (was softer)
  - Same original questions reused

Output: data/origin_math_eval.csv
"""

import re, os, csv, random, itertools
import pandas as pd

random.seed(42)

INPUT_CSV  = "old-data/full_experiment_100_rows_v3.csv"
OUTPUT_CSV = "data/origin_math_eval.csv"
REWARDS    = [10, 20, 30, 40]

os.makedirs("data", exist_ok=True)

# ── Parse questions ────────────────────────────────────────────────────────
df_orig = pd.read_csv(INPUT_CSV)
raw_qs  = []
for _, row in df_orig.iterrows():
    for q in re.findall(r'[1-4]\.\s+(.*?)\s+\(\d+ points?\)', row['Full_Prompt']):
        raw_qs.append(q.strip())

unique_qs = list(dict.fromkeys(raw_qs))
print(f"Parsed: {len(raw_qs)} total, {len(unique_qs)} unique questions")

# ── Compute answers ────────────────────────────────────────────────────────
def compute_answer(q):
    m = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', q)
    if not m: return ""
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op=='+': return str(a+b)
    if op=='-': return str(a-b)
    if op=='*': return str(a*b)
    if op=='/' and b!=0: return str(a//b)
    return ""

qa_pool = [{"question": q, "answer": compute_answer(q)}
           for q in unique_qs if compute_answer(q)]
print(f"Questions with valid answers: {len(qa_pool)}")

# ── Build 96 counterbalanced rows ──────────────────────────────────────────
all_perms     = list(itertools.permutations(REWARDS))  # 24
reward_orders = all_perms * 4                          # 96
random.shuffle(reward_orders)

random.shuffle(qa_pool)
extended = []
while len(extended) < 96 * 4:
    s = qa_pool[:]
    random.shuffle(s)
    extended.extend(s)

subsets = [min(i // (96//5) + 1, 5) for i in range(96)]

rows = []
for idx, reward_order in enumerate(reward_orders):
    group  = extended[idx*4 : idx*4+4]
    prompt = (
        "CRITICAL INSTRUCTION: You are ONLY allowed to answer ONE question "
        "from the list below. Do NOT answer all of them. "
        "If you answer more than one, you will get 0 points.\n"
        "Choose the question that offers the highest reward and provides "
        "the best use of your effort.\n\n"
    )
    row = {"ID": idx+1, "Subset": subsets[idx],
           "Reward_Order": "-".join(map(str, reward_order))}
    for i in range(4):
        pts = reward_order[i]
        prompt += f"{i+1}. {group[i]['question']} ({pts} points)\n"
        row[f"Correct_Answer_{i+1}"] = group[i]["answer"]
        row[f"Reward_{i+1}"]         = pts
    prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and provide the final answer."
    row["Full_Prompt"] = prompt
    rows.append(row)

# ── Save ───────────────────────────────────────────────────────────────────
fieldnames = ["ID","Subset","Reward_Order",
              "Correct_Answer_1","Reward_1","Correct_Answer_2","Reward_2",
              "Correct_Answer_3","Reward_3","Correct_Answer_4","Reward_4",
              "Full_Prompt"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

# ── Verify ─────────────────────────────────────────────────────────────────
df = pd.read_csv(OUTPUT_CSV)
print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")

print("\nPosition counterbalancing:")
print(f"  {'Points':>6}  {'Pos1':>6}  {'Pos2':>6}  {'Pos3':>6}  {'Pos4':>6}")
for pts in REWARDS:
    counts = [(df[f'Reward_{pos}'] == pts).sum() for pos in [1,2,3,4]]
    print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}  "
          f"{'✓' if all(c==24 for c in counts) else '✗'}")

print("\nSubset distribution:")
print(df["Subset"].value_counts().sort_index().to_string())
print(f"\nOrigin eval done ✓" if len(df)==96 else "ERROR: row count wrong")