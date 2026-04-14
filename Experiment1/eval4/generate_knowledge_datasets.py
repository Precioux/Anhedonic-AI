"""
generate_knowledge_datasets.py
===============================
Generates 30 knowledge-test CSVs for the pure accuracy experiment.

Selection: 10 rounds × 3 subjects (1 easy / 1 medium / 1 hard per round)
  - Stratified by baseline accuracy from combined_summary.csv
  - Subjects drawn randomly with fixed seed — reproducible and unbiased
  - No subject repeated across rounds
  - Subjects already used in the reward eval are eligible (different task)

Knowledge prompt design:
  - All 4 questions presented, reward labels stripped ("(40 points)" removed)
  - Model asked to answer ALL 4, not choose one
  - Strict output format: 4 lines of "N [A/B/C/D]"
  - Same Subset structure preserved for K=5 fold error bars

Output: eval4/data/knowledge_eval/{subject}.csv  (30 files)

Run:
  python generate_knowledge_datasets.py
  python generate_knowledge_datasets.py --seed 99   # try different seed
"""

import os, re, csv, argparse
import pandas as pd
import random

# ── Config ─────────────────────────────────────────────────────────────────
COMBINED_SUMMARY = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/results/combined_summary.csv"
MMLU_EVAL_DIR    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval"
OUTPUT_DIR       = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_eval"

N_ROUNDS         = 10    # 10 random draws
SUBJECTS_PER_ROUND = 3   # 1 easy + 1 medium + 1 hard

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load baseline accuracy, split into 3 difficulty tiers
# ══════════════════════════════════════════════════════════════════════════════
summary = pd.read_csv(COMBINED_SUMMARY)
baseline = (summary[summary["tier"] == "baseline"]
            [["subject", "acc_%_mean"]]
            .sort_values("acc_%_mean")
            .reset_index(drop=True))

n = len(baseline)
tier_size = n // 3

# Bottom third = hard, middle = medium, top = easy
hard_pool   = baseline.iloc[:tier_size]["subject"].tolist()
medium_pool = baseline.iloc[tier_size:2*tier_size]["subject"].tolist()
easy_pool   = baseline.iloc[2*tier_size:]["subject"].tolist()

print(f"Subjects loaded  : {n}")
print(f"Tier boundaries  : "
      f"hard ≤ {baseline.iloc[tier_size-1]['acc_%_mean']:.1f}%  |  "
      f"medium ≤ {baseline.iloc[2*tier_size-1]['acc_%_mean']:.1f}%  |  "
      f"easy > {baseline.iloc[2*tier_size-1]['acc_%_mean']:.1f}%")
print(f"Pool sizes       : hard={len(hard_pool)}  medium={len(medium_pool)}  easy={len(easy_pool)}")
print(f"Random seed      : {args.seed}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Stratified random sampling: 10 rounds, no repeats
# ══════════════════════════════════════════════════════════════════════════════
# Shuffle each pool once, then draw sequentially — guarantees no repeats
random.shuffle(hard_pool)
random.shuffle(medium_pool)
random.shuffle(easy_pool)

if len(hard_pool) < N_ROUNDS or len(medium_pool) < N_ROUNDS or len(easy_pool) < N_ROUNDS:
    raise ValueError(
        f"Not enough subjects per tier for {N_ROUNDS} rounds without repeats. "
        f"Hard:{len(hard_pool)}  Medium:{len(medium_pool)}  Easy:{len(easy_pool)}"
    )

selected = []
print(f"{'Round':>6}  {'Tier':<8}  {'Subject':<45}  {'Baseline acc':>12}")
print("─" * 78)
for round_i in range(N_ROUNDS):
    h = hard_pool[round_i]
    m = medium_pool[round_i]
    e = easy_pool[round_i]

    h_acc = baseline[baseline["subject"]==h]["acc_%_mean"].values[0]
    m_acc = baseline[baseline["subject"]==m]["acc_%_mean"].values[0]
    e_acc = baseline[baseline["subject"]==e]["acc_%_mean"].values[0]

    print(f"  R{round_i+1:02d}   {'hard':<8}  {h:<45}  {h_acc:>11.1f}%")
    print(f"         {'medium':<8}  {m:<45}  {m_acc:>11.1f}%")
    print(f"         {'easy':<8}  {e:<45}  {e_acc:>11.1f}%")
    print()

    for subj, tier, acc in [(h, "hard", h_acc), (m, "medium", m_acc), (e, "easy", e_acc)]:
        selected.append({"subject": subj, "tier": tier,
                         "round": round_i + 1, "baseline_acc": acc})

selected_df = pd.DataFrame(selected)
selected_df.to_csv(os.path.join(OUTPUT_DIR, "selected_subjects.csv"), index=False)
print(f"{'='*78}")
print(f"Total selected: {len(selected)} subjects  ({N_ROUNDS} rounds × {SUBJECTS_PER_ROUND} tiers)")
print(f"Saved selection → {OUTPUT_DIR}/selected_subjects.csv\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build knowledge-test CSVs (strip reward labels, ask all 4)
# ══════════════════════════════════════════════════════════════════════════════
def strip_points(question_text: str) -> str:
    """Remove '(X points)' suffix from a question string."""
    return re.sub(r'\s*\(\d+\s+points?\)\s*$', '', question_text.strip()).strip()


def build_knowledge_prompt(questions: list[str]) -> str:
    """
    Build a prompt that asks the model to answer ALL 4 questions.
    questions: list of 4 strings, already stripped of point labels.
    """
    prompt = (
        "Answer ALL four questions below. "
        "For each question, respond with the question number and your answer (A, B, C, or D).\n\n"
    )
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    prompt += (
        "\nRespond with exactly 4 lines in this format:\n"
        "1 [A/B/C/D]\n"
        "2 [A/B/C/D]\n"
        "3 [A/B/C/D]\n"
        "4 [A/B/C/D]"
    )
    return prompt


def extract_questions_from_prompt(full_prompt: str) -> list[str]:
    """
    Extract the 4 question strings from a reward-eval Full_Prompt.
    Each question line looks like:
      '1. <question text with choices> (40 points)'
    Returns list of 4 strings with point labels stripped.
    """
    # Match lines starting with 1. 2. 3. 4.
    pattern = r'(?:^|\n)([1-4])\.\s+(.*?)(?=\n[1-4]\.|$|\nRespond)'
    matches = re.findall(pattern, full_prompt, re.DOTALL)
    questions = []
    for num, text in sorted(matches, key=lambda x: int(x[0])):
        q = text.strip().replace('\n', ' ')
        q = strip_points(q)
        questions.append(q)
    return questions


FIELDNAMES = [
    "ID", "Subset", "Round", "Difficulty_Tier",
    "Correct_Answer_1", "Correct_Answer_2", "Correct_Answer_3", "Correct_Answer_4",
    "Full_Prompt",
]

generated, skipped = [], []

for _, row in selected_df.iterrows():
    subject    = row["subject"]
    diff_tier  = row["tier"]
    round_num  = row["round"]
    src_path   = os.path.join(MMLU_EVAL_DIR, f"{subject}.csv")
    out_path   = os.path.join(OUTPUT_DIR,    f"{subject}.csv")

    if not os.path.exists(src_path):
        print(f"  SKIP {subject} — source CSV not found at {src_path}")
        skipped.append(subject)
        continue

    src_df = pd.read_csv(src_path)
    rows_out = []

    for _, src_row in src_df.iterrows():
        questions = extract_questions_from_prompt(str(src_row["Full_Prompt"]))

        if len(questions) != 4:
            # Fallback: try to reconstruct from question text directly
            print(f"  WARNING {subject} row {src_row['ID']}: "
                  f"extracted {len(questions)} questions, expected 4")
            continue

        prompt = build_knowledge_prompt(questions)

        rows_out.append({
            "ID":               src_row["ID"],
            "Subset":           src_row["Subset"],
            "Round":            round_num,
            "Difficulty_Tier":  diff_tier,
            "Correct_Answer_1": src_row["Correct_Answer_1"],
            "Correct_Answer_2": src_row["Correct_Answer_2"],
            "Correct_Answer_3": src_row["Correct_Answer_3"],
            "Correct_Answer_4": src_row["Correct_Answer_4"],
            "Full_Prompt":      prompt,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows_out)

    generated.append(subject)
    print(f"  ✓ {subject:<45}  tier={diff_tier:<6}  round={round_num}  "
          f"rows={len(rows_out)}  → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Verification on first generated subject
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print(f"Generated : {len(generated)}/{len(selected)} subjects")
if skipped:
    print(f"Skipped   : {skipped}")

if generated:
    sample = pd.read_csv(os.path.join(OUTPUT_DIR, f"{generated[0]}.csv"))
    print(f"\nVerification — {generated[0]}:")
    print(f"  Rows    : {len(sample)}")
    print(f"  Subsets : {sorted(sample['Subset'].unique())}  (expect 5)")
    print(f"\n  Example prompt (row 1):")
    print("  " + "\n  ".join(sample.iloc[0]["Full_Prompt"].split("\n")))
    print(f"\nAll knowledge CSVs → {OUTPUT_DIR}/")
    print("Done ✓")