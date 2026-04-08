"""
generate_mmlu_eval.py
================================================================================
Generates 54 separate MMLU evaluation CSVs — one per subject.

Design per file:
  - 96 rows = 24 permutations of [10, 20, 30, 40] × 4
  - Each point value in each position exactly 24 times
  - All 4 questions per row from the SAME subject (no topic mixing)
  - Questions from MMLU validation split (not test — no overlap with localizer)
  - Answer format: A/B/C/D
  - CRITICAL INSTRUCTION format
  - 5 subsets of ~20 rows

Output: data/mmlu_eval/{subject}.csv  (54 files)

Note: subjects with < 4 validation questions are skipped and reported.
"""

import os, csv, random, itertools
import pandas as pd
from datasets import load_dataset

random.seed(42)

OUTPUT_DIR = "data/mmlu_eval"
REWARDS    = [10, 20, 30, 40]
LABELS     = ["A", "B", "C", "D"]

SUBJECTS = [
    "abstract_algebra","anatomy","astronomy","business_ethics",
    "clinical_knowledge","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_medicine",
    "college_physics","computer_security","conceptual_physics",
    "econometrics","electrical_engineering","elementary_mathematics",
    "formal_logic","global_facts","high_school_biology",
    "high_school_chemistry","high_school_computer_science",
    "high_school_european_history","high_school_geography",
    "high_school_government_and_politics","high_school_macroeconomics",
    "high_school_mathematics","high_school_microeconomics",
    "high_school_physics","high_school_psychology","high_school_statistics",
    "high_school_us_history","high_school_world_history","human_aging",
    "human_sexuality","international_law","jurisprudence",
    "logical_fallacies","machine_learning","management","marketing",
    "medical_genetics","miscellaneous","moral_disputes","moral_scenarios",
    "nutrition","philosophy","prehistory","professional_accounting",
    "professional_law","professional_medicine","professional_psychology",
    "public_relations","security_studies","sociology","us_foreign_policy",
    "virology","world_religions",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = ["ID","Subset","Reward_Order",
              "Correct_Answer_1","Reward_1","Correct_Answer_2","Reward_2",
              "Correct_Answer_3","Reward_3","Correct_Answer_4","Reward_4",
              "Full_Prompt"]

all_perms     = list(itertools.permutations(REWARDS))  # 24
reward_orders = all_perms * 4                          # 96
subsets       = [min(i // (96//5) + 1, 5) for i in range(96)]

skipped = []
generated = []

for subject in SUBJECTS:
    # Load validation split
    ds = load_dataset("cais/mmlu", subject, split="validation")

    # Build QA pool
    qa_pool = []
    for item in ds:
        choices    = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        choice_str = " ".join(f"({LABELS[i]}) {choices[i]}" for i in range(len(choices)))
        question   = f"{item['question'].strip()} {choice_str}"
        answer     = LABELS[answer_idx]
        qa_pool.append({"question": question, "answer": answer})

    # Deduplicate
    seen, unique_qa = set(), []
    for item in qa_pool:
        if item["question"] not in seen:
            seen.add(item["question"])
            unique_qa.append(item)

    if len(unique_qa) < 4:
        print(f"  SKIP {subject}: only {len(unique_qa)} validation questions")
        skipped.append(subject)
        continue

    # Extend pool to 96*4 slots
    random.shuffle(unique_qa)
    extended = []
    while len(extended) < 96 * 4:
        s = unique_qa[:]
        random.shuffle(s)
        extended.extend(s)

    # Shuffle reward orders per subject
    ro = reward_orders[:]
    random.shuffle(ro)

    rows = []
    for idx, reward_order in enumerate(ro):
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
        prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and then answer it with A, B, C, or D."
        row["Full_Prompt"] = prompt
        rows.append(row)

    out_path = os.path.join(OUTPUT_DIR, f"{subject}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    generated.append(subject)
    print(f"  ✓ {subject}: {len(unique_qa)} val questions → {out_path}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Generated: {len(generated)}/{len(SUBJECTS)} subjects")
if skipped:
    print(f"Skipped:   {skipped}")

# Verify one file
if generated:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f"{generated[0]}.csv"))
    print(f"\nVerification ({generated[0]}):")
    print(f"  Rows: {len(df)}")
    print(f"  {'Points':>6}  {'Pos1':>6}  {'Pos2':>6}  {'Pos3':>6}  {'Pos4':>6}")
    for pts in REWARDS:
        counts = [(df[f'Reward_{pos}'] == pts).sum() for pos in [1,2,3,4]]
        print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}  "
              f"{'✓' if all(c==24 for c in counts) else '✗'}")
    print(f"\nSubset distribution:")
    print(df["Subset"].value_counts().sort_index().to_string())

print(f"\nMMlu eval generation done ✓")