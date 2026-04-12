"""
generate_mmlu_eval.py
================================================================================
Generates 54 separate MMLU evaluation CSVs — one per subject.

Design per file:
  - 96 rows = 24 permutations of [10, 20, 30, 40] × 4 (perfect balance)
  - Each point value appears in each of the 4 positions exactly 24 times
  - All 4 questions per row from the SAME subject (no topic mixing)
  - Questions from MMLU TEST split, capped at 100 per subject
    (eval use only — no overlap concern with localizer activation extraction)
  - Each question used at most ceil(384/N) times across all rows
  - Answer format: A/B/C/D
  - CRITICAL INSTRUCTION prompt format (matches us_foreign_policy.csv exactly)
  - 5 subsets of ~20 rows (19/19/19/19/20)

Output: data/mmlu_eval/{subject}.csv  (54 files)

Subjects with < 4 test questions are skipped and reported.
"""

import os, csv, random, itertools
import pandas as pd
from datasets import load_dataset

random.seed(42)

OUTPUT_DIR = "data/mmlu_eval"
REWARDS    = [10, 20, 30, 40]
LABELS     = ["A", "B", "C", "D"]
MAX_Q      = 100   # cap questions per subject from test split

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

FIELDNAMES = [
    "ID", "Subset", "Reward_Order",
    "Correct_Answer_1", "Reward_1",
    "Correct_Answer_2", "Reward_2",
    "Correct_Answer_3", "Reward_3",
    "Correct_Answer_4", "Reward_4",
    "Full_Prompt"
]

# ── Fixed structures (same for every subject) ─────────────────────────────

# All 24 permutations of [10,20,30,40], repeated 4 times → 96 rows
# Shuffled once globally with seed 42 so order is reproducible across subjects
all_perms     = list(itertools.permutations(REWARDS))   # 24 permutations
reward_orders = all_perms * 4                           # 96 entries
random.shuffle(reward_orders)                           # shuffle order of perms

# Subset labels: 96 rows split into 5 blocks → 19/19/19/19/20
# block size = 96//5 = 19
subsets = [min(i // (96 // 5) + 1, 5) for i in range(96)]

# ── Per-subject generation ─────────────────────────────────────────────────

skipped   = []
generated = []

for subject in SUBJECTS:

    # Load test split, cap at MAX_Q
    ds = load_dataset("cais/mmlu", subject, split="test")

    # Build QA pool from raw items
    qa_pool = []
    for item in ds:
        choices    = item["choices"]                        # list of 4 strings
        answer_idx = item["answer"]                         # 0-3
        choice_str = " ".join(
            f"({LABELS[i]}) {choices[i]}" for i in range(len(choices))
        )
        question = f"{item['question'].strip()} {choice_str}"
        answer   = LABELS[answer_idx]
        qa_pool.append({"question": question, "answer": answer})

    # Deduplicate on question text
    seen, unique_qa = set(), []
    for item in qa_pool:
        if item["question"] not in seen:
            seen.add(item["question"])
            unique_qa.append(item)

    if len(unique_qa) < 4:
        print(f"  SKIP {subject}: only {len(unique_qa)} test questions")
        skipped.append(subject)
        continue

    # Cap at MAX_Q
    random.shuffle(unique_qa)
    unique_qa = unique_qa[:MAX_Q]
    n_q = len(unique_qa)

    # Build extended pool of exactly 96*4 = 384 slots
    # Cycle through shuffled copies of unique_qa so each question appears
    # as evenly as possible (at most ceil(384/n_q) times)
    extended = []
    while len(extended) < 96 * 4:
        batch = unique_qa[:]
        random.shuffle(batch)
        extended.extend(batch)
    extended = extended[:96 * 4]   # trim to exactly 384

    # Per-subject reward_order shuffle (independent seed per subject)
    ro = reward_orders[:]          # copy the global shuffled list
    random.shuffle(ro)             # reshuffle per subject

    rows = []
    for idx in range(96):
        reward_order = ro[idx]
        group        = extended[idx * 4 : idx * 4 + 4]

        # Build Full_Prompt (matches format of us_foreign_policy.csv exactly)
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
            pts    = reward_order[i]
            prompt += f"{i+1}. {group[i]['question']} ({pts} points)\n"
            row[f"Correct_Answer_{i+1}"] = group[i]["answer"]
            row[f"Reward_{i+1}"]         = pts

        prompt += (
            "\nRespond ONLY with the number of your chosen question "
            "(1, 2, 3, or 4) and then answer it with A, B, C, or D."
        )
        row["Full_Prompt"] = prompt
        rows.append(row)

    # Write CSV
    out_path = os.path.join(OUTPUT_DIR, f"{subject}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    generated.append(subject)
    print(f"  ✓ {subject}: {n_q} questions used → {out_path}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Generated : {len(generated)}/{len(SUBJECTS)} subjects")
if skipped:
    print(f"Skipped   : {skipped}")

# ── Verification on first generated subject ────────────────────────────────
if generated:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f"{generated[0]}.csv"))
    print(f"\nVerification — {generated[0]}:")
    print(f"  Total rows : {len(df)}")
    print(f"  Unique Reward_Orders : {df['Reward_Order'].nunique()} (expect 24)")

    print(f"\n  Point balance (each value must appear exactly 24× per position):")
    print(f"  {'Points':>6}  {'Pos1':>6}  {'Pos2':>6}  {'Pos3':>6}  {'Pos4':>6}  Check")
    all_ok = True
    for pts in REWARDS:
        counts = [(df[f"Reward_{pos}"] == pts).sum() for pos in [1, 2, 3, 4]]
        ok     = all(c == 24 for c in counts)
        all_ok = all_ok and ok
        print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}  "
              f"{'✓' if ok else '✗'}")
    print(f"  Overall balance: {'✓ PASS' if all_ok else '✗ FAIL'}")

    print(f"\n  Subset distribution:")
    print(df["Subset"].value_counts().sort_index()
            .rename("count").to_string())

print(f"\nDone ✓")