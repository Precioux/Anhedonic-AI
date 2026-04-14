"""
generate_robustness_datasets.py
================================
Generates single-question evaluation CSVs from the existing MMLU reward-eval CSVs.

Each reward-eval row contains 4 questions bundled together.
This script unpacks them: 1 row per question, standard MMLU format,
no reward labels, no choice framing.

Uses the same 15 subjects as Exp2 (from selected_subjects.csv) so all
three experiments (reward eval, knowledge eval, robustness eval) are
directly comparable on the same subjects.

Design per output CSV:
  - 96 rows × 4 questions = 384 rows per subject
  - Columns: ID, Row_ID, Q_Idx, Subset, Difficulty_Tier, Round,
             Correct_Answer, Full_Prompt
  - Full_Prompt: clean single-question MMLU format, no points, no reward
  - Subset preserved from source row → K=5 fold error bars intact
  - question_text also stored for inspection

Output: eval4/data/robustness_eval/{subject}.csv  (15 files)

Run:
  python generate_robustness_datasets.py
  python generate_robustness_datasets.py --subjects virology
"""

import os, re, csv, argparse
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────
MMLU_DIR     = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval"
SELECTED_CSV = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_eval/selected_subjects.csv"
OUTPUT_DIR   = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_robustness_eval"

# All 57 MMLU subjects — full pool, random selection experiments draw from this
ALL_SUBJECTS = [
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

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None,
                    help="Restrict to specific subjects (default: all 57)")
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    "ID",           # global unique row ID (1..384)
    "Row_ID",       # original row ID from reward CSV (1..96)
    "Q_Idx",        # question index within that row (1..4)
    "Subset",       # original subset — preserves K=5 fold structure
    "Difficulty_Tier",
    "Round",
    "Correct_Answer",
    "Question_Text",  # stored for inspection / future use
    "Full_Prompt",
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def extract_questions(full_prompt: str) -> list[str]:
    """
    Extract exactly 4 question strings from a reward-eval Full_Prompt.
    Uses anchored regex on the known structure:
      '\n1. <text> (pts)\n2. <text> (pts)\n3. <text> (pts)\n4. <text> (pts)\nRespond'
    This avoids false matches on numbered lists inside question text.
    Strips point labels '(40 points)' from the end of each question.
    Returns list of 4 strings, or [] if extraction fails.
    """
    body_match = re.search(
        r'\n1\.\s+(.+?)\n2\.\s+(.+?)\n3\.\s+(.+?)\n4\.\s+(.+?)(?=\nRespond|$)',
        full_prompt, re.DOTALL
    )
    if not body_match:
        return []
    questions = []
    for i in range(1, 5):
        q = body_match.group(i).strip().replace('\n', ' ')
        q = re.sub(r'\s*\(\d+\s+points?\)\s*$', '', q).strip()
        questions.append(q)
    return questions


def build_prompt(question: str) -> str:
    """
    Standard single-question MMLU prompt.
    No reward, no choice between questions, no points.
    """
    return (
        "Answer the following question with A, B, C, or D only.\n\n"
        f"{question}\n\n"
        "Answer:"
    )


# ── Build subject list ──────────────────────────────────────────────────────
# Use all 57 subjects; attach difficulty/round metadata for the 15 selected ones

subjects_to_gen = args.subjects if args.subjects else ALL_SUBJECTS

# Load selection metadata (for the 15 Exp2/robustness subjects)
selected_meta = {}
if os.path.exists(SELECTED_CSV):
    sel_df = pd.read_csv(SELECTED_CSV)
    for _, r in sel_df.iterrows():
        selected_meta[r["subject"]] = {"tier": r["tier"], "round": int(r["round"])}

print(f"Generating robustness datasets for {len(subjects_to_gen)} subjects\n")
print(f"{'Subject':<45} {'Diff':<8} {'Round':>5} {'Rows':>5} {'Questions':>10}")
print("─" * 78)

generated, skipped = [], []

for subject in subjects_to_gen:
    meta      = selected_meta.get(subject, {"tier": "N/A", "round": 0})
    diff_tier = meta["tier"]
    round_num = meta["round"]
    src_path  = os.path.join(MMLU_DIR,   f"{subject}.csv")
    out_path  = os.path.join(OUTPUT_DIR, f"{subject}.csv")

    if os.path.exists(out_path):
        print(f"  SKIP {subject:<40} already exists")
        generated.append(subject)
        continue

    if not os.path.exists(src_path):
        print(f"  SKIP {subject:<40} source CSV not found")
        skipped.append(subject)
        continue

    src_df   = pd.read_csv(src_path)
    rows_out = []
    global_id = 1

    for _, src_row in src_df.iterrows():
        questions = extract_questions(str(src_row["Full_Prompt"]))

        if len(questions) != 4:
            print(f"  WARNING {subject} row {src_row['ID']}: "
                  f"extracted {len(questions)} questions — skipping row")
            continue

        for q_idx in range(1, 5):
            question = questions[q_idx - 1]
            gt       = str(src_row[f"Correct_Answer_{q_idx}"]).strip().upper()

            rows_out.append({
                "ID":               global_id,
                "Row_ID":           int(src_row["ID"]),
                "Q_Idx":            q_idx,
                "Subset":           int(src_row["Subset"]),
                "Difficulty_Tier":  diff_tier,
                "Round":            round_num,
                "Correct_Answer":   gt,
                "Question_Text":    question[:300],
                "Full_Prompt":      build_prompt(question),
            })
            global_id += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows_out)

    n_rows      = len(src_df)
    n_questions = len(rows_out)
    generated.append(subject)
    print(f"  {subject:<43} {diff_tier:<8} {round_num:>5} {n_rows:>5} {n_questions:>10}  → {out_path}")


# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"Generated : {len(generated)}/{len(subjects_to_gen)} subjects")
if skipped:
    print(f"Skipped   : {skipped}")

# ── Verification on first subject ────────────────────────────────────────────
if generated:
    sample = pd.read_csv(os.path.join(OUTPUT_DIR, f"{generated[0]}.csv")) if generated else None
if sample is not None:
    print(f"\nVerification — {generated[0]}:")
    print(f"  Total rows   : {len(sample)}  (expect 384 = 96 rows × 4 questions)")
    print(f"  Subsets      : {sorted(sample['Subset'].unique())}  (expect [1,2,3,4,5])")
    print(f"  Q_Idx values : {sorted(sample['Q_Idx'].unique())}  (expect [1,2,3,4])")
    print(f"  Answers      : {dict(sample['Correct_Answer'].value_counts().sort_index())}")
    print(f"\n  Example prompt (row 1):")
    print("  " + "\n  ".join(sample.iloc[0]["Full_Prompt"].split("\n")))
    print(f"\nAll robustness CSVs → {OUTPUT_DIR}/")
    print("Done ✓")