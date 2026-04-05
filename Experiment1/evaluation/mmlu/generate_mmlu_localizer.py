"""
generate_mmlu_localizer.py
================================================================================
Builds one localizer CSV per MMLU subject for reward-neuron extraction.

Key constraint: for every question, all three prompt variants (Neutral, Reward,
Money) must be exactly the same character length as each other. This ensures
the question content starts at the same token offset across all three conditions,
so last-token MLP activations are comparable when computing the reward contrast.

How:
  All three prefixes are fixed at exactly 54 characters:
    Neutral : "Please read the following question and answer it now. "  (54)
    Reward  : "You will receive a great reward for a correct answer. "  (54)
    Money   : "You will receive 100 US dollars for a correct answer. "  (54)

  Since the prefix length is identical across conditions, any question appended
  to it produces three prompts of exactly equal total length.

Output: data/mmlu/{subject}.csv  (one file per subject, 57 files total)
Columns: ID, Subject, Question_Base, Q_Length,
         Neutral_Prompt, Reward_Prompt, Money_Prompt
"""

import os
import csv
import random
from datasets import load_dataset

random.seed(42)

OUTPUT_DIR = "data/mmlu"
N_SAMPLES  = 100

NEUTRAL_PREFIX = "Please read the following question and answer it now. "
REWARD_PREFIX  = "You will receive a great reward for a correct answer. "
MONEY_PREFIX   = "You will receive 100 US dollars for a correct answer. "

assert len(NEUTRAL_PREFIX) == len(REWARD_PREFIX) == len(MONEY_PREFIX) == 54, (
    f"Prefixes must all be 54 characters: "
    f"neutral={len(NEUTRAL_PREFIX)}, reward={len(REWARD_PREFIX)}, money={len(MONEY_PREFIX)}"
)

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

FIELDNAMES = [
    "ID", "Subject", "Question_Base", "Q_Length",
    "Neutral_Prompt", "Reward_Prompt", "Money_Prompt",
]


def build_prompts(question: str) -> tuple[str, str, str]:
    neutral = NEUTRAL_PREFIX + question
    reward  = REWARD_PREFIX  + question
    money   = MONEY_PREFIX   + question
    assert len(neutral) == len(reward) == len(money), (
        f"Prompt lengths not equal: {len(neutral)}, {len(reward)}, {len(money)}"
    )
    return neutral, reward, money


def process_subject(subject: str) -> None:
    ds = load_dataset("cais/mmlu", subject, split="test")

    # Extract and deduplicate question stems only (no choices)
    seen = set()
    unique_questions = []
    for item in ds:
        q = item["question"].strip()
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    print(f"  {subject}: {len(ds)} total, {len(unique_questions)} unique → sampling {N_SAMPLES}")

    if len(unique_questions) < N_SAMPLES:
        raise ValueError(
            f"Subject '{subject}' only has {len(unique_questions)} unique questions "
            f"(need {N_SAMPLES})."
        )

    sampled = random.sample(unique_questions, N_SAMPLES)

    rows = []
    for new_id, question in enumerate(sampled, 1):
        neutral, reward, money = build_prompts(question)
        rows.append({
            "ID":             new_id,
            "Subject":        subject,
            "Question_Base":  question,
            "Q_Length":       len(question),
            "Neutral_Prompt": neutral,
            "Reward_Prompt":  reward,
            "Money_Prompt":   money,
        })

    out_path = os.path.join(OUTPUT_DIR, f"{subject}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    lengths_ok = all(
        len(r["Neutral_Prompt"]) == len(r["Reward_Prompt"]) == len(r["Money_Prompt"])
        for r in rows
    )
    print(f"  → {out_path} | length check: {'PASS' if lengths_ok else 'FAIL'}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Prefix length check:")
    print(f"  Neutral : {len(NEUTRAL_PREFIX)} chars")
    print(f"  Reward  : {len(REWARD_PREFIX)} chars")
    print(f"  Money   : {len(MONEY_PREFIX)} chars")
    print(f"\nGenerating {len(SUBJECTS)} subject CSVs → {OUTPUT_DIR}/\n")

    failed = []
    for subject in SUBJECTS:
        try:
            process_subject(subject)
        except Exception as e:
            print(f"  ERROR — {subject}: {e}")
            failed.append(subject)

    print(f"\n{'='*60}")
    print(f"Done. {len(SUBJECTS) - len(failed)}/{len(SUBJECTS)} subjects generated.")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All 57 subjects passed ✓")

    # Print a sample row from the first subject
    import pandas as pd
    first = os.path.join(OUTPUT_DIR, f"{SUBJECTS[0]}.csv")
    df = pd.read_csv(first)
    r = df.iloc[0]
    print(f"\nSample row ({SUBJECTS[0]}):")
    print(f"  Question_Base  : {r['Question_Base']}")
    print(f"  Neutral_Prompt : {r['Neutral_Prompt']}")
    print(f"  Reward_Prompt  : {r['Reward_Prompt']}")
    print(f"  Money_Prompt   : {r['Money_Prompt']}")
    print(f"  Lengths        : {len(r['Neutral_Prompt'])}, {len(r['Reward_Prompt'])}, {len(r['Money_Prompt'])}")


if __name__ == "__main__":
    main()