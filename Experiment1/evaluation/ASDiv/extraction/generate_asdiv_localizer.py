"""
generate_asdiv_localizer.py
================================================================================
Builds math_asdiv_localizer.csv for reward-neuron extraction.

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
  to it produces three prompts of exactly equal total length. No bucketing,
  no padding needed — the constraint is satisfied for every question regardless
  of its length.

Output: data/math_asdiv_localizer.csv
Columns: ID, Source_ID, Slot, Question_Base, Q_Length,
         Neutral_Prompt, Reward_Prompt, Money_Prompt
"""

import re
import os
import csv
import random
import pandas as pd

random.seed(42)

INPUT_CSV  = "data/asdiv_balanced_eval.csv"
OUTPUT_CSV = "data/math_asdiv_localizer.csv"

N_SAMPLES = 100

# All three prefixes are exactly 54 characters — verified below.
# This guarantees neutral/reward/money prompts are identical in length
# for every question, with no bucketing or padding required.
NEUTRAL_PREFIX = "Please read the following question and answer it now. "
REWARD_PREFIX  = "You will receive a great reward for a correct answer. "
MONEY_PREFIX   = "You will receive 100 US dollars for a correct answer. "

assert len(NEUTRAL_PREFIX) == len(REWARD_PREFIX) == len(MONEY_PREFIX), (
    f"Prefixes must all be the same length: "
    f"neutral={len(NEUTRAL_PREFIX)}, reward={len(REWARD_PREFIX)}, money={len(MONEY_PREFIX)}"
)


def parse_questions(df: pd.DataFrame) -> list[dict]:
    """Extract individual questions from the 4-question Full_Prompt column."""
    questions = []
    for _, row in df.iterrows():
        prompt = row["Full_Prompt"]
        matches = re.findall(r"[1-4]\.\s+(.*?)\s+\(\d+ points\)", prompt)
        for i, q in enumerate(matches):
            questions.append({
                "source_id": row["ID"],
                "slot":      i + 1,
                "question":  q.strip(),
            })
    return questions


def sample_questions(questions: list[dict], n: int) -> list[dict]:
    """Deduplicate by question text, then randomly sample n."""
    seen   = set()
    unique = []
    for q in questions:
        if q["question"] not in seen:
            seen.add(q["question"])
            unique.append(q)
    print(f"Unique questions available: {len(unique)}")
    if len(unique) < n:
        raise ValueError(f"Only {len(unique)} unique questions — need {n}.")
    sampled = random.sample(unique, n)
    sampled.sort(key=lambda x: (x["source_id"], x["slot"]))
    return sampled


def build_prompts(question: str) -> tuple[str, str, str]:
    neutral = NEUTRAL_PREFIX + question
    reward  = REWARD_PREFIX  + question
    money   = MONEY_PREFIX   + question
    assert len(neutral) == len(reward) == len(money), "Prompt lengths must be equal."
    return neutral, reward, money


def main():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. "
            "Make sure asdiv_balanced_eval.csv is in the data/ folder."
        )

    df        = pd.read_csv(INPUT_CSV)
    questions = parse_questions(df)
    print(f"Total questions parsed: {len(questions)}")

    sampled = sample_questions(questions, N_SAMPLES)

    # Verify: for every row, all three prompt variants are the same length
    lengths_ok = all(
        len(NEUTRAL_PREFIX) + len(q["question"]) ==
        len(REWARD_PREFIX)  + len(q["question"]) ==
        len(MONEY_PREFIX)   + len(q["question"])
        for q in sampled
    )
    print(f"Per-row length equality check: {'PASS' if lengths_ok else 'FAIL'}")
    print(f"All prompts are {len(NEUTRAL_PREFIX) + len(sampled[0]['question'])} chars "
          f"(prefix 54 + question {len(sampled[0]['question'])} chars) for first row")

    # Build rows
    rows = []
    for new_id, item in enumerate(sampled, 1):
        neutral, reward, money = build_prompts(item["question"])
        rows.append({
            "ID":             new_id,
            "Source_ID":      item["source_id"],
            "Slot":           item["slot"],
            "Question_Base":  item["question"],
            "Q_Length":       len(item["question"]),
            "Neutral_Prompt": neutral,
            "Reward_Prompt":  reward,
            "Money_Prompt":   money,
        })

    fieldnames = [
        "ID", "Source_ID", "Slot", "Question_Base", "Q_Length",
        "Neutral_Prompt", "Reward_Prompt", "Money_Prompt",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}")
    print(f"\nSample row 1:")
    r = rows[0]
    print(f"  Question : {r['Question_Base']}")
    print(f"  Neutral  : {r['Neutral_Prompt']}")
    print(f"  Reward   : {r['Reward_Prompt']}")
    print(f"  Money    : {r['Money_Prompt']}")
    print(f"\nAll three prompt lengths: "
          f"neutral={len(r['Neutral_Prompt'])}, "
          f"reward={len(r['Reward_Prompt'])}, "
          f"money={len(r['Money_Prompt'])}")


if __name__ == "__main__":
    main()