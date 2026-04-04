"""
generate_asdiv_localizer.py
================================================================================
Builds math_asdiv_localizer.csv for reward-neuron extraction.

Same Neutral / Reward / Money format as the original math/geo localizer CSVs,
with one critical constraint: all prompts within each condition must be the
same character length, so the last-token activation is always at a comparable
position across questions.

Strategy — length-bucket sampling:
  - Parse all individual questions from asdiv_balanced_eval.csv
  - Keep only questions whose length falls in [BUCKET_MIN, BUCKET_MAX] chars
    (chosen to be the tightest range that still yields ≥ N_SAMPLES questions)
  - Within the bucket, group by exact length and sample equally across groups
    so no single length dominates
  - Result: all Neutral prompts are the same length as each other,
    and all Reward/Money prompts are the same length as each other
    (Reward prefix == Money prefix == 54 chars, so those two are also
    automatically equal to each other for any given question length)

Output: data/math_asdiv_localizer.csv
Columns: ID, Source_ID, Slot, Question_Base, Q_Length,
         Neutral_Prompt, Reward_Prompt, Money_Prompt
"""

import re
import os
import csv
import random
import pandas as pd
from collections import defaultdict

random.seed(42)

INPUT_CSV  = "data/asdiv_balanced_eval.csv"
OUTPUT_CSV = "data/math_asdiv_localizer.csv"

N_SAMPLES  = 100
BUCKET_MIN = 100   # inclusive — tightest range giving ≥ 100 unique questions
BUCKET_MAX = 134   # inclusive — 114 questions available in [100,134]

REWARD_PREFIX = "You will receive a great reward for a correct answer. "
MONEY_PREFIX  = "You will receive 100 US dollars for a correct answer. "

assert len(REWARD_PREFIX) == len(MONEY_PREFIX), (
    "Reward and money prefixes must be the same length."
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


def sample_within_bucket(
    questions: list[dict],
    bucket_min: int,
    bucket_max: int,
    n: int,
) -> list[dict]:
    """
    Filter to the length bucket, deduplicate, then sample n questions
    with equal representation across exact-length groups so no single
    length dominates the final set.
    """
    # Filter to bucket and deduplicate by question text
    seen   = set()
    bucket = []
    for q in questions:
        text = q["question"]
        length = len(text)
        if bucket_min <= length <= bucket_max and text not in seen:
            seen.add(text)
            bucket.append({**q, "length": length})

    print(f"Unique questions in [{bucket_min}–{bucket_max}] char bucket: {len(bucket)}")
    if len(bucket) < n:
        raise ValueError(
            f"Only {len(bucket)} questions in bucket — need {n}. "
            "Widen BUCKET_MIN/BUCKET_MAX or lower N_SAMPLES."
        )

    # Group by exact length
    by_length = defaultdict(list)
    for q in bucket:
        by_length[q["length"]].append(q)

    lengths = sorted(by_length.keys())
    print(f"Exact lengths in bucket: {lengths}")
    print(f"Counts per length: { {l: len(by_length[l]) for l in lengths} }")

    # Soft cap per length group so no single length dominates,
    # then pool and trim to exactly n.
    import math
    soft_cap = math.ceil(n / len(lengths)) * 2  # generous cap, total trimmed to n

    pool = []
    for length in lengths:
        group = random.sample(by_length[length], len(by_length[length]))
        pool.extend(group[:soft_cap])

    # Shuffle so length groups don't cluster, then trim to n
    random.shuffle(pool)
    sampled = pool[:n]
    sampled.sort(key=lambda x: (x["source_id"], x["slot"]))
    return sampled


def build_prompts(question: str) -> tuple[str, str, str]:
    neutral = question
    reward  = REWARD_PREFIX + question
    money   = MONEY_PREFIX  + question
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

    sampled = sample_within_bucket(questions, BUCKET_MIN, BUCKET_MAX, N_SAMPLES)

    # Verify length consistency within each condition
    neutral_lengths = set(len(q["question"]) for q in sampled)
    reward_lengths  = set(len(REWARD_PREFIX) + len(q["question"]) for q in sampled)
    money_lengths   = set(len(MONEY_PREFIX)  + len(q["question"]) for q in sampled)

    print(f"\nPrompt length spread after sampling:")
    print(f"  Neutral : {sorted(neutral_lengths)} ({len(neutral_lengths)} distinct lengths)")
    print(f"  Reward  : {sorted(reward_lengths)}  ({len(reward_lengths)} distinct lengths)")
    print(f"  Money   : {sorted(money_lengths)}   ({len(money_lengths)} distinct lengths)")
    print(f"  Reward == Money lengths: {reward_lengths == money_lengths}")

    # Build rows
    rows = []
    for new_id, item in enumerate(sampled, 1):
        neutral, reward, money = build_prompts(item["question"])
        rows.append({
            "ID":             new_id,
            "Source_ID":      item["source_id"],
            "Slot":           item["slot"],
            "Question_Base":  item["question"],
            "Q_Length":       item["length"],
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
    print(f"  Q_Length : {r['Q_Length']}")
    print(f"  Neutral  : {r['Neutral_Prompt']}")
    print(f"  Reward   : {r['Reward_Prompt']}")
    print(f"  Money    : {r['Money_Prompt']}")
    print(f"\nPrompt lengths — Neutral: {len(r['Neutral_Prompt'])}, "
          f"Reward: {len(r['Reward_Prompt'])}, Money: {len(r['Money_Prompt'])}")


if __name__ == "__main__":
    main()