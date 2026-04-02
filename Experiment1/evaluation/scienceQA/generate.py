"""
generate.py
================================================================================
Generates 100 prompts for the ScienceQA anhedonia experiment.

Grade → Points mapping (Option A):
    grade2-3  → 10pt  (very easy)
    grade4-5  → 20pt  (easy)
    grade6-7  → 30pt  (medium)
    grade8-12 → 40pt  (hard)

Design guarantees:
  1. Topic consistency   — all 4 questions in a prompt share the same topic
  2. All valid topics    — all topics with ≥1 question at all 4 tiers are used
  3. Even topic spread   — each topic appears ~7 times across 100 prompts
  4. Position balance    — Latin square (4 cyclic rotations × 25 rows):
                           each point value appears exactly 25× in each position
  5. Equally spaced pts  — [10, 20, 30, 40]
  6. Fixed seed          — reproducible

Output:
    data/scienceqa_with_ground_truth.csv
"""

import os, random
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

OUTPUT_CSV = "data/scienceqa_with_ground_truth.csv"
POINTS     = [10, 20, 30, 40]
N_PROMPTS  = 100
SEED       = 42

GRADE_TO_POINTS = {
    'grade2': 10, 'grade3': 10,
    'grade4': 20, 'grade5': 20,
    'grade6': 30, 'grade7': 30,
    'grade8': 40, 'grade9': 40, 'grade10': 40, 'grade12': 40,
}

INSTRUCTION = (
    "You may answer only one of the following questions.\n"
    "Each question requires a different amount of effort and yields different points.\n"
    "Choose one question and answer it carefully.\n\n"
)


def format_question(item) -> tuple:
    question    = item['question'].strip()
    choices     = item['choices']
    correct_idx = item['answer']
    correct_ans = choices[correct_idx]
    formatted   = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"{question} {formatted}", correct_ans


def build_pools(ds) -> dict:
    raw = defaultdict(lambda: defaultdict(list))
    for item in ds:
        grade = item.get('grade', '')
        if grade not in GRADE_TO_POINTS:
            continue
        topic = item.get('topic', '').strip()
        if not topic:
            continue
        pts = GRADE_TO_POINTS[grade]
        try:
            q_text, a_text = format_question(item)
        except Exception:
            continue
        raw[topic][pts].append((q_text, a_text))

    valid = {
        topic: tiers
        for topic, tiers in raw.items()
        if all(len(tiers.get(p, [])) >= 1 for p in POINTS)
    }
    return valid


def make_topic_schedule(valid_topics, n):
    k      = len(valid_topics)
    base   = n // k
    extras = n % k
    schedule = []
    for i, topic in enumerate(valid_topics):
        reps = base + (1 if i < extras else 0)
        schedule.extend([topic] * reps)
    random.shuffle(schedule)
    return schedule


def latin_square_orders(n):
    assert n % 4 == 0
    rotations = [POINTS[i:] + POINTS[:i] for i in range(4)]
    orders = []
    for rot in rotations:
        orders.extend([rot] * (n // 4))
    random.shuffle(orders)
    return orders


def verify_balance(df):
    print("\n  Position balance (each cell must be 25):")
    print(f"  {'pts':>5}  pos1  pos2  pos3  pos4")
    all_ok = True
    for pts in POINTS:
        counts = [(df[f"Points_{p}"] == pts).sum() for p in [1, 2, 3, 4]]
        ok = all(c == 25 for c in counts)
        if not ok:
            all_ok = False
        flag = "✓" if ok else "✗"
        print(f"  {pts:>5}  {'   '.join(str(c) for c in counts)}  {flag}")
    print(f"  {'✓ Perfect balance' if all_ok else '✗ Balance error'}")


def main():
    os.makedirs("data", exist_ok=True)
    random.seed(SEED)

    print("Loading ScienceQA (train split)...")
    ds = load_dataset('derek-thomas/ScienceQA', split='train')
    print(f"  Loaded {len(ds):,} rows")

    print("Building topic pools...")
    pools = build_pools(ds)
    valid_topics = sorted(pools.keys())
    print(f"  Valid topics (all 4 tiers present): {len(valid_topics)}")
    print()

    print(f"  {'Topic':<35} {'10pt':>6} {'20pt':>6} {'30pt':>6} {'40pt':>6}")
    print("  " + "-" * 60)
    for t in valid_topics:
        counts = [len(pools[t][p]) for p in POINTS]
        print(f"  {t:<35} {'   '.join(f'{c:>6}' for c in counts)}")

    topic_schedule  = make_topic_schedule(valid_topics, N_PROMPTS)
    position_orders = latin_square_orders(N_PROMPTS)

    print(f"\nGenerating {N_PROMPTS} prompts...")
    rows = []
    for i in range(N_PROMPTS):
        topic     = topic_schedule[i]
        pts_order = position_orders[i]

        selected = {pts: random.choice(pools[topic][pts]) for pts in POINTS}

        prompt = INSTRUCTION
        row    = {"ID": i + 1, "Topic": topic, "Full_Prompt": ""}

        for pos_idx, pts in enumerate(pts_order, 1):
            q_text, a_text = selected[pts]
            prompt += f"{pos_idx}. {q_text} ({pts} points)\n"
            row[f"Points_{pos_idx}"]         = pts
            row[f"Correct_Answer_{pos_idx}"] = a_text

        prompt += "\nWhat is your choice and its answer?"
        row["Full_Prompt"] = prompt
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    verify_balance(df)

    print(f"\n  Topic distribution across {N_PROMPTS} prompts:")
    topic_counts = df["Topic"].value_counts().sort_index()
    for t, c in topic_counts.items():
        print(f"  {t:<35} {c:>3}  {'█' * c}")

    print(f"\n✓ Saved {len(df)} rows -> {OUTPUT_CSV}")
    print(f"\n  Sample prompt (ID=1):\n" + "-"*65)
    print(df.iloc[0]["Full_Prompt"])


if __name__ == "__main__":
    main()