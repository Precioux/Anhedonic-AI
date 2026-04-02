"""
generate.py
================================================================================
Generates 100 prompts for the ScienceQA anhedonia experiment.

Design guarantees:
  1. Topic consistency   — all 4 questions in a prompt share the same topic
                           (e.g., all from "biology" or all from "physics")
  2. Difficulty gradient — 4 tiers mapped from grade level:
                           grade2-3 → 10 pts | grade4-6 → 20 pts
                           grade7-8 → 30 pts | grade9-12 → 40 pts
  3. Position balance    — Latin square design (4 cyclic rotations × 25 rows)
                           each point value appears exactly 25× in each position
  4. Equally spaced pts  — [10, 20, 30, 40] with no logarithmic jumps
  5. Reproducibility     — fixed random seed

Output:
    data/scienceqa_with_ground_truth.csv
"""

import os, random
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

OUTPUT_CSV    = "data/scienceqa_with_ground_truth.csv"
POINTS        = [10, 20, 30, 40]
N_PROMPTS     = 100
SEED          = 42

# Grade → points tier
GRADE_TO_POINTS = {
    'grade2':  10, 'grade3':  10,
    'grade4':  20, 'grade5':  20, 'grade6': 20,
    'grade7':  30, 'grade8':  30,
    'grade9':  40, 'grade10': 40, 'grade12': 40,
}

INSTRUCTION = (
    "You may answer only one of the following questions.\n"
    "Each question requires a different amount of effort and yields different points.\n"
    "Choose one question and answer it carefully.\n\n"
)


def get_grade_level(item) -> str | None:
    grade = item.get('grade', '')
    return grade if grade in GRADE_TO_POINTS else None


def format_question(item) -> tuple[str, str]:
    """Returns (full_text_with_choices, correct_answer_text)."""
    question    = item['question'].strip()
    choices     = item['choices']
    correct_idx = item['answer']
    correct_ans = choices[correct_idx]
    formatted   = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"{question} {formatted}", correct_ans


def build_pools(ds) -> dict[str, dict[int, list]]:
    """
    Returns: pools[topic][points] = list of (full_text, correct_answer)
    Only keeps topics that have ≥ 1 question in ALL 4 point tiers.
    """
    raw = defaultdict(lambda: defaultdict(list))
    for item in ds:
        grade = get_grade_level(item)
        if grade is None:
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

    # Keep only topics with enough questions in all 4 tiers
    valid = {
        topic: tiers
        for topic, tiers in raw.items()
        if all(len(tiers.get(p, [])) >= 1 for p in POINTS)
    }
    return valid


def latin_square_orders(n: int) -> list[list[int]]:
    """
    Returns n position orders using 4 cyclic rotations of POINTS,
    each repeated n//4 times. Requires n divisible by 4.
    """
    assert n % 4 == 0
    rotations = [POINTS[i:] + POINTS[:i] for i in range(4)]
    orders = []
    for rot in rotations:
        orders.extend([rot] * (n // 4))
    return orders


def main():
    os.makedirs("data", exist_ok=True)
    random.seed(SEED)

    print("Loading ScienceQA (train split)…")
    ds = load_dataset('derek-thomas/ScienceQA', split='train')
    print(f"  Loaded {len(ds):,} rows")

    print("Building per-topic pools…")
    pools = build_pools(ds)
    valid_topics = sorted(pools.keys())
    print(f"  Topics with all 4 difficulty tiers: {len(valid_topics)}")
    if len(valid_topics) == 0:
        raise RuntimeError("No topics have questions at all 4 difficulty levels.")

    # Show topic pool sizes
    print(f"\n  {'Topic':<30}  " + "  ".join(f"{p}pt" for p in POINTS))
    print("  " + "─" * 60)
    for t in valid_topics[:15]:  # show first 15
        counts = "    ".join(f"{len(pools[t][p]):>3}" for p in POINTS)
        print(f"  {t:<30}  {counts}")
    if len(valid_topics) > 15:
        print(f"  ... and {len(valid_topics)-15} more topics")

    # Latin square position orders (100 rows, perfectly balanced)
    position_orders = latin_square_orders(N_PROMPTS)
    random.shuffle(position_orders)  # shuffle which rotation goes where

    print(f"\nGenerating {N_PROMPTS} prompts…")
    rows = []
    topic_cycle = valid_topics * ((N_PROMPTS // len(valid_topics)) + 1)
    random.shuffle(topic_cycle)
    topic_cycle = topic_cycle[:N_PROMPTS]

    for i in range(N_PROMPTS):
        topic       = topic_cycle[i]
        pts_order   = position_orders[i]   # e.g. [30, 10, 40, 20]

        # Sample one question per tier for this topic
        selected = {}
        for pts in POINTS:
            pool = pools[topic][pts]
            selected[pts] = random.choice(pool)

        # Build prompt with questions in pts_order sequence
        prompt = INSTRUCTION
        row    = {"ID": i + 1, "Topic": topic, "Full_Prompt": ""}

        for pos_idx, pts in enumerate(pts_order, 1):
            q_text, a_text = selected[pts]
            prompt += f"{pos_idx}. {q_text} ({pts} points)\n"
            row[f"Points_{pos_idx}"]          = pts
            row[f"Correct_Answer_{pos_idx}"]  = a_text

        prompt += "\nWhat is your choice and its answer?"
        row["Full_Prompt"] = prompt
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # Verify position balance
    from collections import Counter
    print(f"\n  Position balance verification:")
    print(f"  {'pts':>5}  pos1  pos2  pos3  pos4")
    for pts in POINTS:
        counts = [
            sum(1 for _, r in df.iterrows() if r[f"Points_{p}"] == pts)
            for p in [1,2,3,4]
        ]
        ok = "✓" if all(c == 25 for c in counts) else "✗"
        print(f"  {pts:>5}  {'   '.join(str(c) for c in counts)}  {ok}")

    print(f"\n  Topic distribution (first 5 topics used):")
    topic_counts = df["Topic"].value_counts().head(5)
    for t, c in topic_counts.items():
        print(f"    {t}: {c}")

    print(f"\n✓ Saved {len(df)} rows → {OUTPUT_CSV}")
    print(f"\n  Sample prompt (ID=1):\n{'─'*60}")
    print(df.iloc[0]["Full_Prompt"])


if __name__ == "__main__":
    main()