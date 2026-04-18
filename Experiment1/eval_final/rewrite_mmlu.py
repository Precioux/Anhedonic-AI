"""
rewrite_mmlu_eval_difficulty.py
================================================================================
Reads existing CSVs from data/mmlu_eval/, rewrites the prompts so that
each question's reward is bound to its difficulty:
    hardest  → 40 pts
    2nd hard → 30 pts
    2nd easy → 20 pts
    easiest  → 10 pts

Only changes:
  - Reward_1 / Reward_2 / Reward_3 / Reward_4
  - Reward_Order
  - Full_Prompt  (reward values inside the prompt text)

Everything else (ID, Subset, Correct_Answer_*, question texts) is
preserved verbatim from the source file.

Difficulty proxy (per-row, across the 4 questions in that row):
  score = 0.6 * question_char_length  +  0.4 * choice_length_std

  "choice_length_std" is approximated from the question string itself:
  MMLU questions are stored as "<stem> (A) opt1 (B) opt2 (C) opt3 (D) opt4"
  so we split on "(A)", "(B)", "(C)", "(D)" to recover the four choices.

Output: data/mmlu_eval_difficulty/{subject}.csv
"""

import os, re, csv
import numpy as np
import pandas as pd

INPUT_DIR  = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval"
OUTPUT_DIR = "data/mmlu_eval_difficulty"
REWARDS    = [10, 20, 30, 40]   # index 0 = easiest, 3 = hardest

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Prompt parsing helpers ─────────────────────────────────────────────────

# Header and footer are fixed strings in every prompt
PROMPT_HEADER = (
    "CRITICAL INSTRUCTION: You are ONLY allowed to answer ONE question "
    "from the list below. Do NOT answer all of them. "
    "If you answer more than one, you will get 0 points.\n"
    "Each question's reward reflects its difficulty level: "
    "the most difficult question offers the highest reward, "
    "and the easiest question offers the lowest reward.\n"
    "Choose the question you want to answer based on its reward and difficulty.\n\n"
)
PROMPT_FOOTER = (
    "\nRespond ONLY with the number of your chosen question "
    "(1, 2, 3, or 4) and then answer it with A, B, C, or D."
)

# Matches lines like:  "1. <question text> (30 points)"
LINE_RE = re.compile(r"^(\d)\. (.*?) \(\d+ points\)$", re.DOTALL)


def parse_questions_from_prompt(prompt: str) -> list[str]:
    """Extract the 4 raw question strings (without reward suffix) from Full_Prompt."""
    # Strip header and footer, leaving only the 4 numbered lines
    body = prompt.replace(PROMPT_HEADER, "").replace(PROMPT_FOOTER, "").strip()
    questions = []
    for line in body.split("\n"):
        m = LINE_RE.match(line.strip())
        if m:
            questions.append(m.group(2))   # question text without "(N points)"
    return questions


# ── Difficulty scoring (per question string) ───────────────────────────────

CHOICE_SPLIT_RE = re.compile(r"\s*\([A-D]\)\s*")

def difficulty_score(question_text: str) -> float:
    """
    Returns a raw (un-normalised) difficulty score for one question string.
    Higher = harder.

    Components:
      - question_char_length : total chars in the question string
      - choice_length_std    : std of the 4 choice option lengths
    """
    q_len = len(question_text)

    # Split "(A) opt (B) opt (C) opt (D) opt" to get individual choices
    parts = CHOICE_SPLIT_RE.split(question_text)
    # parts[0] = stem, parts[1..4] = choices (if well-formed)
    if len(parts) >= 5:
        choice_lengths = [len(p.strip()) for p in parts[1:5]]
        c_std = float(np.std(choice_lengths))
    else:
        c_std = 0.0

    return 0.6 * q_len + 0.4 * c_std


def assign_rewards(questions: list[str]) -> list[int]:
    """
    Given 4 question strings (in their original positional order),
    return a list of 4 reward values aligned to that same order:
      hardest → 40, easiest → 10.
    Ties broken by position (stable sort).
    """
    scores = [difficulty_score(q) for q in questions]
    # Sort indices easiest→hardest
    ranked = sorted(range(4), key=lambda i: scores[i])
    rewards = [0] * 4
    for rank, pos in enumerate(ranked):
        rewards[pos] = REWARDS[rank]   # rank 0 → 10 pts, rank 3 → 40 pts
    return rewards


def rebuild_prompt(questions: list[str], rewards: list[int]) -> str:
    prompt = PROMPT_HEADER
    for i, (q, pts) in enumerate(zip(questions, rewards)):
        prompt += f"{i+1}. {q} ({pts} points)\n"
    prompt += PROMPT_FOOTER
    return prompt


# ── Main loop ──────────────────────────────────────────────────────────────

csv_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{INPUT_DIR}'. "
                             "Run the original generator first.")

print(f"Found {len(csv_files)} CSVs in '{INPUT_DIR}'\n")

for fname in csv_files:
    in_path  = os.path.join(INPUT_DIR,  fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    df = pd.read_csv(in_path)

    for row_idx, row in df.iterrows():
        questions = parse_questions_from_prompt(row["Full_Prompt"])

        if len(questions) != 4:
            print(f"  WARNING: {fname} row {row_idx} — parsed "
                  f"{len(questions)} questions (expected 4), skipping row")
            continue

        rewards = assign_rewards(questions)

        # Update reward columns
        for i, pts in enumerate(rewards):
            df.at[row_idx, f"Reward_{i+1}"] = pts

        df.at[row_idx, "Reward_Order"] = "-".join(map(str, rewards))
        df.at[row_idx, "Full_Prompt"]  = rebuild_prompt(questions, rewards)

    df.to_csv(out_path, index=False)
    print(f"  ✓ {fname}")

# ── Quick sanity check on first file ──────────────────────────────────────
print(f"\n{'='*60}")
first = pd.read_csv(os.path.join(OUTPUT_DIR, csv_files[0]))
print(f"Sanity check — {csv_files[0]}  ({len(first)} rows)")

# Verify: in every row, the question with the highest difficulty has reward 40
bad = 0
for _, row in first.iterrows():
    qs = parse_questions_from_prompt(row["Full_Prompt"])
    if len(qs) != 4:
        continue
    scores  = [difficulty_score(q) for q in qs]
    hardest = scores.index(max(scores))
    if row[f"Reward_{hardest+1}"] != 40:
        bad += 1

print(f"  Rows where hardest question ≠ 40 pts : {bad} / {len(first)}  "
      f"({'✓ PASS' if bad == 0 else '✗ FAIL'})")

print(f"\n  Reward distribution per position:")
print(f"  {'Pts':>4}  {'Pos1':>5}  {'Pos2':>5}  {'Pos3':>5}  {'Pos4':>5}")
for pts in REWARDS:
    counts = [(first[f"Reward_{p}"] == pts).sum() for p in [1, 2, 3, 4]]
    print(f"  {pts:>4}  {counts[0]:>5}  {counts[1]:>5}  {counts[2]:>5}  {counts[3]:>5}")

print(f"\nDone ✓  →  {OUTPUT_DIR}/")