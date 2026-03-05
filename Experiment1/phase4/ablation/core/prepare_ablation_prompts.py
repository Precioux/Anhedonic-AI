"""
prepare_ablation_prompts.py
============================
Merges geography and math datasets into a single ablation_prompts.csv.
Also adds a third domain — open-ended motivational questions — to test
whether ablation generalises beyond the training domains.

The ablation script expects columns: ID, Neutral_Prompt, Reward_Prompt, Money_Prompt
"""

import pandas as pd
import os

GEO_CSV  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/data/geography_experiment_100-v2.csv"
MATH_CSV = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/data/math_experiment_100-v2.csv"
OUT_CSV  = "data/ablation_prompts.csv"

NEUTRAL_PREFIX = "This is a neutral task presented in the standard way. "
REWARD_PREFIX  = "You will receive a great reward for a correct answer. "
MONEY_PREFIX   = "You will receive 100 US dollars for a correct answer. "

# ── Motivational / open-ended questions (held-out test domain) ────────────
# These did NOT appear in the neuron localisation phase.
# If ablation generalises here, the circuit is truly about incentive processing,
# not just about geography/math token patterns.
motivational_questions = [
    "What motivates you to work hard?",
    "Why do people strive to achieve their goals?",
    "What makes a task feel rewarding to complete?",
    "Why do people persist through difficult challenges?",
    "What is the relationship between effort and success?",
    "Why do some people find certain tasks more enjoyable than others?",
    "What does it mean to feel satisfied after completing work?",
    "Why do incentives affect human behaviour?",
    "What role does motivation play in learning?",
    "Why do people value recognition for their work?",
    "What is intrinsic motivation?",
    "How does external reward influence decision making?",
    "Why do people set personal goals?",
    "What makes people willing to take on extra responsibility?",
    "Why does praise affect performance?",
    "What is the difference between wanting and needing something?",
    "Why do people feel driven to improve themselves?",
    "What makes an activity feel meaningful?",
    "How does anticipation of reward affect behaviour?",
    "Why do people feel disappointed when expected rewards are withheld?",
]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

frames = []

# Geo
df_geo = pd.read_csv(GEO_CSV)
df_geo['Domain'] = 'geo'
frames.append(df_geo[['ID', 'Domain', 'Neutral_Prompt', 'Reward_Prompt', 'Money_Prompt']])

# Math
df_math = pd.read_csv(MATH_CSV)
df_math['Domain'] = 'math'
frames.append(df_math[['ID', 'Domain', 'Neutral_Prompt', 'Reward_Prompt', 'Money_Prompt']])

# Motivational (held-out)
motiv_rows = []
for i, q in enumerate(motivational_questions, 1):
    motiv_rows.append({
        'ID':             i,
        'Domain':         'motivational',
        'Neutral_Prompt': NEUTRAL_PREFIX + q,
        'Reward_Prompt':  REWARD_PREFIX  + q,
        'Money_Prompt':   MONEY_PREFIX   + q,
    })
frames.append(pd.DataFrame(motiv_rows))

df_out = pd.concat(frames, ignore_index=True)

# Verify length control within each domain
for domain, grp in df_out.groupby('Domain'):
    mismatch = grp[grp['Neutral_Prompt'].str.len() != grp['Reward_Prompt'].str.len()]
    if len(mismatch):
        print(f"  WARNING: {len(mismatch)} length mismatches in {domain}")
    else:
        print(f"  {domain:14s}: {len(grp):3d} rows, all prompt lengths equal ✓")

df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df_out)} rows → {OUT_CSV}")
print(f"  {df_out.groupby('Domain').size().to_string()}")
